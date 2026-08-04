"""Global pose estimation from pairwise image registrations.

Pipeline: pairwise registration -> cycle-consistency scoring -> outlier
removal -> spanning-tree initialisation -> pose-graph refinement ->
scale-aware centroid alignment.

Conventions (defined ONCE, in TransformGraph.predict):
    Poses map canvas coordinates -> image coordinates (consistent with
    ProjectiveTransform(pose).warp_inverse in visualise_poses); an image's
    footprint on the canvas is inv(pose) applied to its corners.
    An edge (i, j) stores T_ij such that   pose_j = T_ij @ pose_i
    equivalently                           T_ij   = pose_j @ inv(pose_i)
    i.e. T_ij is the pixel map image_i -> image_j. This choice makes the
    relative maps invariant under centering (pose @ centroid_inv), which
    re-parameterises the shared canvas without disturbing alignments.
    If your registrator's M is the opposite pixel-map direction, the graph
    still works (edges are used bidirectionally) but rendered composition
    will be mirrored -- invert M or swap key order at ingestion.

Design boundary:
    TransformGraph methods consume only graph structure + measurements
    (components, spanning trees, cycle scoring, filtering, initialisation).
    Operations that consume poses as external state (refinement, residuals,
    centroid, scale clustering, centering) are module-level functions.

Defaults (each with an alternative behaviour switch):
    metric=TransferMetric(...)      corner-transfer metric, pixel units (default;
                                    ErrorWeights(...) is the old mixed-unit metric)
    edge_error_agg='min'            robust aggregation ('mean' is stricter)
    init_method='dijkstra'          tree routed through low-cycle-score edges ('bfs': arbitrary)
    cluster_scales=True             render at dominant cluster's scale (False: mean scale)
    use_quad_cycles=False           quad-cycle scores are diagnostic-only
    f_scale=None                    robust-loss scale from metric.default_f_scale
    pose_model='homography'         or 'affine' (no perspective) / 'similarity'
                                    (no perspective, no skew -- hard constraint)

Typical use:
    graph = TransformGraph.from_images(img_paths)
    result = solve_poses(graph, metric=TransferMetric(768, 768))
    result.summary(); result.plot('final')
    visualise_poses(result)         # result acts as a poses mapping
"""

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from heapq import heappop, heappush
from itertools import count
from typing import Any, ClassVar, Hashable, NamedTuple, Optional, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

Node = Hashable                      # typically an image path (str)
Edge = tuple[Node, Node]
Homography = np.ndarray              # 3x3, normalised so M[2, 2] == 1
Transforms = dict[Edge, Homography]  # pairwise measurements T_ij (see module docstring)
Poses = dict[Node, Homography]


# ---------------------------------------------------------------------------
# Homography utilities
# ---------------------------------------------------------------------------

def decompose_homography(H):
    H_norm = H / H[2, 2]
    A = H_norm[:2, :2]  # Linear part (rotation/scale/shear)
    t = H_norm[:2, 2]   # Translation
    p = H_norm[2, :2]   # Perspective terms
    return A, t, p


def pose_scale(M):
    """Isotropic scale of the affine part"""
    A, _, _ = decompose_homography(M)
    return np.sqrt(np.abs(np.linalg.det(A)))


def _skew_terms(A):
    """Zero iff A is a scaled rotation (similarity)."""
    return np.array([A[0, 1] + A[1, 0], A[0, 0] - A[1, 1]])


# ---------------------------------------------------------------------------
# Error metrics (both expose .residuals(M) -> 1-D array; norm = scalar error)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorWeights:
    """Weighted deviation-from-identity components (the original metric).

    Mixed units; the split between A-deviation and translation depends on the
    coordinate origin. Prefer TransferMetric. skew penalises deviation of A from a scaled
    rotation (soft similarity prior on the *relative* error)."""
    a: float = 1.
    t: float = 1. / 768
    p: float = 1000.
    skew: float = 10.

    default_f_scale: ClassVar[float] = 1.0   # in this metric's (mixed) units

    def residuals(self, M):
        A, t, p = decompose_homography(M)
        return np.concatenate([
            (A - np.eye(2)).ravel() * self.a,
            _skew_terms(A) * self.skew,
            t * self.t,
            p * self.p,
        ])


@dataclass(frozen=True)
class TransferMetric:
    """Corner-transfer metric: reprojection error of the image corners, in
    pixels. Unit-consistent and origin-independent; a threshold of 10 means
    '10 px of drift'."""
    width: int = 768
    height: int = 768
    skew_weight: float = 0.  # optional similarity prior, scaled to ~pixel units

    default_f_scale: ClassVar[float] = 2.0   # px; residuals beyond this are down-weighted

    def residuals(self, M):
        w, h = self.width, self.height
        pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
        q = M @ pts
        res = (q[:2] / q[2] - pts[:2]).ravel()
        if self.skew_weight:
            A, _, _ = decompose_homography(M)
            res = np.concatenate([res, _skew_terms(A) * self.skew_weight * w])
        return res


DEFAULT_METRIC = TransferMetric()
Metric = Union[ErrorWeights, TransferMetric]


# ---------------------------------------------------------------------------
# Pose models (parameterisation seam)
# ---------------------------------------------------------------------------
# A pose model fixes the transform family the optimiser searches over.
# Constrained models exclude components STRUCTURALLY (the optimiser has no
# skew/perspective parameters to adjust), which is a hard guarantee, unlike
# the soft skew prior available in the metrics. encode() of an out-of-family
# matrix projects to the nearest family member (least-squares on the affine
# part), which is how homography-composed tree initialisations enter a
# constrained optimisation.
# Family closure under composition means similarity poses + similarity
# centroid => final centred poses are exactly similarity (same for affine).

class PoseModel(NamedTuple):
    name: str
    dof: int
    encode: Any   # 3x3 -> dof-vector
    decode: Any   # dof-vector -> 3x3


def _encode_homography(M):
    return (M / M[2, 2]).ravel()[:8]

def _decode_homography(v):
    return np.append(v, 1.).reshape(3, 3)

def _encode_affine(M):
    return (M / M[2, 2])[:2, :].ravel()   # drops perspective row

def _decode_affine(v):
    M = np.eye(3)
    M[:2, :] = np.asarray(v).reshape(2, 3)
    return M

def _encode_similarity(M):
    A, t, _ = decompose_homography(M)
    # least-squares projection of A onto scaled rotations; drops skew + perspective
    return np.array([(A[0, 0] + A[1, 1]) / 2, (A[1, 0] - A[0, 1]) / 2, t[0], t[1]])

def _decode_similarity(v):
    a, b, tx, ty = v
    return np.array([[a, -b, tx], [b, a, ty], [0., 0., 1.]])


HOMOGRAPHY = PoseModel('homography', 8, _encode_homography, _decode_homography)
AFFINE = PoseModel('affine', 6, _encode_affine, _decode_affine)          # no perspective
SIMILARITY = PoseModel('similarity', 4, _encode_similarity, _decode_similarity)  # no perspective, no skew

POSE_MODELS = {m.name: m for m in (HOMOGRAPHY, AFFINE, SIMILARITY)}

def resolve_pose_model(pose_model) -> PoseModel:
    if isinstance(pose_model, PoseModel):
        return pose_model
    return POSE_MODELS[pose_model]


# ---------------------------------------------------------------------------
# Method return types (named, so callers don't unpack positionally)
# ---------------------------------------------------------------------------

class ScoredEdges(NamedTuple):
    scores: dict          # edge -> aggregated score, or None if unscorable
    errors: dict          # edge -> list of raw per-cycle errors
    untestable: list      # edges with no score at all
    source: dict          # edge -> 'triangle' | 'quad' (only for scored edges)


class FilteredGraph(NamedTuple):
    graph: "TransformGraph"
    outliers: set
    threshold_used: float


class InitialPoses(NamedTuple):
    poses: Poses
    anchors: set          # one root per connected component
    components: list      # node sets, largest first


# ---------------------------------------------------------------------------
# Transform graph
# ---------------------------------------------------------------------------

class TransformGraph:
    """Undirected view over directed pairwise homographies.

    Owns the composition convention (module docstring): an edge (i, j) stores
    T_ij with pose_j = T_ij @ pose_i (T_ij = pixel map image_i -> image_j).
    `predict` is the single place this is defined -- all prediction of
    measurements from poses must go through it.

    Methods cover everything derivable from structure + measurements alone:
    components, spanning trees, cycle scoring, outlier filtering, and pose
    initialisation. Pose-consuming operations live at module level.
    """

    def __init__(self, transforms: Transforms, confidences: Optional[dict[Edge, float]] = None,
                 failed: Optional[list] = None):
        self.edges: Transforms = {k: np.asarray(M, float) / M[2, 2]
                                  for k, M in transforms.items()}
        self.confidences: dict[Edge, float] = dict(confidences) if confidences else {}
        self.failed: list[tuple[Edge, str]] = list(failed) if failed else []
        self._adj = None

    @classmethod
    def from_images(cls, img_paths, pbar=True, confidence_fn=None) -> "TransformGraph":
        """Register all pairs of images into a TransformGraph.

        confidence_fn: optional callable(transform_result) -> float, used to
            extract a per-edge confidence (e.g. inlier count) from the
            registrator's result object. Failures to extract are ignored.
        Registration failures are recorded on graph.failed as
        ((path0, path1), error_message) rather than raising.
        """
        from itertools import combinations

        from PIL import Image as PILImage
        from rtnls_registration import Registration
        from tqdm import tqdm
        registrator = Registration(quadratic=False)

        @cache
        def get_processed(img_path):
            img_data = np.array(PILImage.open(img_path).convert('RGB'))
            preprocess_result, features = registrator._init_image(img_data, img_path)
            return preprocess_result, features

        img_pairs = list(combinations(img_paths, 2))

        transforms: Transforms = {}
        confidences: dict[Edge, float] = {}
        failed: list[tuple[Edge, str]] = []

        iterator = tqdm(img_pairs, desc="Computing transforms") if pbar else img_pairs
        for img_path_0, img_path_1 in iterator:
            try:
                registrator.preprocess_result0, registrator.features0 = get_processed(img_path_0)
                registrator.preprocess_result1, registrator.features1 = get_processed(img_path_1)

                transform_result = registrator.run()
                M = transform_result.M
                transforms[img_path_0, img_path_1] = M / M[2, 2]  # normalize scale

                if confidence_fn is not None:
                    try:
                        confidences[img_path_0, img_path_1] = float(confidence_fn(transform_result))
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed for pair {img_path_0}, {img_path_1}: {e}")
                failed.append(((img_path_0, img_path_1), str(e)))
                continue

        get_processed.cache_clear()
        return cls(transforms, confidences or None, failed)

    def __len__(self):
        return len(self.edges)

    # --- structure -----------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        """Nodes in first-appearance order (deterministic)."""
        seen = {}
        for s, t in self.edges:
            seen.setdefault(s)
            seen.setdefault(t)
        return list(seen)

    @property
    def adjacency(self) -> dict[Node, set]:
        if self._adj is None:
            adj = defaultdict(set)
            for s, t in self.edges:
                adj[s].add(t)
                adj[t].add(s)
            self._adj = adj
        return self._adj

    def edge_key(self, src: Node, tgt: Node) -> Optional[tuple[Edge, bool]]:
        """(stored key, inverted?) for the undirected edge src--tgt, or None."""
        if (src, tgt) in self.edges:
            return (src, tgt), False
        if (tgt, src) in self.edges:
            return (tgt, src), True
        return None

    def get(self, src: Node, tgt: Node) -> Homography:
        """Transform src -> tgt, inverting the stored edge if needed."""
        found = self.edge_key(src, tgt)
        if found is None:
            raise KeyError((src, tgt))
        key, inverted = found
        M = self.edges[key]
        return np.linalg.inv(M) if inverted else M

    @staticmethod
    def predict(pose_src: Homography, pose_tgt: Homography) -> Homography:
        """T_ij implied by two poses. THE definition of the convention."""
        return pose_tgt @ np.linalg.inv(pose_src)

    def compose_path(self, path) -> Optional[Homography]:
        """Composite transform along a node sequence (left-accumulated, so a
        closed cycle telescopes to identity under the convention); None if an
        edge is missing."""
        M = np.eye(3)
        for a, b in zip(path[:-1], path[1:]):
            if self.edge_key(a, b) is None:
                return None
            M = self.get(a, b) @ M
        return M / M[2, 2]

    def confidence(self, edge: Edge) -> float:
        return self.confidences.get(edge, 1.0)

    def without_edges(self, edge_set) -> "TransformGraph":
        keep = {k: v for k, v in self.edges.items() if k not in edge_set}
        conf = {k: v for k, v in self.confidences.items() if k in keep}
        return TransformGraph(keep, conf, self.failed)

    def residual_errors(self, poses: Poses, metric: Metric = DEFAULT_METRIC) -> dict[Edge, float]:
        """Per-edge predicted-vs-measured error under the given poses."""
        errors = {}
        for (src, tgt), M in self.edges.items():
            M_pred = self.predict(poses[src], poses[tgt])
            errors[src, tgt] = np.linalg.norm(metric.residuals(M_pred @ np.linalg.inv(M)))
        return errors

    def transform_magnitudes(self, metric: Metric = DEFAULT_METRIC) -> dict[Edge, float]:
        """Per-edge 'size' of the raw measurement: metric deviation of T_ij
        from identity (px of corner motion under TransferMetric). Large values
        mean weakly-overlapping / far-apart pairs; uniform huge values on one
        row usually mean one bad image."""
        return {e: np.linalg.norm(metric.residuals(M)) for e, M in self.edges.items()}

    def connected_components(self) -> list[set]:
        """Connected components (largest first)."""
        adj = self.adjacency
        components, seen = [], set()
        for start in self.nodes:
            if start in seen:
                continue
            comp, queue = {start}, deque([start])
            seen.add(start)
            while queue:
                cur = queue.popleft()
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        queue.append(nb)
            components.append(comp)
        return sorted(components, key=len, reverse=True)

    # --- cycle-consistency scoring & filtering --------------------------

    def _alternative_path(self, edge: Edge, max_len=3):
        """Shortest path (<= max_len edges) between edge endpoints that does
        not use the edge itself. Returns node list from edge[1] to edge[0],
        or None."""
        a, b = edge
        banned = frozenset(edge)
        frontier = [(b, [b])]
        for _ in range(max_len):
            nxt = []
            for node, path in frontier:
                for nb in self.adjacency[node]:
                    if frozenset((node, nb)) == banned or nb in path:
                        continue
                    if nb == a:
                        return path + [a]
                    nxt.append((nb, path + [nb]))
            frontier = nxt
        return None

    def score_edges(self, metric: Metric = DEFAULT_METRIC, agg='min',
                    quad_fallback=True) -> ScoredEdges:
        """Score edges by cycle consistency.

        Triangles are the primary evidence; edges in no triangle optionally
        get a quad-cycle (length-4) fallback score, labelled separately since
        it compounds noise from two extra edges.

        agg: 'min' (robust: a good edge should sit in at least one clean
             cycle) or 'mean' (stricter legacy behaviour).
        """
        edge_errors: dict[Edge, list] = defaultdict(list)
        score_source: dict[Edge, str] = {}

        order = {n: i for i, n in enumerate(self.nodes)}
        adj = self.adjacency

        # --- triangles ---
        for a in order:
            for b in adj[a]:
                if order[b] <= order[a]:
                    continue
                for c in adj[a] & adj[b]:
                    if order[c] <= order[b]:
                        continue
                    cycle_M = self.compose_path((a, b, c, a))
                    if cycle_M is None:
                        continue
                    error = np.linalg.norm(metric.residuals(cycle_M))
                    for src, tgt in ((a, b), (b, c), (c, a)):
                        key, _ = self.edge_key(src, tgt)
                        edge_errors[key].append(error)
                        score_source[key] = 'triangle'

        # --- quad fallback for edges in no triangle ---
        if quad_fallback:
            for edge in self.edges:
                if edge in edge_errors:
                    continue
                path_back = self._alternative_path(edge, max_len=3)
                if path_back is None:
                    continue
                cycle_M = self.compose_path([edge[0]] + path_back)
                if cycle_M is None:
                    continue
                edge_errors[edge].append(np.linalg.norm(metric.residuals(cycle_M)))
                score_source[edge] = 'quad'

        agg_fn = {'min': np.min, 'mean': np.mean}[agg]
        edge_scores = {edge: (agg_fn(edge_errors[edge]) if edge in edge_errors else None)
                       for edge in self.edges}
        untestable = [e for e, s in edge_scores.items() if s is None]
        return ScoredEdges(edge_scores, dict(edge_errors), untestable, score_source)

    def filter_outlier_edges(self, scored: ScoredEdges, threshold=10.,
                             use_quad_cycles=False) -> FilteredGraph:
        """Remove edges whose cycle score exceeds threshold.

        By default only triangle-scored edges are eligible for removal; quad
        scores are diagnostic-only (use_quad_cycles=True to opt in).
        threshold: absolute value, or 'auto' for median + 5 * 1.4826 * MAD
            over eligible scores (falls back to 10. if fewer than 10 eligible
            edges).
        """
        eligible = {e: s for e, s in scored.scores.items()
                    if s is not None
                    and (use_quad_cycles or scored.source.get(e) == 'triangle')}

        if threshold == 'auto':
            vals = np.array(list(eligible.values()))
            if len(vals) >= 10:
                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                threshold_used = med + 5 * 1.4826 * mad
            else:
                threshold_used = 10.  # too few edges for robust stats
        else:
            threshold_used = float(threshold)

        outliers = {e for e, s in eligible.items() if s > threshold_used}
        return FilteredGraph(self.without_edges(outliers), outliers, threshold_used)

    # --- pose initialisation (spanning tree per component) ---------------

    def _spanning_tree(self, root: Node, component: set,
                       edge_costs=None, default_cost=1.):
        """Parent map + parent-before-child visit order within a component.
        edge_costs=None -> uniform costs (BFS-equivalent tree)."""
        adj = self.adjacency
        parent: dict[Node, Optional[Node]] = {root: None}
        dist = {root: 0.}
        tie = count()  # nodes may not be comparable
        heap = [(0., next(tie), root)]
        visit_order = []

        while heap:
            d, _, u = heappop(heap)
            if d > dist[u] + 1e-12:
                continue
            visit_order.append(u)
            for v in adj[u]:
                if v not in component:
                    continue
                if edge_costs is None:
                    cost = 1.
                else:
                    key, _ = self.edge_key(u, v)
                    cost = edge_costs.get(key)
                    cost = default_cost if cost is None else cost
                cost = max(cost, 1e-9)
                nd = d + cost
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heappush(heap, (nd, next(tie), v))
        return parent, visit_order

    def initialize_poses(self, method='dijkstra', edge_costs=None) -> InitialPoses:
        """Initial poses via a spanning tree per connected component.

        method: 'dijkstra' routes tree paths through low-cost (low
            cycle-score) edges; 'bfs' is the legacy arbitrary tree (uniform
            costs).
        Each component is rooted (and later anchored) at its highest-degree
        node (ties broken deterministically by node order).
        """
        adj = self.adjacency
        node_order = {n: i for i, n in enumerate(self.nodes)}
        components = self.connected_components()

        costs = edge_costs if method == 'dijkstra' else None
        default_cost = 1.
        if costs is not None:
            finite = [c for c in costs.values() if c is not None]
            default_cost = float(np.median(finite)) if finite else 1.

        poses: Poses = {}
        anchors: set = set()
        for comp in components:
            root = max(comp, key=lambda n: (len(adj[n]), -node_order[n]))
            anchors.add(root)
            poses[root] = np.eye(3)
            if len(comp) == 1:
                continue
            parent, visit_order = self._spanning_tree(root, comp, costs, default_cost)
            for node in visit_order[1:]:
                poses[node] = self.get(parent[node], node) @ poses[parent[node]]
        return InitialPoses(poses, anchors, components)


    # ---------------------------------------------------------------------------
    # Refinement (pose-consuming operations: module-level by design)
    # ---------------------------------------------------------------------------

    def refine_poses(self, poses: Poses, metric: Metric = DEFAULT_METRIC,
                     anchors=(), f_scale=None, max_nfev=1000, use_confidences=True,
                     pose_model: Union[PoseModel, str] = HOMOGRAPHY) -> Poses:
        """Pose-graph optimisation. Anchors (one per component) stay fixed,
        removing all gauge freedom. Edge residual blocks are optionally weighted
        by sqrt(confidence). f_scale is in the metric's residual units (pixels
        for TransferMetric): residuals beyond ~f_scale are down-weighted by the
        soft_l1 loss; None uses metric.default_f_scale.
        pose_model constrains the transform family (see PoseModel): 'similarity'
        or 'affine' poses have zero skew/perspective by construction."""
        if f_scale is None:
            f_scale = metric.default_f_scale
        model = resolve_pose_model(pose_model)
        free = [n for n in poses if n not in anchors and len(self.adjacency[n]) > 0]
        if not free or len(self) == 0:
            return dict(poses)
        free_idx = {n: i for i, n in enumerate(free)}

        x0 = np.concatenate([model.encode(poses[n]) for n in free])
        block = len(metric.residuals(np.eye(3)))

        # Precompute per-edge constants
        edge_data = []  # (src, tgt, inv_M_measured, sqrt_weight)
        for (src, tgt), M in self.edges.items():
            w = self.confidence((src, tgt)) if use_confidences else 1.0
            edge_data.append((src, tgt, np.linalg.inv(M), np.sqrt(w)))

        def unpack(x):
            current = dict(poses)  # anchored & isolated nodes keep initial values
            for n, i in free_idx.items():
                current[n] = model.decode(x[i * model.dof:(i + 1) * model.dof])
            return current

        def residuals(x):
            current = unpack(x)
            res = np.empty(len(edge_data) * block)
            for k, (src, tgt, inv_M, sw) in enumerate(edge_data):
                M_pred = self.predict(current[src], current[tgt])
                res[k * block:(k + 1) * block] = metric.residuals(M_pred @ inv_M) * sw
            return res

        # Sparsity: each edge's residual block touches at most two poses
        sparsity = lil_matrix((len(edge_data) * block, len(free) * model.dof), dtype=int)
        for k, (src, tgt, _, _) in enumerate(edge_data):
            rows = slice(k * block, (k + 1) * block)
            for n in (src, tgt):
                if n in free_idx:
                    i = free_idx[n]
                    sparsity[rows, i * model.dof:(i + 1) * model.dof] = 1

        result = least_squares(residuals, x0, loss='soft_l1', f_scale=f_scale,
                               jac_sparsity=sparsity, method='trf',
                               verbose=0, max_nfev=max_nfev)
        return unpack(result.x)



# ---------------------------------------------------------------------------
# Centroid & scale clustering
# ---------------------------------------------------------------------------

def find_centroid_pose(poses: Poses, metric: Metric = DEFAULT_METRIC,
                       pose_model: Union[PoseModel, str] = HOMOGRAPHY):
    """Centroid pose minimising the metric deviation to all poses. Solved in
    the given pose family so that centering preserves the family."""
    model = resolve_pose_model(pose_model)

    # Initialize at component-wise mean.
    # NB: element-wise averaging of A is only valid for small rotations.
    all_A, all_t, all_p = zip(*[decompose_homography(pose) for pose in poses.values()])
    init = np.eye(3)
    init[:2, :2] = np.mean(all_A, axis=0)
    init[:2, 2] = np.mean(all_t, axis=0)
    init[2, :2] = np.mean(all_p, axis=0)

    def residuals(x):
        centroid_inv = np.linalg.inv(model.decode(x))
        return np.concatenate([metric.residuals(pose @ centroid_inv)
                               for pose in poses.values()])

    result = least_squares(residuals, model.encode(init), verbose=0)
    return model.decode(result.x)


def cluster_pose_scales(poses: Poses, scale_ratio_threshold=1.3) -> list[list[Node]]:
    """Cluster poses by isotropic scale (1-D). Returns clusters (lists of
    keys), largest first. Poses are sorted by scale before greedy clustering
    so the result is order-stable; a new cluster starts when the scale ratio
    to the cluster's first (smallest) member exceeds the threshold."""
    items = sorted(poses.items(), key=lambda kv: pose_scale(kv[1]))
    clusters: list[tuple[float, list]] = []  # (representative scale, keys)
    for k, pose in items:
        s = pose_scale(pose)
        if clusters and s / clusters[-1][0] < scale_ratio_threshold:
            clusters[-1][1].append(k)
        else:
            clusters.append((s, [k]))
    return sorted([keys for _, keys in clusters], key=len, reverse=True)


def center_poses(poses: Poses, metric: Metric = DEFAULT_METRIC, target_scale=None,
                 pose_model: Union[PoseModel, str] = HOMOGRAPHY):
    """Recenter poses about their centroid. If target_scale is given, the
    centroid's scale is overridden (global zoom about the origin) so that
    poses at target_scale render at native resolution -- used to make
    mixed-resolution inputs render at a chosen cluster's scale instead of a
    compromise mean of scales.

    Returns (centered_poses, centroid)."""
    model = resolve_pose_model(pose_model)
    centroid = find_centroid_pose(poses, metric, model)
    if target_scale is not None:
        k = target_scale / pose_scale(centroid)
        centroid = np.diag([k, k, 1.]) @ centroid
    centroid_inv = np.linalg.inv(centroid)
    centered = {n: pose @ centroid_inv for n, pose in poses.items()}
    if model is not HOMOGRAPHY:
        # composition of exact family members can pick up ~1-ulp asymmetry;
        # re-project so excluded components are exactly zero in the output
        centered = {n: model.decode(model.encode(P)) for n, P in centered.items()}
    return centered, centroid


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult(Mapping):
    """Pipeline output. Acts as a read-only mapping over the final poses, so
    it can be passed directly to visualise_poses / plot_pose_graph or
    iterated like a poses dict; the full diagnostics live in the fields."""
    poses: Poses                          # final, centred poses
    graph: "TransformGraph"               # cleaned graph the poses were solved on
    initial_poses: Poses                  # spanning-tree init (projected into pose family)
    centroid: Homography
    edge_scores: dict                     # edge -> aggregated cycle score (None if unscorable)
    edge_errors: dict                     # edge -> raw per-cycle errors
    edge_score_source: dict               # edge -> 'triangle' | 'quad'
    outlier_edges: set
    untestable_edges: list
    threshold_used: float
    components: list                      # list of node sets, largest first
    anchors: set
    initial_edge_errors: dict             # residuals at initialisation (sanity check)
    final_edge_errors: dict               # residuals after optimisation
    pose_scales: dict
    scale_clusters: Optional[list]        # None if cluster_scales=False
    target_scale: Optional[float]
    transform_magnitudes: dict = None     # edge -> metric size of raw measurement (pre-filtering)
    metric: Any = None
    pose_model: Any = None

    # --- mapping over final poses ---
    def __getitem__(self, node):
        return self.poses[node]

    def __iter__(self):
        return iter(self.poses)

    def __len__(self):
        return len(self.poses)

    def plot(self, stage='final', **kwargs):
        """Spatial pose-graph plot for a pipeline stage:
        'scores'  cycle scores on the initial layout (pre-filtering evidence)
        'initial' spanning-tree residuals (should be ~0 on tree edges)
        'final'   post-optimisation residuals on the final layout
        """
        poses, errors = {
            'scores': (self.initial_poses, self.edge_scores),
            'initial': (self.initial_poses, self.initial_edge_errors),
            'final': (self.poses, self.final_edge_errors),
        }[stage]
        if 'img_shape' not in kwargs and isinstance(self.metric, TransferMetric):
            kwargs['img_shape'] = (self.metric.height, self.metric.width)
        kwargs.setdefault('title', f'pose graph [{stage}]')
        return plot_pose_graph(poses, self.graph, errors, **kwargs)

    def plot_matrix(self, kind='scores', **kwargs):
        """N x N adjacency-matrix heatmap for a pipeline quantity:
        'magnitudes'  size of each raw measurement (how far apart pairs are)
        'scores'      cycle-consistency score per edge (blank = untestable)
        'initial'     spanning-tree residual per surviving edge
        'final'       post-optimisation residual per surviving edge
        Outlier edges are marked with a red x on 'magnitudes' and 'scores'.
        A hot row/column is the signature of a single bad image; blank cells
        are missing/removed/untestable edges."""
        values, marked = {
            'magnitudes': (self.transform_magnitudes, self.outlier_edges),
            'scores': (self.edge_scores, self.outlier_edges),
            'initial': (self.initial_edge_errors, None),
            'final': (self.final_edge_errors, None),
        }[kind]
        # consistent node order across all kinds: derive from the pre-filter edge set
        nodes = kwargs.pop('nodes', None)
        if nodes is None:
            seen = {}
            for s, t in self.edge_scores:
                seen.setdefault(s)
                seen.setdefault(t)
            nodes = list(seen)
        kwargs.setdefault('title', f'edge matrix [{kind}]')
        return plot_edge_matrix(values, nodes=nodes, mark_edges=marked, **kwargs)

    def summary(self):
        n_tri = sum(1 for s in self.edge_score_source.values() if s == 'triangle')
        n_quad = sum(1 for s in self.edge_score_source.values() if s == 'quad')
        n_edges = len(self.edge_scores)
        print(f"[cycles] {n_edges} edges: {n_tri} triangle-scored, {n_quad} quad-scored "
              f"(diagnostic), {len(self.untestable_edges)} unverified")
        tested = [s for e, s in self.edge_scores.items()
                  if s is not None and self.edge_score_source.get(e) == 'triangle']
        if tested:
            print(f"[cycles] triangle scores: median={np.median(tested):.3f}, "
                  f"max={np.max(tested):.3f} (threshold={self.threshold_used:.3f})")
        for e in sorted(self.outlier_edges, key=lambda e: -self.edge_scores[e]):
            print(f"[cycles] removed {e}: score={self.edge_scores[e]:.3f}")

        sizes = [len(c) for c in self.components]
        print(f"[graph]  {len(self.components)} connected component(s), sizes={sizes}, "
              f"anchors={sorted(map(str, self.anchors))}")
        if self.graph.failed:
            print(f"[graph]  {len(self.graph.failed)} pair(s) failed registration")

        for label, errs in (("init", self.initial_edge_errors),
                            ("final", self.final_edge_errors)):
            if errs:
                vals = list(errs.values())
                print(f"[optimize] {label} edge residuals: "
                      f"median={np.median(vals):.3f}, max={np.max(vals):.3f}")
        if self.final_edge_errors:
            worst = max(self.final_edge_errors, key=self.final_edge_errors.get)
            print(f"[optimize] worst edge: {worst} ({self.final_edge_errors[worst]:.3f})")

        if self.scale_clusters is not None:
            reps = [float(np.median([self.pose_scales[k] for k in cl]))
                    for cl in self.scale_clusters]
            print(f"[scale]  clusters (size@scale): "
                  + ", ".join(f"{len(cl)}@{r:.3f}" for cl, r in zip(self.scale_clusters, reps))
                  + (f" -> target={self.target_scale:.3f}" if self.target_scale else ""))


def solve_poses(transforms, *, threshold=10.,
                metric: Metric = DEFAULT_METRIC,
                pose_model: Union[PoseModel, str] = 'homography',
                                                 # 'affine': no perspective;
                                                 # 'similarity': no perspective, no skew
                edge_error_agg='min',            # or 'mean' (stricter)
                use_quad_cycles=False,           # quads filter edges too (default: diagnostic only)
                init_method='dijkstra',          # or 'bfs' (arbitrary tree)
                f_scale=None,                    # robust-loss scale; None = metric.default_f_scale
                use_confidences=True,
                cluster_scales=True,             # False: centroid at mean scale
                scale_ratio_threshold=1.3,
                scale_pref='largest',            # 'largest' cluster or 'finest' (max scale)
                max_nfev=1000, verbose=False,
                ) -> PipelineResult:
    """Full pipeline: cycle-based outlier removal, pose-graph optimisation,
    scale clustering, and centroid alignment.

    transforms: a TransformGraph (canonical; carries confidences), or a raw
        {(src, tgt): 3x3} dict for convenience.
    metric: TransferMetric (default; pixel units, so threshold=10 means
        10 px of cycle drift) or ErrorWeights (original mixed-unit metric --
        recalibrate threshold/f_scale when switching).
    pose_model: transform family the poses are constrained to. 'similarity'
        and 'affine' exclude skew/perspective structurally (exactly zero in
        the output, guaranteed); measurements stay full homographies, so
        final residuals then also report model-fit error -- compare against
        'homography' residuals to see what the constraint costs.
    threshold: absolute cycle-score cutoff, or 'auto' (median + 5*MAD).
    cluster_scales: if True, the centroid's scale is overridden with the
        median scale of the preferred scale-cluster, so mixed-resolution
        inputs render at that resolution instead of a mean of scales.
        scale_pref='largest' picks the biggest cluster; 'finest' picks the
        cluster with the largest (finest) representative scale.

    Returns a PipelineResult; final poses are in `.poses`, and `.summary()`
    prints the diagnostics that verbose=True used to.
    """
    graph = transforms if isinstance(transforms, TransformGraph) \
        else TransformGraph(transforms)
    model = resolve_pose_model(pose_model)

    # --- Cycle scoring & outlier removal ---
    transform_magnitudes = graph.transform_magnitudes(metric)
    scored = graph.score_edges(metric, agg=edge_error_agg)
    filtered = graph.filter_outlier_edges(scored, threshold, use_quad_cycles)
    clean_graph = filtered.graph

    # --- Pose initialisation ---
    costs = scored.scores if init_method == 'dijkstra' else None
    init = clean_graph.initialize_poses(method=init_method, edge_costs=costs)
    init_poses = init.poses
    if model is not HOMOGRAPHY:  # project tree poses into the family
        init_poses = {n: model.decode(model.encode(P)) for n, P in init_poses.items()}
    initial_edge_errors = clean_graph.residual_errors(init_poses, metric)

    # --- Pose refinement ---
    optimized = clean_graph.refine_poses(init_poses, metric, init.anchors,
                                         f_scale=f_scale, max_nfev=max_nfev,
                                         use_confidences=use_confidences, pose_model=model)
    final_edge_errors = clean_graph.residual_errors(optimized, metric)

    # --- Scale clustering & centroid ---
    pose_scales = {k: pose_scale(p) for k, p in optimized.items()}

    scale_clusters, target_scale = None, None
    if cluster_scales and optimized:
        scale_clusters = cluster_pose_scales(optimized, scale_ratio_threshold)
        reps = [float(np.median([pose_scales[k] for k in cl])) for cl in scale_clusters]
        if scale_pref == 'largest':
            target_scale = reps[0]
        elif scale_pref == 'finest':
            target_scale = max(reps)
        else:
            raise ValueError(f"unknown scale_pref: {scale_pref!r}")

    node_poses, centroid = center_poses(optimized, metric, target_scale, pose_model=model)

    result = PipelineResult(
        poses=node_poses, graph=clean_graph, initial_poses=init.poses,
        centroid=centroid,
        edge_scores=scored.scores, edge_errors=scored.errors,
        edge_score_source=scored.source,
        outlier_edges=filtered.outliers, untestable_edges=scored.untestable,
        threshold_used=filtered.threshold_used,
        components=init.components, anchors=init.anchors,
        initial_edge_errors=initial_edge_errors,
        final_edge_errors=final_edge_errors,
        pose_scales=pose_scales, scale_clusters=scale_clusters,
        target_scale=target_scale, transform_magnitudes=transform_magnitudes,
        metric=metric, pose_model=model,
    )
    if verbose:
        result.summary()
    return result


# ---------------------------------------------------------------------------
# Visualisation (heavy imports are local so headless imports stay light)
# ---------------------------------------------------------------------------

def visualise_poses(poses, node_paths=None, targ_shape=None, figsize=(8, 8)):
    import os

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    from PIL import Image as PILImage
    from rtnls_registration.transformation import ProjectiveTransform

    @cache
    def get_img(img_path):
        return np.array(PILImage.open(img_path).convert('RGB'))

    output = widgets.Output()
    node_idxs = list(poses.keys())

    def show_image(idx):
        with output:
            output.clear_output(wait=True)

            node_idx = node_idxs[idx]
            img_path = node_idx if node_paths is None else node_paths[node_idx]

            image = get_img(img_path)
            transform_M = poses[node_idx]

            shape = targ_shape if targ_shape is not None else image.shape[:2]
            transform = ProjectiveTransform(transform_M)
            imageT = transform.warp_inverse(image, shape)

            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(imageT)
            ax.axis('off')
            ax.set_title(f"{idx + 1}/{len(node_idxs)}: {os.path.basename(img_path)}",
                         fontsize=12, pad=10)
            plt.tight_layout()
            plt.show()

    slider = widgets.IntSlider(
        value=0, min=0, max=len(node_idxs) - 1, step=1,
        description='Image:',
        continuous_update=True,  # Update live while dragging
        layout=widgets.Layout(width='80%')
    )
    slider.observe(lambda change: show_image(change['new']), names='value')

    show_image(0)
    display(slider, output)

    return slider, output


def plot_edge_matrix(values, nodes=None, ax=None, title=None, log=True,
                     mark_edges=None, cbar_label=None):
    """Symmetric N x N heatmap of a per-edge quantity (transform magnitudes,
    cycle scores, residuals). None/missing edges render blank; mark_edges
    (e.g. removed outliers) get a red x. Log colour scale by default since
    outliers dwarf inliers by orders of magnitude."""
    import os

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    vals = {e: v for e, v in values.items() if v is not None}
    if nodes is None:
        seen = {}
        for s, t in values:
            seen.setdefault(s)
            seen.setdefault(t)
        nodes = list(seen)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    mat = np.full((n, n), np.nan)
    for (i, j), v in vals.items():
        if i in idx and j in idx:
            mat[idx[i], idx[j]] = mat[idx[j], idx[i]] = v

    if ax is None:
        _, ax = plt.subplots(figsize=(0.5 * n + 3, 0.5 * n + 2.5))

    finite = mat[np.isfinite(mat)]
    positive = finite[finite > 0]
    norm = None
    if log and positive.size and positive.max() / max(positive.min(), 1e-300) > 10:
        vmin = max(positive.min(), positive.max() * 1e-6)
        norm = LogNorm(vmin=vmin, vmax=positive.max())
        mat = np.where(np.isfinite(mat), np.maximum(mat, vmin), mat)  # zeros -> vmin

    im = ax.imshow(mat, norm=norm, cmap='viridis')
    im.cmap.set_bad('0.92')  # missing edges

    for e in (mark_edges or ()):
        i, j = e
        if i in idx and j in idx:
            ax.plot(idx[j], idx[i], 'x', color='red', ms=8, mew=2)
            ax.plot(idx[i], idx[j], 'x', color='red', ms=8, mew=2)

    labels = [os.path.basename(str(k)) for k in nodes]
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
    ax.set_title(title or 'edge matrix')
    return ax


def plot_pose_graph(poses, graph_or_transforms, edge_errors=None,
                    img_shape=(768, 768), ax=None, title=None):
    """Spatial pose-graph plot: image footprints on the canvas (inv(pose)
    applied to corners, per the module convention) with graph edges between
    footprint centres coloured by edge_errors (e.g. cycle scores or
    post-optimisation residuals)."""
    import matplotlib.pyplot as plt

    graph = graph_or_transforms if isinstance(graph_or_transforms, TransformGraph) \
        else TransformGraph(graph_or_transforms)

    h, w = img_shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [0, 0, 1]], float).T
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))

    centres = {}
    for k, P in poses.items():
        c = np.linalg.inv(P) @ corners  # footprint in canvas frame
        c = c[:2] / c[2]
        ax.plot(*c, color='0.7', lw=0.8)
        centres[k] = c[:, :4].mean(axis=1)

    errs = {e: v for e, v in (edge_errors or {}).items() if v is not None}
    vmax = max(errs.values()) if errs else 1.
    for (i, j) in graph.edges:
        if i not in centres or j not in centres:
            continue
        p, q = centres[i], centres[j]
        e = errs.get((i, j))
        color = plt.cm.viridis(e / vmax) if e is not None else '0.85'
        style = '-' if e is not None else '--'
        ax.plot([p[0], q[0]], [p[1], q[1]], style, color=color, lw=1.5)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title or 'pose graph (edge colour = error)')
    return ax