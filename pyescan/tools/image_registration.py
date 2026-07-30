import heapq
import os
from collections import defaultdict, deque
from functools import cache
from itertools import combinations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image as PILImage
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tqdm import tqdm


def get_transforms(img_paths, pbar=True):
    from rtnls_registration import Registration
    registrator = Registration(quadratic=False)

    @cache
    def get_processed(img_path):
        img_data = np.array(PILImage.open(img_path).convert('RGB'))
        preprocess_result, features = registrator._init_image(img_data, img_path)
        return preprocess_result, features

    img_pairs = list(combinations(img_paths, 2))

    transforms = {}
    iterator = tqdm(img_pairs, desc="Computing transforms") if pbar else img_pairs
    for img_path_0, img_path_1 in iterator:
        try:
            registrator.preprocess_result0, registrator.features0 = get_processed(img_path_0)
            registrator.preprocess_result1, registrator.features1 = get_processed(img_path_1)

            transform_result = registrator.run()
            M = transform_result.M
            transforms[img_path_0, img_path_1] = M / M[2, 2]  # normalize scale
        except Exception as e:
            print(f"Failed for pair {img_path_0}, {img_path_1}: {e}")
            continue

    get_processed.cache_clear()
    return transforms


def decompose_homography(H):
    H_norm = H / H[2, 2]

    A = H_norm[:2, :2]  # Linear part (rotation/scale/shear)
    t = H_norm[:2, 2]   # Translation
    p = H_norm[2, :2]   # Perspective terms
    return A, t, p


def homography_residuals(M, a_weight=1., t_weight=1./768, p_weight=1000., skew_weight=10.):
    """Weighted deviation-from-identity components as a residual vector.

    skew_weight penalizes deviation of A from a scaled rotation (similarity),
    acting as a soft prior for low-skew problems."""
    A, t, p = decompose_homography(M)
    skew = np.array([A[0, 1] + A[1, 0], A[0, 0] - A[1, 1]])  # zero iff A is a scaled rotation
    return np.concatenate([
        (A - np.eye(2)).ravel() * a_weight,
        skew * skew_weight,
        t * t_weight,
        p * p_weight,
    ])


def homography_error(M, a_weight=1., t_weight=1./768, p_weight=1000., skew_weight=10.):
    """Scalar error (norm of the residual vector)"""
    return np.linalg.norm(homography_residuals(M, a_weight, t_weight, p_weight, skew_weight))

def transfer_error(M, img_size=(768, 768)):
    """Pixel displacement of image corners under homography M.
    M == identity (up to scale) -> zero loss.
    Returns 8 residuals (4 corners x 2 coords), or scalar RMS if reduce=True."""
    w, h = img_size
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T  # 3x4
    p = M @ corners
    p = p[:2] / p[2]                       # perspective divide (handles scale ambiguity)
    res = (p - corners[:2]).ravel()        # pixel errors
    return np.sqrt((res**2).mean())

def pose_scale(M):
    """Isotropic scale of the affine part"""
    A, _, _ = decompose_homography(M)
    return np.sqrt(np.abs(np.linalg.det(A)))


def _find_edge(transforms, src, tgt, symmetric=True):
    """Return (key, inverted) for the edge between src and tgt, or None"""
    if (src, tgt) in transforms:
        return (src, tgt), False
    if symmetric and (tgt, src) in transforms:
        return (tgt, src), True
    return None


def _compose_path(transforms, path_idxs, symmetric=True):
    """Compute product of matrices along node sequence"""
    M = np.eye(3)
    for i in range(len(path_idxs) - 1):
        edge = _find_edge(transforms, path_idxs[i], path_idxs[i + 1], symmetric)
        if edge is None:
            return None  # Missing edge
        key, inverted = edge
        M_edge = np.linalg.inv(transforms[key]) if inverted else transforms[key]
        M = M @ M_edge
    return M / M[2, 2]


def _get_node_idxs(transforms):
    node_idxs = set()
    for (source, target) in transforms.keys():
        node_idxs.add(source)
        node_idxs.add(target)
    return list(node_idxs)


def _get_adjacency(transforms):
    """node -> set of neighbours (undirected)"""
    adj = defaultdict(set)
    for src, tgt in transforms.keys():
        adj[src].add(tgt)
        adj[tgt].add(src)
    return adj


def _calculate_cycle_errors(transforms, error_fn, node_idxs=None, symmetric=True,):
    """Detect erroneous transforms using cycle consistency.
    Scores cycle accoring to error_fn on composite homography.
    Only enumerates triangles where all three edges exist."""
    edge_errors = {edge: [] for edge in transforms.keys()}

    if node_idxs is None: node_idxs = _get_node_idxs(transforms)
    adj = _get_adjacency(transforms)

    ordered = {node: i for i, node in enumerate(node_idxs)}
    for a in node_idxs:
        for b in adj[a]:
            if ordered.get(b, -1) <= ordered[a]:
                continue
            for c in adj[a] & adj[b]:
                if ordered.get(c, -1) <= ordered[b]:
                    continue
                cycle_M = _compose_path(transforms, (a, b, c, a), symmetric=symmetric)
                if cycle_M is None:
                    continue
                error = error_fn(cycle_M)
                for src, tgt in ((a, b), (b, c), (c, a)):
                    edge = _find_edge(transforms, src, tgt, symmetric)
                    if edge is not None:
                        edge_errors[edge[0]].append(error)
    return edge_errors


def get_poses(transforms, node_idxs=None, weights=None):
    if node_idxs is None: node_idxs = _get_node_idxs(transforms)
    adj = _get_adjacency(transforms)
    if weights is None: weights = {k: 1.0 for k in transforms}

    def edge_weight(u, v):
        return weights[u, v] if (u, v) in weights else weights[v, u]

    poses = dict()
    for start_node in node_idxs:
        if start_node in poses:
            continue
        dist = {start_node: 0.0}
        poses[start_node] = np.eye(3)
        pq = [(0.0, start_node)]
        while pq:
            d, current = heapq.heappop(pq)
            if d > dist[current]:
                continue
            for neighbour in adj[current]:
                nd = d + edge_weight(current, neighbour)
                if neighbour not in dist or nd < dist[neighbour]:
                    dist[neighbour] = nd
                    if (current, neighbour) in transforms:
                        # T(cur,nb) = pose_nb @ inv(pose_cur)  =>  pose_nb = T @ pose_cur
                        poses[neighbour] = transforms[current, neighbour] @ poses[current]
                    else:
                        poses[neighbour] = np.linalg.inv(transforms[neighbour, current]) @ poses[current]
                    heapq.heappush(pq, (nd, neighbour))
    return poses


def _flatten_matrix(M):
    """Convert 3x3 matrix (with M[2,2]=1) to 8D vector"""
    return (M / M[2, 2]).flatten()[:8]

def _unflatten_matrix(v):
    """Convert 8D vector to 3x3 matrix with M[2,2]=1"""
    return np.append(v, 1.).reshape(3, 3)

def _optimize_poses(poses, transforms, img_size=(768, 768), edge_weights=None,
                    robust_loss='huber', f_scale=2.0):
    pose_keys = list(poses.keys())
    pose_to_idx = {k: i for i, k in enumerate(pose_keys)}
    anchor = poses[pose_keys[0]]

    x0 = np.concatenate([_flatten_matrix(poses[k]) for k in pose_keys[1:]])

    # Constant edge data, stacked
    i_idx = np.array([pose_to_idx[s] for s, t in transforms])
    j_idx = np.array([pose_to_idx[t] for s, t in transforms])
    Mi = np.linalg.inv(np.stack(list(transforms.values())))     # (E, 3, 3)
    if edge_weights is None:
        edge_w = np.ones(len(Mi))
    else:
        edge_w = np.sqrt(np.array([edge_weights[k] for k in transforms]))

    w, h = img_size
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T  # 3x4
    n_res_per_edge = 8

    def residuals(x):
        P = np.empty((len(pose_keys), 3, 3))
        P[0] = anchor
        for n in range(1, len(pose_keys)):          # unflatten is cheap; keep the loop
            P[n] = _unflatten_matrix(x[(n - 1) * 8:n * 8])
        P_inv = np.linalg.inv(P)                    # batched, one LAPACK call
        M_diff = P[j_idx] @ P_inv[i_idx] @ Mi       # (E, 3, 3), fully batched
        p = M_diff @ corners                        # (E, 3, 4)
        p = p[:, :2] / p[:, 2:3]                    # perspective divide
        res = (p - corners[:2]) * edge_w[:, None, None]
        return res.reshape(-1)

    sparsity = lil_matrix((n_res_per_edge * len(Mi), len(x0)), dtype=int)
    for e in range(len(Mi)):
        rows = slice(e * n_res_per_edge, (e + 1) * n_res_per_edge)
        if i_idx[e] > 0: sparsity[rows, (i_idx[e] - 1) * 8:i_idx[e] * 8] = 1
        if j_idx[e] > 0: sparsity[rows, (j_idx[e] - 1) * 8:j_idx[e] * 8] = 1

    result = least_squares(residuals, x0, jac_sparsity=sparsity.tocsr(),
                           loss=robust_loss, f_scale=f_scale, x_scale='jac',
                           ftol=1e-6, xtol=1e-6, max_nfev=200)

    return {k: (anchor if n == 0 else _unflatten_matrix(result.x[(n - 1) * 8:n * 8]))
            for n, k in enumerate(pose_keys)}


def find_centroid_pose(poses):
    """Simpy return the mean over A, t, p"""

    all_A, all_t, all_p = zip(*[decompose_homography(pose) for pose in poses.values()])

    init_centroid = np.eye(3)
    init_centroid[:2, :2] = np.mean(all_A, axis=0)
    init_centroid[:2, 2] = np.mean(all_t, axis=0)
    init_centroid[2, :2] = np.mean(all_p, axis=0)

    return init_centroid


def cluster_pose_scales(poses, scale_ratio_threshold=1.3):
    """Cluster poses by isotropic scale (1-D). Returns clusters (lists of keys),
    largest first. Two poses are in the same cluster if their scale ratio is
    below scale_ratio_threshold relative to the cluster representative."""
    clusters = []  # list of (representative_scale, [keys])
    for k, pose in poses.items():
        s = pose_scale(pose)
        for cl in clusters:
            rep_s = cl[0]
            if max(s / rep_s, rep_s / s) < scale_ratio_threshold:
                cl[1].append(k)
                break
        else:
            clusters.append((s, [k]))
    return sorted([keys for _, keys in clusters], key=len, reverse=True)


def get_cleaned_poses(transforms,
                      node_idxs=None,
                      error_fn=transfer_error,
                      edge_error_agg='min',           # 'min' (default) or 'mean'
                      threshold=10.,
                      optimize_poses=False,
                      verbose=False, return_diagnostics=False):
    """Full pipeline: cycle-based outlier removal, pose-graph optimization,
    scale clustering, and centroid alignment.

    edge_error_agg: how to aggregate per-cycle errors per edge. 'min' is robust
        to contamination from a bad neighbouring edge (a good edge should sit in
        at least one clean cycle); 'mean' is the stricter legacy behaviour.
    cluster_scales: if True, the centroid's scale is overridden with the scale
        of the largest scale-cluster, so mixed-resolution inputs render at the
        dominant resolution instead of a mean of scales.
    return_diagnostics: if True, returns (poses, diagnostics dict).
    """
    
    if node_idxs is None: node_idxs = _get_node_idxs(transforms)

    # --- Cycle errors & outlier removal ---
    edge_errors = _calculate_cycle_errors(transforms, error_fn, node_idxs)
    agg = {'min': np.min, 'mean': np.mean}.get(edge_error_agg, edge_error_agg)
    edge_scores = {edge: (agg(errors) if len(errors) > 0 else None)
                   for edge, errors in edge_errors.items()}

    # --- Pose graph & optimization ---
    poses = get_poses(transforms, node_idxs, weights=edge_scores) #Use original uncleaned
    
    # --- Pose graph & optimization ---
    if optimize_poses:
        outlier_edges = {e for e, s in edge_scores.items() if s is not None and s > threshold}
        untestable_edges = [e for e, s in edge_scores.items() if s is None]
        transforms_clean = {k: v for k, v in transforms.items() if k not in outlier_edges}

        if verbose:
            tested = [s for s in edge_scores.values() if s is not None]
            print(f"[cycles] {len(transforms)} edges, {len(tested)} testable, "
                  f"{len(untestable_edges)} in no cycle (unverified)")
            if tested:
                print(f"[cycles] edge {edge_error_agg}-error: median={np.median(tested):.3f}, "
                      f"max={np.max(tested):.3f} (threshold={threshold})")
            for e in sorted(outlier_edges, key=lambda e: -edge_scores[e]):
                print(f"[cycles] removed {e}: {edge_error_agg}-error={edge_scores[e]:.3f}")

        optimized_poses = _optimize_poses(poses, transforms_clean)

        # Post-optimization residual per edge
        final_edge_errors = {}
        for (src, tgt), M in transforms_clean.items():
            M_pred = optimized_poses[tgt] @ np.linalg.inv(optimized_poses[src])
            final_edge_errors[src, tgt] = error_fn(M_pred @ np.linalg.inv(M))

        if verbose and final_edge_errors:
            vals = list(final_edge_errors.values())
            print(f"[optimize] post-opt edge residuals: median={np.median(vals):.3f}, max={np.max(vals):.3f}")
            worst = max(final_edge_errors, key=final_edge_errors.get)
            print(f"[optimize] worst edge: {worst} ({final_edge_errors[worst]:.3f})")
    else:
        optimized_poses = poses

    # --- Centroid ---
    centroid = find_centroid_pose(optimized_poses)
    centroid_inv = np.linalg.inv(centroid)
    node_poses = {node_idx: pose @ centroid_inv for node_idx, pose in optimized_poses.items()}

    if return_diagnostics:
        diagnostics = {
            'edge_errors': edge_errors,               # raw per-cycle errors per edge
            'edge_scores': edge_scores,               # aggregated score per edge (None if untestable)
            'outlier_edges': outlier_edges,
            'untestable_edges': untestable_edges,
            'final_edge_errors': final_edge_errors,   # post-optimization residual per edge
            'pose_scales': {k: pose_scale(p) for k, p in optimized_poses.items()},
            'centroid': centroid,
        }
        return node_poses, diagnostics

    return node_poses


def visualise_poses(poses, node_paths=None, targ_shape=None, figsize=(8, 8)):
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