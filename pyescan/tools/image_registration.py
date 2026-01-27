from scipy.optimize import least_squares
import numpy as np

import os

from collections import deque
from functools import cache
from itertools import combinations

from PIL import Image as PILImage
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_transforms(img_paths, pbar=True):
    from rtnls_registration import Registration
    registrator = Registration(quadratic=False)

    @cache
    def get_processed(img_path):
        img_data = np.array(PILImage.open(img_path).convert('RGB'))
        preprocess_result, features = registrator._init_image(img_data, img_path)
        return preprocess_result, features
    
    # Generate all pairs
    img_pairs = list(combinations(img_paths, 2))
    
    # Compute pairwise transforms
    transforms = {}
    iterator = tqdm(img_pairs, desc="Computing transforms") if pbar else img_pairs
    for img_path_0, img_path_1 in iterator:
        try:
            registrator.preprocess_result0, registrator.features0 = get_processed(img_path_0)
            registrator.preprocess_result1, registrator.features1 = get_processed(img_path_1)
            
            transform_result = registrator.run()
            transforms[img_path_0, img_path_1] = transform_result.M
        except Exception as e:
            print(f"Failed for pair {img_path_0}, {img_path_1}: {e}")
            continue
    
    get_processed.cache_clear()
    return transforms


def decompose_homography(H):
    H_norm = H / H[2,2]

    A = H_norm[:2, :2] # Linear part (rotation/scale/shear)
    t = H_norm[:2, 2]  # Translation
    p = H_norm[2, :2]  # Perspective terms
    return A, t, p


def homography_error(M, a_weight=1., t_weight=1./768, p_weight=1000.):
    A, t, p = decompose_homography(M)
    
    rotation_error = np.linalg.norm(A - np.eye(2), 'fro') * a_weight
    translation_error = np.linalg.norm(t) * t_weight
    perspective_error = np.linalg.norm(p) * p_weight
    return rotation_error + translation_error + perspective_error


def _compose_path(transforms, path_idxs, symmetric=True):
    """Compute porduct of matrices in node sequence"""

    M = np.eye(3)
    for i, _ in enumerate(path_idxs[:-2]):
        src, tgt = path_idxs[i], path_idxs[i + 1]
        if (src, tgt) in transforms:
            M = M @ transforms[src, tgt]
        elif symmetric and ((tgt, src) in transforms):
            M = M @ np.linalg.inv(transforms[tgt, src])
        else:
            return None  # Missing edge
    return M


def _get_node_idxs(transforms):
    node_idxs = set()
    for (source, target) in transforms.keys():
        node_idxs.add(source)
        node_idxs.add(target)
    return list(node_idxs)


def _calculate_cycle_errors(transforms, node_idxs=None, symmetric=True, a_weight=1., t_weight=1./768, p_weight=1000.):
    """Detect erroneous transforms using cycle consistency"""
    outlier_edges = set()
    edge_errors = {edge: [] for edge in transforms.keys()}
    
    if node_idxs is None: node_idxs = _get_node_idxs(transforms)
    
    for cycle in combinations(node_idxs, 3):
        full_cycle = (*cycle, cycle[0]) # extend to full cycle
        cycle_M = _compose_path(transforms, cycle, symmetric=symmetric)
        
        if cycle_M is not None:
            # Record error for each edge in this cycle
            error = homography_error(cycle_M, a_weight, t_weight, p_weight)
            
            for i in range(3):
                src, tgt = cycle[i], cycle[(i + 1) % 3]
                edge = (src, tgt) if ((src, tgt) in transforms) or not symmetric else (tgt, src)

                if edge in transforms:
                    edge_errors[edge].append(error)
    return edge_errors


def get_poses(transforms, node_idxs=None):
    if node_idxs is None: node_idxs = _get_node_idxs(transforms)
    
    poses = dict()
    
    # Process each connected component
    for start_node in node_idxs:
        if start_node in poses:
            continue
            
        # BFS to find connected component
        poses[start_node] = np.eye(3)  # Anchor this component
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            
            # Check all transforms involving current node
            for src, tgt in transforms.keys():
                if src == current and tgt not in poses:
                    poses[tgt] = poses[src] @ transforms[src, tgt]
                    queue.append(tgt)
                elif tgt == current and src not in poses:
                    poses[src] = poses[tgt] @ np.linalg.inv(transforms[src, tgt])
                    queue.append(src)
    
    return poses


def _flatten_matrix(M):
    """Convert 3x3 matrix to 9D vector"""
    return M.flatten()

def _unflatten_matrix(v):
    """Convert 9D vector to 3x3 matrix"""
    M = v.reshape(3, 3)
    # Normalize so M[2,2] = 1
    return M / M[2, 2]

def _optimize_poses(poses, transforms, a_weight=1., t_weight=1./768, p_weight=1000.):
    """Optimize homographies using cycle consistency"""
    n = len(poses)
    
    pose_to_idx = {k : i for i, k in enumerate(poses.keys()) }

    # Flatten initial poses
    x0 = np.concatenate([_flatten_matrix(p) for p in poses.values()])
    
    def residuals(x):
        # Reshape to matrices
        current_poses = [_unflatten_matrix(x[i*9:(i+1)*9]) for i in range(n)]
        
        res = []
        for (src, tgt), M_measured in transforms.items():
            # M_ij should equal pose_j @ inv(pose_i)
            M_predicted = current_poses[pose_to_idx[tgt]] @ np.linalg.inv(current_poses[pose_to_idx[src]])
            
            # Decompose error
            M_diff = M_predicted @ np.linalg.inv(M_measured)
            res.append(homography_error(M_diff, a_weight, t_weight, p_weight))
        
        return np.array(res)
    
    # Optimize
    result = least_squares(residuals, x0, verbose=0, max_nfev=1000)
    
    # Extract optimized poses
    optimized_poses = [_unflatten_matrix(result.x[i*9:(i+1)*9]) for i in range(n)]
    
    return { pose_idx: optimized_poses[i] for i, pose_idx in enumerate(poses.keys()) }


def find_centroid_pose(poses, a_weight=1., t_weight=1./768, p_weight=1000.):
    """Find centroid pose that minimizes weighted deviation to all poses"""
    
    # Initialize at component-wise mean
    all_A, all_t, all_p = zip(*[decompose_homography(pose) for pose in poses.values()])
    
    init_centroid = np.eye(3)
    init_centroid[:2, :2] = np.mean(all_A, axis=0)
    init_centroid[:2, 2] = np.mean(all_t, axis=0)
    init_centroid[2, :2] = np.mean(all_p, axis=0)
    
    def residuals(x):
        """Per-pose weighted deviations from candidate centroid"""
        centroid_inv = np.linalg.inv(_unflatten_matrix(x))
        return [homography_error(pose @ centroid_inv, a_weight, t_weight, p_weight) 
                for pose in poses.values()]
    
    result = least_squares(residuals, _flatten_matrix(init_centroid), verbose=0)
    return _unflatten_matrix(result.x)


def get_cleaned_poses(transforms, node_idxs=None, threshold=10., a_weight=1., t_weight=1./768, p_weight=1000.):
    
    if node_idxs is None: node_idxs = _get_node_idxs(transforms)
    
    # Calculate cycle errors
    edge_errors = _calculate_cycle_errors(transforms, node_idxs,
                                          a_weight=a_weight, t_weight=t_weight, p_weight=p_weight)

    # Exclude bad transforms
    outlier_edges = set()
    for edge, errors in edge_errors.items():
        if len(errors) > 0 and np.mean(errors) > threshold:
            outlier_edges.add(edge)
    transforms_clean = {k: v for k, v in transforms.items() if k not in outlier_edges}
    
    # Compute and optimise pose graph
    poses = get_poses(transforms_clean, node_idxs)
    optimized_poses = _optimize_poses(poses, transforms_clean, a_weight=a_weight, t_weight=t_weight, p_weight=p_weight)
    
    # Find centroid
    centroid = find_centroid_pose(optimized_poses, a_weight=a_weight, t_weight=t_weight, p_weight=p_weight)
    centroid_inv = np.linalg.inv(centroid)
    
    # Update poses by centroid
    node_poses = { node_idx: pose @ centroid_inv for node_idx, pose in optimized_poses.items() }
    return node_poses


def visualise_poses(poses, node_paths=None, targ_shape=None, figsize=(8, 8)):
    from rtnls_registration.transformation import ProjectiveTransform
    
    @cache
    def get_img(img_path):
        return np.array(PILImage.open(img_path).convert('RGB'))

    # Create output widget for the plot
    output = widgets.Output()
    
    node_idxs = list(poses.keys())
    
    def show_image(idx):
        """Display the image at the given index"""
        with output:
            output.clear_output(wait=True)
            
            node_idx = node_idxs[idx]
            
            if node_paths is None:
                img_path = node_idx # Assume node_idx is the img path
            else:
                img_path = node_paths[node_idx]
            
            # Load and transform image
            image = get_img(img_path)
            transform_M = poses[node_idx]
            
            # Apply transformation
            shape = targ_shape if targ_shape is not None else image.shape[:2]
            transform = ProjectiveTransform(transform_M)
            imageT = transform.warp_inverse(image, shape)
            
            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(imageT)
            ax.axis('off')
            ax.set_title(f"{idx + 1}/{len(node_idxs)}: {os.path.basename(img_path)}", 
                        fontsize=12, pad=10)
            plt.tight_layout()
            plt.show()
    
    # Create slider
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(node_idxs) - 1,
        step=1,
        description='Image:',
        continuous_update=True,  # Only update when slider is released
        layout=widgets.Layout(width='80%')
    )
    
    # Link slider to display function
    widgets.interactive(show_image, idx=slider)
    
    # Display initial image
    show_image(0)
    
    # Display widgets
    display(slider, output)
    
    return slider, output