"""
Motion-aware metrics for analyzing tracking performance on static vs dynamic points.

This addresses Reviewer UTcz and b4nj's request for motion regime breakdowns.
"""
import torch
from metrics.tapvid360_metrics import calc_angular_dist, compute_metrics


def classify_points_by_motion(
    gt_trajectory: torch.Tensor,
    rotations: torch.Tensor,
    static_threshold_degrees: float = 2.0,
) -> dict:
    """
    Classify points as static or dynamic based on their world-frame motion.
    
    Args:
        gt_trajectory: (T, N, 3) unit vectors in camera coordinates
        rotations: (T, 3, 3) camera rotation matrices (R_w2c)
        static_threshold_degrees: points with total world motion below this are static
    
    Returns:
        dict with:
            - static_mask: (N,) boolean - True for static points
            - dynamic_mask: (N,) boolean - True for dynamic points  
            - total_object_motion: (N,) angular displacement in degrees per point
    """
    T, N, _ = gt_trajectory.shape
    
    # Transform all points to world coordinates
    # R_w2c^(-1) = R_w2c^T (rotation matrices are orthogonal)
    world_positions = []
    for t in range(T):
        # vc_to_vw: vc -> vw using R^(-1) = R^T
        # world_pos = R[t]^T @ vc[t]
        R_inv = rotations[t].T  # (3, 3)
        vc = gt_trajectory[t]   # (N, 3)
        vw = torch.einsum('ij,nj->ni', R_inv, vc)  # (N, 3)
        vw = torch.nn.functional.normalize(vw, dim=-1)
        world_positions.append(vw)
    
    world_positions = torch.stack(world_positions)  # (T, N, 3)
    
    # Compute frame-to-frame angular displacement in world frame
    world_angular_displacement = calc_angular_dist(
        world_positions[:-1].unsqueeze(0), 
        world_positions[1:].unsqueeze(0)
    ).squeeze(0)  # (T-1, N) in degrees
    
    # Total motion per point
    total_object_motion = world_angular_displacement.sum(dim=0)  # (N,) degrees
    
    # Classify
    static_mask = total_object_motion < static_threshold_degrees
    dynamic_mask = ~static_mask
    
    return {
        'static_mask': static_mask,
        'dynamic_mask': dynamic_mask,
        'total_object_motion': total_object_motion,
    }


def bin_points_by_motion(
    total_motion: torch.Tensor,
    bins: list = [0, 5, 30, float('inf')],
) -> dict:
    """
    Bin points by their total motion magnitude.
    
    Args:
        total_motion: (N,) angular displacement in degrees
        bins: list of bin edges
    
    Returns:
        dict mapping bin names to boolean masks
    """
    masks = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        if high == float('inf'):
            name = f"motion_{low}+"
        else:
            name = f"motion_{low}-{high}"
        masks[name] = (total_motion >= low) & (total_motion < high)
    return masks


def compute_metrics_by_motion_type(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_visibility: torch.Tensor,
    rotations: torch.Tensor,
    static_threshold_degrees: float = 2.0,
) -> dict:
    """
    Compute metrics broken down by static vs dynamic points.
    
    Args:
        pred_trajectory: (T, N, 3) predicted unit vectors
        gt_trajectory: (T, N, 3) ground truth unit vectors
        gt_visibility: (T, N) visibility mask
        rotations: (T, 3, 3) camera rotation matrices
        static_threshold_degrees: threshold for static classification
    
    Returns:
        dict with overall and per-motion-type metrics
    """
    # Standard metrics
    overall_metrics = compute_metrics(pred_trajectory, gt_trajectory, gt_visibility)
    
    # Classify points
    motion_info = classify_points_by_motion(gt_trajectory, rotations, static_threshold_degrees)
    static_mask = motion_info['static_mask']
    dynamic_mask = motion_info['dynamic_mask']
    
    result = {
        'overall': {k: v.mean().item() for k, v in overall_metrics.items()},
        'num_static_points': int(static_mask.sum()),
        'num_dynamic_points': int(dynamic_mask.sum()),
    }
    
    # Compute metrics for static points only
    if static_mask.any():
        static_metrics = compute_metrics(
            pred_trajectory[:, static_mask], 
            gt_trajectory[:, static_mask], 
            gt_visibility[:, static_mask]
        )
        result['static'] = {k: v.mean().item() for k, v in static_metrics.items()}
    
    # Compute metrics for dynamic points only
    if dynamic_mask.any():
        dynamic_metrics = compute_metrics(
            pred_trajectory[:, dynamic_mask], 
            gt_trajectory[:, dynamic_mask], 
            gt_visibility[:, dynamic_mask]
        )
        result['dynamic'] = {k: v.mean().item() for k, v in dynamic_metrics.items()}
    
    # Bin by motion magnitude
    motion_bins = bin_points_by_motion(motion_info['total_object_motion'], bins=[0, 10, 30, float('inf')])
    for bin_name, bin_mask in motion_bins.items():
        if bin_mask.any():
            bin_metrics = compute_metrics(
                pred_trajectory[:, bin_mask],
                gt_trajectory[:, bin_mask],
                gt_visibility[:, bin_mask]
            )
            result[bin_name] = {k: v.mean().item() for k, v in bin_metrics.items()}
    
    return result
