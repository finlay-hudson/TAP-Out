import torch


def rot_x(alpha: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices around the X-axis. For our purpose this is a roll rotation"""
    if degrees:
        alpha = torch.deg2rad(alpha)  # Convert to radians if needed

    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)

    R_x = torch.stack([
        torch.ones_like(alpha), torch.zeros_like(alpha), torch.zeros_like(alpha),
        torch.zeros_like(alpha), cos_a, -sin_a,
        torch.zeros_like(alpha), sin_a, cos_a
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

    return R_x


def rot_y(beta: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices around the Y-axis. For our purpose this is a up/down rotation"""
    if degrees:
        beta = torch.deg2rad(beta)

    cos_b, sin_b = torch.cos(beta), torch.sin(beta)

    R_y = torch.stack([
        cos_b, torch.zeros_like(beta), sin_b,
        torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
        -sin_b, torch.zeros_like(beta), cos_b
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

    return R_y


def rot_z(gamma: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices around the Z-axis. For our purpose this is a left/right rotation"""
    if degrees:
        gamma = torch.deg2rad(gamma)

    cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)

    R_z = torch.stack([
        cos_g, -sin_g, torch.zeros_like(gamma),
        sin_g, cos_g, torch.zeros_like(gamma),
        torch.zeros_like(gamma), torch.zeros_like(gamma), torch.ones_like(gamma)
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

    return R_z


def rot(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, degrees=False) -> torch.Tensor:
    """Computes batched 3x3 rotation matrices for given roll (α), pitch (β), and yaw (γ)."""
    assert alpha.shape == beta.shape == gamma.shape, "Input angles must have the same shape"

    R_x = rot_x(alpha, degrees)  # (B, 3, 3)
    R_y = rot_y(beta, degrees)  # (B, 3, 3)
    R_z = rot_z(gamma, degrees)  # (B, 3, 3)

    # Batched matrix multiplication -> (B, 3, 3), defaults to bfloat16 but we want float32 later on
    return (R_x @ R_y @ R_z).to(torch.float32)
