"""Small helpers shared by ``run_tapout_release.py`` and ``scripts/demo.py``."""

import torch

from runners.model_utils import decode_latents


def run_through_cotracker(co_tracker, batchsize, device, persp_imgs, persp_pos_points):
    video_resize, _, query_points = co_tracker.preprocess(persp_imgs, query_points=persp_pos_points,
                                                          seg_mask=None, device=device)
    all_pred_tracks = []
    all_pred_vis = []
    for bs in range(0, len(persp_imgs), batchsize):
        pred_tracks_batch, pred_vis_batch = co_tracker.run(video_resize[bs:bs + batchsize].contiguous(),
                                                           query_points=query_points[bs:bs + batchsize].contiguous())
        all_pred_tracks.append(pred_tracks_batch)
        all_pred_vis.append(pred_vis_batch)
    all_pred_tracks = torch.vstack(all_pred_tracks)
    all_pred_vis = torch.vstack(all_pred_vis)
    # Rescale the cotracker points back to expected image size
    if co_tracker.interp_shape is not None:
        all_pred_tracks = co_tracker.rescale_points(all_pred_tracks, co_tracker.interp_shape,
                                                    persp_imgs.shape[-3:-1])

    return all_pred_tracks, all_pred_vis


def _decode_pred_eq_frames(argus_latents, settings, vae):
    pred_eq_frames = decode_latents(vae.eval(), argus_latents[None].to(vae.dtype), argus_latents.shape[0],
                                    settings.decode_chunk_size, False, 4)
    pred_eq_frames_torch = ((pred_eq_frames[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return pred_eq_frames_torch


def _resize_eq_frames(eq_frames, new_shape=(1024, 2048)):
    return torch.nn.functional.interpolate(eq_frames.permute(0, 3, 1, 2).float(),
                                           new_shape, mode='bicubic',
                                           align_corners=True).permute(0, 2, 3, 1).clamp(0, 255).byte()
