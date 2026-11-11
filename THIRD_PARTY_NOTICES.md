# Third-party notices

TAP-Out's own code is released under the MIT License (see [LICENSE](LICENSE)).
The components below are **not** covered by that license.

## Vendored code

### Argus — `third_party/argus/`

Adapted from [Red-Fairy/argus-code](https://github.com/Red-Fairy/argus-code)
(Luo et al., *Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos*, ICCV 2025).
Files have been trimmed to what TAP-Out uses. The upstream repository does not
state a license; all rights in this code remain with the Argus authors.

`pipelines/argus/custom_argus_pipeline.py` is a TAP-Out modification of the Argus
sampling pipeline, which itself builds on the Stable Video Diffusion pipeline from
[huggingface/diffusers](https://github.com/huggingface/diffusers) (Apache License 2.0).

## Dependencies and model weights

These are installed or downloaded at setup/run time and are governed by their own licenses:

| Component | Used for | Code license | Weights license |
| --- | --- | --- | --- |
| [Argus](https://github.com/Red-Fairy/argus-code) | 360° out-painting U-Net | not stated | not stated |
| [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) | VAE + image encoder | — | Stability AI Community License |
| [SAM 2](https://github.com/facebookresearch/sam2) | Segmentation | Apache 2.0 | Apache 2.0 |
| [VGGT](https://github.com/facebookresearch/vggt) | Camera poses | see repo | CC BY-NC 4.0 (`facebook/VGGT-1B`) |
| [CoTracker3](https://github.com/facebookresearch/co-tracker) | 2D point tracking | see repo | CC BY-NC 4.0 (`facebook/cotracker3`) |

Because VGGT-1B and CoTracker3 weights are licensed for non-commercial use, the
full pipeline as shipped is suitable for research and other non-commercial use only.
