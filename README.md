# Fly3D

Fly3D ingests a phone video and drives a low-VRAM Nerfstudio Gaussian-splat pipeline. It is designed for Ubuntu with an NVIDIA GPU under 8 GB VRAM: one room per scene, sequential COLMAP matching, downscaled training, and plain `splatfacto`.

## Setup

Install Miniconda, then from this directory run:

```bash
chmod +x setup.sh
./setup.sh
conda activate fly3d
fly3d doctor
```

The setup script installs FFmpeg, COLMAP, GLOMAP, Nerfstudio, and the CUDA 11.8 PyTorch build. Keep the NVIDIA driver installed and verify it with `nvidia-smi` first.

## Run a room

```bash
fly3d run ~/Videos/room_4k.mp4 --scene scenes/living-room
```

Defaults are conservative: 250 frames, three downscale levels, 15,000 iterations, and 0.5 camera resolution. To run stages independently:

```bash
fly3d ingest ~/Videos/room_4k.mp4 --scene scenes/living-room
fly3d process --scene scenes/living-room
fly3d train --scene scenes/living-room --iterations 7000 --resolution-scale 0.35
fly3d export --scene scenes/living-room
fly3d status --scene scenes/living-room
```

The exported `splat.ply` can be cleaned and converted to `.spz` in SuperSplat. Nerfstudio's training viewer normally appears at `http://localhost:7007` while `train` is running.

## Capture guidance

Use the main 1x camera at 4K/30, lock exposure and focus, walk slowly with strong overlap, and capture one room at a time. Avoid motion blur, mirrors, blank walls, and changing light. A 60-120 second clip is usually enough.

## Notes

The Python wrapper does not fake a splat detector: Nerfstudio and COLMAP do the actual pose recovery and Gaussian optimization. If training runs out of memory, reduce the scene to roughly 150 frames, use `--resolution-scale 0.35`, and stop around 7,000 iterations.