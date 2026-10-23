from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot run safely."""


@dataclass(frozen=True)
class ScenePaths:
    root: Path

    @property
    def input_video(self) -> Path:
        return self.root / "input.mp4"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def exports(self) -> Path:
        return self.root / "exports"


def run_command(args: Sequence[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(args))
    try:
        subprocess.run(args, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(f"Required command is not installed: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Command failed with exit code {exc.returncode}: {args[0]}") from exc


def ingest(video: Path, scene: ScenePaths) -> Path:
    video = video.expanduser().resolve()
    if not video.is_file():
        raise PipelineError(f"Video does not exist: {video}")
    if video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise PipelineError("Use an MP4, MOV, MKV, or WebM video.")
    scene.root.mkdir(parents=True, exist_ok=True)
    if scene.input_video.exists() or scene.input_video.is_symlink():
        scene.input_video.unlink()
    scene.input_video.symlink_to(video)
    print(f"Ingested {video.name} -> {scene.input_video}")
    return scene.input_video


def process_video(scene: ScenePaths, frames: int, downscales: int) -> None:
    if not scene.input_video.exists():
        raise PipelineError(f"Ingest a video first: {scene.input_video}")
    run_command(["ns-process-data", "video", "--data", str(scene.input_video), "--output-dir", str(scene.processed), "--num-frames-target", str(frames), "--sfm-tool", "colmap", "--matching-method", "sequential", "--num-downscales", str(downscales)])


def train(scene: ScenePaths, iterations: int, resolution_scale: float) -> None:
    if not (scene.processed / "transforms.json").exists():
        raise PipelineError("Processed data is missing transforms.json. Run `fly3d process` first.")
    run_command(["ns-train", "splatfacto", "--data", str(scene.processed), "--max-num-iterations", str(iterations), "--pipeline.datamanager.camera-res-scale-factor", str(resolution_scale)])


def find_config(scene: ScenePaths) -> Path:
    configs = sorted(scene.outputs.glob("**/config.yml"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not configs:
        raise PipelineError(f"No Nerfstudio config found below {scene.outputs}")
    return configs[0]


def export_splat(scene: ScenePaths, config: Path | None = None) -> Path:
    config = config or find_config(scene)
    scene.exports.mkdir(parents=True, exist_ok=True)
    run_command(["ns-export", "gaussian-splat", "--load-config", str(config), "--output-dir", str(scene.exports)])
    ply = scene.exports / "splat.ply"
    if not ply.exists():
        raise PipelineError(f"Export completed but {ply} was not created.")
    return ply


def scene_status(scene: ScenePaths) -> dict[str, object]:
    return {"root": str(scene.root), "input_video": scene.input_video.exists(), "processed_data": (scene.processed / "transforms.json").exists(), "checkpoints": len(list(scene.outputs.glob("**/*.ckpt"))), "ply_exports": [str(path) for path in scene.exports.glob("*.ply")]}


def write_status(scene: ScenePaths) -> None:
    print(json.dumps(scene_status(scene), indent=2))