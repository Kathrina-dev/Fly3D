from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .pipeline import PipelineError, ScenePaths, export_splat, ingest, process_video, train, write_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video to Gaussian splat pipeline using Nerfstudio.")
    parser.add_argument("--scene", type=Path, default=Path("scenes/room"), help="Scene working directory.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="Register a phone video as the scene input.")
    ingest_parser.add_argument("--scene", type=Path, default=Path("scenes/room"))
    ingest_parser.add_argument("video", type=Path)
    process_parser = commands.add_parser("process", help="Extract frames and recover camera poses with COLMAP.")
    process_parser.add_argument("--scene", type=Path, default=Path("scenes/room"))
    process_parser.add_argument("--frames", type=int, default=250)
    process_parser.add_argument("--downscales", type=int, default=3)
    train_parser = commands.add_parser("train", help="Train a low-VRAM splatfacto model.")
    train_parser.add_argument("--scene", type=Path, default=Path("scenes/room"))
    train_parser.add_argument("--iterations", type=int, default=15000)
    train_parser.add_argument("--resolution-scale", type=float, default=0.5)
    export_parser = commands.add_parser("export", help="Export the latest checkpoint as a PLY Gaussian splat.")
    export_parser.add_argument("--scene", type=Path, default=Path("scenes/room"))
    export_parser.add_argument("--config", type=Path)
    run_parser = commands.add_parser("run", help="Ingest, process, train, and export in one command.")
    run_parser.add_argument("--scene", type=Path, default=Path("scenes/room"))
    run_parser.add_argument("video", type=Path)
    run_parser.add_argument("--frames", type=int, default=250)
    run_parser.add_argument("--downscales", type=int, default=3)
    run_parser.add_argument("--iterations", type=int, default=15000)
    run_parser.add_argument("--resolution-scale", type=float, default=0.5)
    status_parser = commands.add_parser("status", help="Show which pipeline stages are complete.")
    status_parser.add_argument("--scene", type=Path, default=Path("scenes/room"))
    commands.add_parser("doctor", help="Check required commands and NVIDIA visibility.")
    return parser


def doctor() -> int:
    checks = ["ns-process-data", "ns-train", "ns-export", "colmap", "ffmpeg"]
    missing = [command for command in checks if shutil.which(command) is None]
    for command in checks:
        print(f"{'OK  ' if command not in missing else 'MISS'} {command}")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False, text=True, capture_output=True)
        print(result.stdout.strip() or "MISS nvidia-smi / no NVIDIA GPU visible")
    except FileNotFoundError:
        print("MISS nvidia-smi / no NVIDIA driver visible")
    return 1 if missing else 0


def main() -> int:
    args = build_parser().parse_args()
    scene = ScenePaths(args.scene.expanduser().resolve())
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "ingest":
            ingest(args.video, scene)
        elif args.command == "process":
            process_video(scene, args.frames, args.downscales)
        elif args.command == "train":
            train(scene, args.iterations, args.resolution_scale)
        elif args.command == "export":
            print(f"Exported {export_splat(scene, args.config)}")
        elif args.command == "run":
            ingest(args.video, scene)
            process_video(scene, args.frames, args.downscales)
            train(scene, args.iterations, args.resolution_scale)
            print(f"Exported {export_splat(scene)}")
        elif args.command == "status":
            write_status(scene)
    except PipelineError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())