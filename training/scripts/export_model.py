"""
Export a trained checkpoint to the formats the rest of the project (and
any downstream deployment) needs: keep best.pt/last.pt as-is, and export
ONNX + TorchScript for portability/non-Ultralytics inference.

After exporting, copies best.pt to the repo root as `custom_human_model.pt`
so main.py / tracker_engine.py can pick it up automatically (see
`PersistentPersonTracker`'s default model resolution in tracker_engine.py).

Usage:
    python training/scripts/export_model.py --weights training/runs/surveillance_human_detector/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from common import REPO_ROOT, load_config, resolve_device


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX/TorchScript and deploy it.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--no-deploy", action="store_true", help="Skip copying to the repo root.")
    args = parser.parse_args()

    config = load_config()
    imgsz = args.imgsz or config["model"]["imgsz"]
    device = resolve_device(config["project"]["device"])

    model = YOLO(str(args.weights))

    print("Exporting ONNX...")
    onnx_path = model.export(format="onnx", imgsz=imgsz, half=config["inference"]["half"], simplify=True, device=device)
    print(f"  -> {onnx_path}")

    print("Exporting TorchScript...")
    torchscript_path = model.export(format="torchscript", imgsz=imgsz, device=device)
    print(f"  -> {torchscript_path}")

    last_path = args.weights.parent / "last.pt"
    print(f"\nbest.pt : {args.weights}")
    print(f"last.pt : {last_path if last_path.exists() else '(not found next to best.pt)'}")

    if not args.no_deploy:
        deploy_path = REPO_ROOT / "custom_human_model.pt"
        shutil.copyfile(args.weights, deploy_path)
        print(f"\nDeployed to: {deploy_path}")
        print("tracker_engine.py / main.py will auto-detect and use this file.")
        print("(Delete it, or pass --model yolo11s.pt, to fall back to the stock model.)")


if __name__ == "__main__":
    main()
