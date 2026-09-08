import argparse
from pathlib import Path

from tracker_engine import PersistentPersonTracker

VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def pick_latest_video(folder):
    videos = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_TYPES
    ]
    if not videos:
        raise FileNotFoundError(
            "No supported video was found in the input folder. "
            "Drop a .mp4, .avi, .mov, .mkv, .webm, or .m4v file into input/ and run again."
        )
    return max(videos, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Track people in a video and save the output to the output folder."
    )
    parser.add_argument("--video", default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Defaults to custom_human_model.pt if present (see training/), else yolo11s.pt.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.model is None:
        custom_model = root / "custom_human_model.pt"
        args.model = str(custom_model) if custom_model.exists() else "yolo11s.pt"
    source = Path(args.video) if args.video else root / "input"

    if source.is_dir():
        video_path = pick_latest_video(source)
    else:
        video_path = source

    if not video_path.is_absolute():
        video_path = root / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = (
        Path(args.output)
        if args.output
        else root / "output" / f"tracked_{video_path.stem}.mp4"
    )
    if not output_path.is_absolute():
        output_path = root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing: {video_path}")
    print(f"Saving to: {output_path}")
    print(f"Model: {args.model}")

    PersistentPersonTracker(args.model).process_video(video_path, output_path)


if __name__ == "__main__":
    main()
