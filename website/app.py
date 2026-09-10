from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import subprocess
import sys
import uuid

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "website" / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = BASE_DIR / "custom_human_model.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No video uploaded.", 400

    video = request.files["video"]

    if video.filename == "":
        return "No video selected.", 400

    # Keep original extension
    extension = Path(video.filename).suffix.lower()

    if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        return "Unsupported video format.", 400

    # Create a unique filename
    job_id = uuid.uuid4().hex[:8]

    input_filename = f"input_{job_id}{extension}"
    output_filename = f"tracked_{job_id}.mp4"

    input_path = UPLOAD_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename

    # Save uploaded video
    video.save(str(input_path))

    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")

    if not MODEL_PATH.exists():
        return f"Model not found: {MODEL_PATH}", 500

    # IMPORTANT:
    # main.py expects a VIDEO FILE as --output, not the output directory.
    command = [
        sys.executable,
        str(BASE_DIR / "main.py"),
        "--video",
        str(input_path),
        "--model",
        str(MODEL_PATH),
        "--output",
        str(output_path),
    ]

    print("Running:")
    print(" ".join(command))

    try:
        result = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)

        print("STDERR:")
        print(result.stderr)

        if result.returncode != 0:
            return render_template(
                "error.html",
                error=result.stderr or result.stdout or "Processing failed."
            ), 500

    except Exception as e:
        return render_template("error.html", error=str(e)), 500

    if not output_path.exists():
        return render_template(
            "error.html",
            error="Processing finished, but output video was not created."
        ), 500

    return render_template(
        "result.html",
        video_name=output_filename
    )


@app.route("/output/<path:filename>")
def output_video(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

