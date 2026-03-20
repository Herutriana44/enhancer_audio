"""Flask Audio Enhancer - Demucs → Denoiser → Resemble Enhance."""
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from config import (
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    VERSION,
)
from processor import process_audio, process_video

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)


def get_version_info():
    """Get version and dependency info for log."""
    info = []
    info.append(f"=== Audio Enhancer v{VERSION} ===")
    info.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        import torch
        info.append(f"PyTorch: {torch.__version__}")
        info.append(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        info.append("PyTorch: not installed")
    try:
        import demucs
        info.append(f"Demucs: {getattr(demucs, '__version__', 'installed')}")
    except ImportError:
        info.append("Demucs: not installed")
    try:
        import audio_denoiser
        info.append("audio-denoiser: installed")
    except ImportError:
        info.append("audio-denoiser: not installed")
    try:
        import noisereduce
        info.append("noisereduce: installed")
    except ImportError:
        info.append("noisereduce: not installed")
    return "\n".join(info)


@app.route("/")
def index():
    """Main page with upload form and log display."""
    return render_template("index.html", version=VERSION)


@app.route("/api/version")
def api_version():
    """Return version info for log display."""
    return jsonify({"version": VERSION, "info": get_version_info()})


@app.route("/api/process", methods=["POST"])
def api_process():
    """Process uploaded file (audio or video)."""
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File tidak dipilih"}), 400

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        return jsonify({
            "error": f"Format tidak didukung. Audio: {', '.join(AUDIO_EXTENSIONS)}. Video: {', '.join(VIDEO_EXTENSIONS)}"
        }), 400

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(input_path)

    logs = []
    is_audio = ext in AUDIO_EXTENSIONS

    def log_callback(msg):
        logs.append({"time": datetime.now().strftime("%H:%M:%S"), "msg": msg})

    try:
        if is_audio:
            output_path = process_audio(input_path, OUTPUT_FOLDER, log_callback)
        else:
            output_path = process_video(input_path, OUTPUT_FOLDER, log_callback)

        return jsonify({
            "success": True,
            "output_file": os.path.basename(output_path),
            "logs": logs,
        })
    except Exception as e:
        log_callback(f"ERROR: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "logs": logs,
        }), 500
    finally:
        # Cleanup input
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            pass


@app.route("/api/download/<filename>")
def api_download(filename):
    """Download processed file."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path) or not os.path.abspath(path).startswith(os.path.abspath(OUTPUT_FOLDER)):
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    print(get_version_info())
    app.run(host="0.0.0.0", port=5000, debug=True)
