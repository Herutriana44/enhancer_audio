"""Configuration for Audio Enhancer Flask App."""
import os

# Upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")
PROCESSING_FOLDER = os.path.join(os.path.dirname(__file__), "processing")

# Allowed extensions
AUDIO_EXTENSIONS = {"mp3", "wav", "flac", "m4a", "ogg", "aac", "wma"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mkv", "mov", "webm", "flv", "wmv"}

# Max file size (100MB)
MAX_CONTENT_LENGTH = 100 * 1024 * 1024

# Version info
VERSION = "1.0.0"
