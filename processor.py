"""Audio/Video processing pipeline: Demucs → Denoiser → Resemble Enhance."""
import os
import subprocess
import uuid
from pathlib import Path

import torch
import torchaudio

# Resemble Enhance imports (optional - may fail if not installed)
try:
    from resemble_enhance.enhancer.inference import denoise, enhance
    RESEMBLE_AVAILABLE = True
except ImportError:
    RESEMBLE_AVAILABLE = False


def get_device():
    """Get available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_demucs(input_path: str, output_dir: str, log_callback=None) -> str:
    """
    Run Demucs for source separation - extract vocals only.
    Returns path to vocals.wav file.
    """
    def log(msg):
        if log_callback:
            log_callback(f"[Demucs] {msg}")

    log("Memulai source separation (ekstraksi vokal)...")
    log(f"Input: {input_path}")

    # Demucs output: separated/htdemucs_ft/{filename}/vocals.wav
    base_name = Path(input_path).stem
    demucs_out = os.path.join(output_dir, "demucs_output")

    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs_ft",
        "-o", demucs_out,
        input_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log(line.strip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Demucs exited with code {proc.returncode}")

        vocals_path = os.path.join(demucs_out, "htdemucs_ft", base_name, "vocals.wav")
        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"File vokal tidak ditemukan: {vocals_path}")

        log(f"Vokal berhasil diekstrak: {vocals_path}")
        return vocals_path

    except FileNotFoundError as e:
        log(f"ERROR: Demucs tidak ditemukan. Pastikan demucs terinstall: pip install demucs")
        raise
    except Exception as e:
        log(f"ERROR: {str(e)}")
        raise


def run_denoiser_and_enhance(input_path: str, output_path: str, log_callback=None) -> str:
    """
    Run Resemble Enhance: Denoiser (HEAVY, conservative) → Enhance (REFINE/polish).
    HEAVY denoising = full denoise pass.
    Conservative = lambd 0.4 for enhance to avoid over-processing.
    """
    def log(msg):
        if log_callback:
            log_callback(f"[Resemble Enhance] {msg}")

    if not RESEMBLE_AVAILABLE:
        raise ImportError("resemble-enhance tidak terinstall. Jalankan: pip install resemble-enhance")

    log("Memulai denoising (HEAVY, konservatif)...")
    device = get_device()
    log(f"Device: {device}")

    try:
        wav, sr = torchaudio.load(input_path)
        wav = wav.mean(dim=0)  # Mono

        # Step 1: Denoise (HEAVY - full denoise)
        log("Menjalankan denoiser...")
        wav_denoised, new_sr = denoise(wav, sr, device)
        log("Denoising selesai.")

        # Step 2: Enhance (REFINE/polish - conservative lambd=0.4)
        log("Memulai enhancement (polish)...")
        wav_enhanced, new_sr = enhance(
            wav_denoised,
            new_sr,
            device,
            nfe=64,
            solver="midpoint",
            lambd=0.4,  # Conservative - avoid over-processing
            tau=0.5,
        )
        log("Enhancement selesai.")

        # Save output
        wav_np = wav_enhanced.cpu().numpy()
        import numpy as np
        import soundfile as sf
        sf.write(output_path, wav_np, new_sr)
        log(f"Output disimpan: {output_path}")

        return output_path

    except Exception as e:
        log(f"ERROR: {str(e)}")
        raise


def extract_audio_from_video(video_path: str, output_path: str, log_callback=None) -> str:
    """Extract audio from video using ffmpeg."""
    def log(msg):
        if log_callback:
            log_callback(f"[FFmpeg Extract] {msg}")

    log("Mengekstrak audio dari video...")
    import ffmpeg

    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream.audio, output_path, acodec="pcm_s16le", ac=1, ar=44100)
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
        log(f"Audio diekstrak: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log(f"ERROR: {err}")
        raise RuntimeError(f"FFmpeg extract gagal: {err}")


def merge_audio_video(video_path: str, audio_path: str, output_path: str, log_callback=None) -> str:
    """Merge enhanced audio back with original video."""
    def log(msg):
        if log_callback:
            log_callback(f"[FFmpeg Merge] {msg}")

    log("Menggabungkan audio dengan video...")
    import ffmpeg

    try:
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        out = ffmpeg.output(
            video.video,
            audio.audio,
            output_path,
            vcodec="copy",
            acodec="aac",
        )
        ffmpeg.run(out, overwrite_output=True, capture_stderr=True)
        log(f"Video output: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log(f"ERROR: {err}")
        raise RuntimeError(f"FFmpeg merge gagal: {err}")


def process_audio(input_path: str, output_dir: str, log_callback=None) -> str:
    """
    Full audio pipeline: Demucs → Denoiser → Resemble Enhance.
    Returns path to output file.
    """
    task_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Demucs
    vocals_path = run_demucs(input_path, work_dir, log_callback)

    # Step 2 & 3: Denoiser + Enhance
    base_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{base_name}_enhanced.wav")
    run_denoiser_and_enhance(vocals_path, output_path, log_callback)

    return output_path


def process_video(input_path: str, output_dir: str, log_callback=None) -> str:
    """
    Full video pipeline: Extract → Demucs → Denoiser → Enhance → Merge.
    Returns path to output video file.
    """
    task_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Extract audio
    extracted_audio = os.path.join(work_dir, "extracted_audio.wav")
    extract_audio_from_video(input_path, extracted_audio, log_callback)

    # Step 2: Demucs
    vocals_path = run_demucs(extracted_audio, work_dir, log_callback)

    # Step 3 & 4: Denoiser + Enhance
    enhanced_audio = os.path.join(work_dir, "enhanced_audio.wav")
    run_denoiser_and_enhance(vocals_path, enhanced_audio, log_callback)

    # Step 5: Merge
    base_name = Path(input_path).stem
    ext = Path(input_path).suffix
    output_path = os.path.join(output_dir, f"{base_name}_enhanced{ext}")
    merge_audio_video(input_path, enhanced_audio, output_path, log_callback)

    return output_path
