"""Audio/Video processing pipeline: Demucs → Denoiser (audio-denoiser) → Polish (noisereduce)."""
import os
import subprocess
import uuid
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

# Denoiser: audio-denoiser (ML-based, kompatibel dengan library terbaru)
try:
    from audio_denoiser.AudioDenoiser import AudioDenoiser
    AUDIO_DENOISER_AVAILABLE = True
except ImportError:
    AUDIO_DENOISER_AVAILABLE = False

# Polish: noisereduce (spectral gating, ringan)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False


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
            log_callback(msg)

    log("[Demucs] Memulai source separation (ekstraksi vokal)...")
    log(f"[Demucs] Input: {input_path}")

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
            log(f"[Demucs] {line.strip()}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Demucs exited with code {proc.returncode}")

        vocals_path = os.path.join(demucs_out, "htdemucs_ft", base_name, "vocals.wav")
        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"File vokal tidak ditemukan: {vocals_path}")

        log(f"[Demucs] Vokal berhasil diekstrak: {vocals_path}")
        return vocals_path

    except FileNotFoundError:
        log("[Demucs] ERROR: Demucs tidak ditemukan. Pastikan demucs terinstall: pip install demucs")
        raise
    except Exception as e:
        log(f"[Demucs] ERROR: {str(e)}")
        raise


def run_denoiser_and_polish(input_path: str, output_path: str, log_callback=None) -> str:
    """
    Denoiser (HEAVY, konservatif) → Polish (refine).
    Menggunakan audio-denoiser untuk denoising + noisereduce untuk polish ringan.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not AUDIO_DENOISER_AVAILABLE:
        raise ImportError(
            "audio-denoiser tidak terinstall. Jalankan: pip install audio-denoiser"
        )

    log("[Denoiser] Memulai denoising (HEAVY, konservatif)...")
    device = get_device()
    log(f"[Denoiser] Device: {device}")

    try:
        # Load audio
        wav, sr = torchaudio.load(input_path)
        wav = wav.mean(dim=0)  # Mono
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Step 1: audio-denoiser (HEAVY denoising)
        log("[Denoiser] Menjalankan audio-denoiser...")
        denoiser = AudioDenoiser(device=torch.device(device))
        wav_denoised = denoiser.process_waveform(wav, sr, auto_scale=True)
        log("[Denoiser] Denoising selesai.")

        # Convert to numpy for noisereduce
        wav_np = wav_denoised.squeeze().cpu().numpy()
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=0)

        # Step 2: noisereduce (polish/refine ringan - konservatif)
        if NOISEREDUCE_AVAILABLE:
            log("[Polish] Memulai polish (refine)...")
            wav_polished = nr.reduce_noise(
                y=wav_np,
                sr=sr,
                stationary=True,
                prop_decrease=0.5,  # Konservatif - tidak over-process
            )
            log("[Polish] Polish selesai.")
        else:
            wav_polished = wav_np
            log("[Denoiser] noisereduce tidak tersedia, skip polish.")

        # Save output
        sf.write(output_path, wav_polished, sr)
        log(f"[Denoiser] Output disimpan: {output_path}")

        return output_path

    except Exception as e:
        log(f"[Denoiser] ERROR: {str(e)}")
        raise


def extract_audio_from_video(video_path: str, output_path: str, log_callback=None) -> str:
    """Extract audio from video using ffmpeg."""
    def log(msg):
        if log_callback:
            log_callback(msg)

    log("[FFmpeg] Mengekstrak audio dari video...")
    import ffmpeg

    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream.audio, output_path, acodec="pcm_s16le", ac=1, ar=44100)
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
        log(f"[FFmpeg] Audio diekstrak: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log(f"[FFmpeg] ERROR: {err}")
        raise RuntimeError(f"FFmpeg extract gagal: {err}")


def merge_audio_video(video_path: str, audio_path: str, output_path: str, log_callback=None) -> str:
    """Merge enhanced audio back with original video."""
    def log(msg):
        if log_callback:
            log_callback(msg)

    log("[FFmpeg] Menggabungkan audio dengan video...")
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
        log(f"[FFmpeg] Video output: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        err = e.stderr.decode() if e.stderr else str(e)
        log(f"[FFmpeg] ERROR: {err}")
        raise RuntimeError(f"FFmpeg merge gagal: {err}")


def process_audio(input_path: str, output_dir: str, log_callback=None) -> str:
    """
    Full audio pipeline: Demucs → Denoiser → Polish.
    Returns path to output file.
    """
    task_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(work_dir, exist_ok=True)

    vocals_path = run_demucs(input_path, work_dir, log_callback)

    base_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{base_name}_enhanced.wav")
    run_denoiser_and_polish(vocals_path, output_path, log_callback)

    return output_path


def process_video(input_path: str, output_dir: str, log_callback=None) -> str:
    """
    Full video pipeline: Extract → Demucs → Denoiser → Polish → Merge.
    Returns path to output video file.
    """
    task_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(work_dir, exist_ok=True)

    extracted_audio = os.path.join(work_dir, "extracted_audio.wav")
    extract_audio_from_video(input_path, extracted_audio, log_callback)

    vocals_path = run_demucs(extracted_audio, work_dir, log_callback)

    enhanced_audio = os.path.join(work_dir, "enhanced_audio.wav")
    run_denoiser_and_polish(vocals_path, enhanced_audio, log_callback)

    base_name = Path(input_path).stem
    ext = Path(input_path).suffix
    output_path = os.path.join(output_dir, f"{base_name}_enhanced{ext}")
    merge_audio_video(input_path, enhanced_audio, output_path, log_callback)

    return output_path
