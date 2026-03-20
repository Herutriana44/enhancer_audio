# Audio Enhancer

Aplikasi Flask untuk memperbaiki kualitas audio dengan pipeline: **Demucs → Denoiser (audio-denoiser) → Polish (noisereduce)**.

## Flow

### Audio Input
```
Audio (MP3, WAV, dll) → Demucs (source separation/vokal) → Denoiser (HEAVY, konservatif) → Polish (refine) → Output
```

### Video Input
```
Video → Extract audio → Demucs → Denoiser → Polish → Merge video
```

## Instalasi

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/enhancer_audio.git
cd enhancer_audio

# Buat virtual environment (opsional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (diperlukan untuk video)
# Ubuntu/Debian: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

## Menjalankan

```bash
python app.py
```

Buka http://localhost:5000 di browser.

## Colab/Kaggle

Gunakan notebook `run_colab_kaggle.ipynb`:
1. Upload ke Google Colab atau Kaggle
2. Ganti `REPO_URL` dengan URL repository Anda
3. Jalankan semua cell
4. Buka URL ngrok yang ditampilkan

## Dependencies

- **Demucs**: Source separation, ekstraksi vokal
- **audio-denoiser**: Denoising (HEAVY, ML-based)
- **noisereduce**: Polish/refine (spectral gating, konservatif)
- **FFmpeg**: Ekstraksi audio dari video, merge video
