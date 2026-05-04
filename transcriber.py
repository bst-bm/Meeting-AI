"""Audio/Video transcription via faster-whisper (fully local, no internet needed)."""

import os
import subprocess
import tempfile
from pathlib import Path

# Prevent faster-whisper from reaching out to HuggingFace after first download
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from faster_whisper import WhisperModel

SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mkv", ".avi"}


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        model_size: tiny | base | small | medium | large-v3
        device:     auto | cpu | cuda  (ROCm maps to cuda in CTranslate2)
        """
        print(f"  Loading Whisper model '{model_size}' ({device})...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        print("  Whisper ready.")

    def transcribe(self, file_path: str) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty (0 bytes): {path.name}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        print(f"  Transcribing: {path.name}")

        # Always pre-process through ffmpeg → clean 16kHz mono WAV
        # This handles broken/incomplete containers, unusual codecs, etc.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _extract_audio(str(path), tmp_path)
            segments, info = self.model.transcribe(tmp_path, beam_size=5)

            result = {
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": [],
            }

            for seg in segments:
                result["segments"].append({
                    "start": round(seg.start, 2),
                    "end":   round(seg.end, 2),
                    "text":  seg.text.strip(),
                })

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        print(
            f"  Done — {len(result['segments'])} segments, "
            f"language={info.language}, duration={result['duration']}s"
        )
        return result

    def full_text(self, transcription: dict) -> str:
        return " ".join(seg["text"] for seg in transcription["segments"])

    def timed_text(self, transcription: dict) -> str:
        lines = []
        for seg in transcription["segments"]:
            start = _fmt_time(seg["start"])
            end   = _fmt_time(seg["end"])
            lines.append(f"[{start} → {end}] {seg['text']}")
        return "\n".join(lines)


def _extract_audio(input_path: str, output_wav: str) -> None:
    """Convert any audio/video to 16kHz mono WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", input_path,
        "-vn",               # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not process '{Path(input_path).name}'.\n"
            f"Is the file a valid audio/video recording?\n"
            f"ffmpeg error: {result.stderr[-400:]}"
        )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
