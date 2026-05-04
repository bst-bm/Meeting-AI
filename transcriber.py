"""Audio/Video transcription via faster-whisper (local, GPU-accelerated)."""

from faster_whisper import WhisperModel
from pathlib import Path


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        model_size: tiny | base | small | medium | large-v3
        device:     auto | cpu | cuda  (ROCm maps to cuda in CTranslate2)
        """
        print(f"  Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        print("  Whisper ready.")

    def transcribe(self, file_path: str) -> dict:
        """Transcribe an audio/video file. Returns segments + metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"  Transcribing: {path.name}")
        segments, info = self.model.transcribe(str(path), beam_size=5)

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

        print(f"  Done — {len(result['segments'])} segments, "
              f"language={info.language}, duration={result['duration']}s")
        return result

    def full_text(self, transcription: dict) -> str:
        return " ".join(seg["text"] for seg in transcription["segments"])

    def timed_text(self, transcription: dict) -> str:
        """Human-readable transcript with timestamps."""
        lines = []
        for seg in transcription["segments"]:
            start = _fmt_time(seg["start"])
            end   = _fmt_time(seg["end"])
            lines.append(f"[{start} → {end}] {seg['text']}")
        return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
