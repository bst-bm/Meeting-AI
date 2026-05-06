"""Audio/Video transcription via WhisperX with optional speaker diarization."""

import os
import subprocess
import tempfile
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mkv", ".avi"}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        diarize: bool = False,
        hf_token: str | None = None,
    ):
        import whisperx

        self._diarize = diarize
        self._hf_token = hf_token
        self._device = _resolve_device(device)

        print(f"  Loading Whisper model '{model_size}' ({self._device})...")
        self.model = whisperx.load_model(model_size, self._device, compute_type="int8")
        print("  Whisper ready.")

    def transcribe(self, file_path: str) -> dict:
        import whisperx

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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _extract_audio(str(path), tmp_path)
            audio = whisperx.load_audio(tmp_path)
            result = self.model.transcribe(audio, batch_size=8)

            # The HF session is cached with OfflineAdapter already mounted.
            # Patch the constant AND reset the session cache via the official API.
            import huggingface_hub.constants as _hf_const
            from huggingface_hub import configure_http_backend
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _hf_const.HF_HUB_OFFLINE = False
            configure_http_backend()  # clears LRU session cache, rebuilds without OfflineAdapter
            prev_offline = None
            try:
                print("  Aligning timestamps...")
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=self._device
                )
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, self._device,
                    return_char_alignments=False,
                )

                if self._diarize:
                    if not self._hf_token:
                        raise ValueError(
                            "HF_TOKEN required for diarization.\n"
                            "  1. Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1\n"
                            "  2. Set HF_TOKEN=hf_... in .env or pass --hf-token"
                        )
                    print("  Running speaker diarization...")
                    from whisperx.diarize import DiarizationPipeline
                    diarize_model = DiarizationPipeline(
                        token=self._hf_token,
                        device=self._device,
                    )
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
            finally:
                if prev_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = prev_offline

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start":   round(float(seg["start"]), 2),
                "end":     round(float(seg["end"]), 2),
                "text":    seg["text"].strip(),
                "speaker": seg.get("speaker"),
            })

        language = result.get("language", "unknown")
        print(f"  Done — {len(segments)} segments, language={language}")
        return {
            "language":    language,
            "segments":    segments,
            "has_speakers": self._diarize,
        }

    def full_text(self, transcription: dict) -> str:
        """Returns plain text, or speaker-labeled blocks when diarization was used."""
        if not transcription.get("has_speakers"):
            return " ".join(seg["text"] for seg in transcription["segments"])

        lines: list[str] = []
        cur_speaker: str | None = None
        cur_texts: list[str] = []
        for seg in transcription["segments"]:
            speaker = seg.get("speaker") or "UNKNOWN"
            if speaker != cur_speaker:
                if cur_texts:
                    lines.append(f"{cur_speaker}: {' '.join(cur_texts)}")
                cur_speaker = speaker
                cur_texts = [seg["text"]]
            else:
                cur_texts.append(seg["text"])
        if cur_texts:
            lines.append(f"{cur_speaker}: {' '.join(cur_texts)}")
        return "\n".join(lines)

    def timed_text(self, transcription: dict) -> str:
        lines = []
        for seg in transcription["segments"]:
            start = _fmt_time(seg["start"])
            end = _fmt_time(seg["end"])
            speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
            lines.append(f"[{start} -> {end}] {speaker}{seg['text']}")
        return "\n".join(lines)


def _extract_audio(input_path: str, output_wav: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not process '{Path(input_path).name}'.\n"
            f"ffmpeg error: {result.stderr[-400:]}"
        )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
