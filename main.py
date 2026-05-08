"""
Meeting Minutes Generator
Usage:  python main.py <audio_or_video_file> [options]
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from transcriber import Transcriber
from minutes_generator import MinutesGenerator
from exporter import save_text, save_pdf, to_text


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate meeting minutes from audio/video files."
    )
    parser.add_argument("file", help="Path to MP4, MP3, WAV, or M4A file")
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "mistral:latest"),
        help="Ollama model to use (default: mistral:latest)",
    )
    parser.add_argument(
        "--whisper",
        default=os.getenv("WHISPER_MODEL", "base"),
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server URL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path without extension (default: same name as input)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "pdf", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Also save the raw transcript",
    )
    parser.add_argument(
        "--show-transcript",
        action="store_true",
        help="Print transcript to console and exit (skips minutes generation)",
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Disable speaker diarization",
    )
    parser.set_defaults(diarize=True)
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token for pyannote diarization models",
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_base = Path(args.output) if args.output else input_path.with_suffix("")

    print(f"\n{'='*56}")
    print(f"  Meeting Minutes Generator")
    print(f"{'='*56}")
    print(f"  Input  : {input_path.name}")
    print(f"  Whisper: {args.whisper}")
    print(f"  Model  : {args.model}")
    print(f"  Diarize: {'yes' if args.diarize else 'no'}")
    print(f"{'='*56}\n")

    # Step 1 — Transcribe
    print("[1/3] Transcribing audio...")
    transcriber = Transcriber(
        model_size=args.whisper,
        diarize=args.diarize,
        hf_token=args.hf_token,
    )
    transcription = transcriber.transcribe(str(input_path))
    transcript_text = transcriber.full_text(transcription)

    if args.show_transcript:
        print("\n" + "─" * 56)
        print(transcriber.timed_text(transcription))
        print("─" * 56 + "\n")
        return

    if args.transcript:
        transcript_path = str(output_base) + "_transcript.txt"
        Path(transcript_path).write_text(
            transcriber.timed_text(transcription), encoding="utf-8"
        )
        print(f"  Transcript saved: {transcript_path}")

    if not transcript_text.strip():
        print("Error: Transcription returned empty text. Check the audio file.")
        sys.exit(1)

    # Step 2 — Generate minutes
    print("\n[2/3] Generating meeting minutes...")
    generator = MinutesGenerator(ollama_url=args.ollama_url, model=args.model)
    minutes = generator.generate(transcript_text)

    # Step 3 — Export
    print("\n[3/3] Exporting...")
    if args.format in ("text", "both"):
        path = save_text(minutes, str(input_path), str(output_base) + "_minutes.txt")
        print(f"  Text : {path}")

    if args.format in ("pdf", "both"):
        path = save_pdf(minutes, str(input_path), str(output_base) + "_minutes.pdf")
        print(f"  PDF  : {path}")

    # Print summary to console
    print(f"\n{'='*56}")
    print(to_text(minutes, str(input_path)))

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
