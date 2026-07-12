"""End-to-end: spoken audio -> text (Whisper) -> ASL gloss -> sign clip video.

Usage: python -m src.pipeline_speech_to_sign path/to/speech.wav output.mp4
"""
import argparse

from src.generation.generate import generate_sign_video
from src.speech.asr import transcribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--whisper-model", type=str, default="base")
    args = parser.parse_args()

    text = transcribe(args.audio, args.whisper_model)
    print(f"Transcribed: {text}")

    missing = generate_sign_video(text, args.out)
    if missing:
        print(f"No clip available for: {', '.join(missing)} (skipped)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
