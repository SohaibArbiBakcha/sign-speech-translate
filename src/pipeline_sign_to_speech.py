"""End-to-end: sign video -> gloss (trained classifier) -> spoken audio.

Usage: python -m src.pipeline_sign_to_speech path/to/clip.mp4
"""
import argparse

from src.recognition.infer import predict_gloss
from src.speech.tts import speak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("--no-speak", action="store_true", help="Print only, don't play audio")
    args = parser.parse_args()

    predictions = predict_gloss(args.video, top_k=1)
    gloss, prob = predictions[0]
    print(f"Predicted: {gloss} ({prob:.3f})")

    if not args.no_speak:
        speak(gloss)


if __name__ == "__main__":
    main()
