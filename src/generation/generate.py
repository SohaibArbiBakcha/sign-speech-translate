"""Text -> ASL gloss -> concatenated sign clip video.

Usage: python -m src.generation.generate "Is your mother fine now?" output.mp4
"""
import argparse
from pathlib import Path

import cv2

from src.generation.dictionary import build_gloss_dictionary, lookup_clips
from src.gloss.translate import text_to_gloss


def concatenate_clips(clips: list[dict], out_path: str, fps: int = 25, size: tuple[int, int] = (480, 360)) -> None:
    """Write each clip's [frame_start, frame_end] span (not the whole file —
    WLASL entries often point at a full source video with only that span
    covering the actual sign) in sequence to one output video."""
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for clip in clips:
        cap = cv2.VideoCapture(clip["path"])
        frame_start, frame_end = clip["frame_start"], clip["frame_end"]
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx < frame_start:
                continue
            if frame_end != -1 and idx > frame_end:
                break
            frame = cv2.resize(frame, size)
            writer.write(frame)
        cap.release()
    writer.release()


def generate_sign_video(text: str, out_path: str) -> list[str]:
    """Full text -> sign pipeline. Returns the list of glosses that had no
    matching clip (so the caller can report them), writes the video for the
    rest to out_path."""
    glosses = text_to_gloss(text)
    dictionary = build_gloss_dictionary()
    clips, missing = lookup_clips(glosses, dictionary)

    if clips:
        concatenate_clips(clips, out_path)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("out", type=str)
    args = parser.parse_args()

    missing = generate_sign_video(args.text, args.out)
    if missing:
        print(f"No clip available for: {', '.join(missing)} (skipped)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
