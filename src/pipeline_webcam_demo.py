"""Live webcam demo: captures video from the default camera, runs the
trained sign recognizer on a rolling window of frames, and overlays the
current top prediction. Press 'q' to quit.

Usage: python -m src.pipeline_webcam_demo
"""
import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.recognition.dataset import FEATURE_DIM, _normalize
from src.recognition.extract_keypoints import (
    landmarks_to_array,
    make_landmarker,
)
from src.recognition.infer import load_model
from src.speech.tts import speak

MAX_FRAMES = 96
PREDICT_EVERY_N_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--speak", action="store_true", help="Speak new high-confidence predictions aloud")
    args = parser.parse_args()

    model, classes = load_model()
    device = next(model.parameters()).device

    cap = cv2.VideoCapture(args.camera)
    landmarker = make_landmarker()

    frame_buffer = deque(maxlen=MAX_FRAMES)
    frame_count = 0
    start_time = time.time()
    last_prediction = ""
    last_spoken = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_buffer.append(landmarks_to_array(result))

            if frame_count % PREDICT_EVERY_N_FRAMES == 0 and len(frame_buffer) >= 10:
                seq = np.stack(frame_buffer)
                seq = _normalize(seq)
                if seq.shape[0] >= MAX_FRAMES:
                    seq = seq[-MAX_FRAMES:]
                else:
                    pad = np.zeros((MAX_FRAMES - seq.shape[0], *seq.shape[1:]), dtype=seq.dtype)
                    seq = np.concatenate([pad, seq], axis=0)

                x = torch.from_numpy(seq.reshape(1, MAX_FRAMES, -1)).float().to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=-1)[0]
                    top_prob, top_idx = probs.max(dim=-1)

                gloss, prob = classes[top_idx.item()], top_prob.item()
                last_prediction = f"{gloss} ({prob:.2f})"

                if args.speak and prob >= CONFIDENCE_THRESHOLD and gloss != last_spoken:
                    speak(gloss)
                    last_spoken = gloss

            cv2.putText(
                frame, last_prediction, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )
            cv2.imshow("Sign recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
