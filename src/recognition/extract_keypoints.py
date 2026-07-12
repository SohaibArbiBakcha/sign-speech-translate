"""
Turn each downloaded WLASL clip into a sequence of pose/hand landmarks using
MediaPipe's HolisticLandmarker task, cropped to the sign's
[frame_start, frame_end] span.

Output: one .npy file per clip in data/keypoints/<gloss>/<video_id>.npy,
shape (num_frames, NUM_LANDMARKS, 3).
"""
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "data" / "wlasl" / "manifest.json"
KEYPOINTS_DIR = ROOT / "data" / "keypoints"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

# 33 pose + 21 left hand + 21 right hand landmarks; face omitted to keep the
# feature vector small enough to train on modest hardware.
NUM_POSE, NUM_HAND = 33, 21

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(MODEL_URL, timeout=120)
        resp.raise_for_status()
        MODEL_PATH.write_bytes(resp.content)
    return MODEL_PATH


def make_landmarker() -> HolisticLandmarker:
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=VisionRunningMode.VIDEO,
    )
    return HolisticLandmarker.create_from_options(options)


def landmarks_to_array(result) -> np.ndarray:
    def coords(landmark_list, count):
        if not landmark_list:
            return np.zeros((count, 3), dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32
        )

    pose = coords(result.pose_landmarks, NUM_POSE)
    left_hand = coords(result.left_hand_landmarks, NUM_HAND)
    right_hand = coords(result.right_hand_landmarks, NUM_HAND)
    return np.concatenate([pose, left_hand, right_hand], axis=0)


def extract_clip(video_path: Path, frame_start: int, frame_end: int, landmarker) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx < frame_start:
            continue
        if frame_end != -1 and idx > frame_end:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frames.append(landmarks_to_array(result))
    cap.release()
    if not frames:
        return np.zeros((0, NUM_POSE + 2 * NUM_HAND, 3), dtype=np.float32)
    return np.stack(frames)


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for entry in tqdm(manifest, desc="Extracting keypoints"):
        gloss, video_id = entry["gloss"], entry["video_id"]
        out_dir = KEYPOINTS_DIR / gloss
        out_path = out_dir / f"{video_id}.npy"
        if out_path.exists():
            continue

        # a fresh landmarker per clip, since VIDEO mode requires strictly
        # increasing timestamps across a landmarker's lifetime and each
        # clip's timestamps restart from 0
        with make_landmarker() as landmarker:
            seq = extract_clip(
                Path(entry["path"]), entry["frame_start"], entry["frame_end"], landmarker
            )
        if seq.shape[0] == 0:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, seq)


if __name__ == "__main__":
    main()
