"""Run the trained recognizer on a single video file and print the predicted
gloss. Usage: python -m src.recognition.infer path/to/clip.mp4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.recognition.dataset import FEATURE_DIM, _normalize
from src.recognition.extract_keypoints import extract_clip, make_landmarker
from src.recognition.model import SignTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
MAX_FRAMES = 96

_model = None
_classes = None


def load_model():
    """Load (and cache) the trained classifier and its class list."""
    global _model, _classes
    if _model is None:
        _classes = json.loads((CHECKPOINT_DIR / "classes.json").read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = SignTransformer(FEATURE_DIM, num_classes=len(_classes)).to(device)
        _model.load_state_dict(torch.load(CHECKPOINT_DIR / "best.pt", map_location=device))
        _model.eval()
    return _model, _classes


def predict_gloss(video_path: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Run the trained classifier on a clip, return [(gloss, prob), ...]
    sorted by descending probability."""
    model, classes = load_model()
    device = next(model.parameters()).device

    with make_landmarker() as landmarker:
        seq = extract_clip(Path(video_path), frame_start=1, frame_end=-1, landmarker=landmarker)

    seq = _normalize(seq)
    if seq.shape[0] >= MAX_FRAMES:
        seq = seq[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - seq.shape[0], *seq.shape[1:]), dtype=seq.dtype)
        seq = np.concatenate([seq, pad], axis=0)

    x = torch.from_numpy(seq.reshape(1, MAX_FRAMES, -1)).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        top = torch.topk(probs, k=min(top_k, len(classes)))

    return [(classes[idx], prob.item()) for prob, idx in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    args = parser.parse_args()

    print("Top predictions:")
    for gloss, prob in predict_gloss(args.video):
        print(f"  {gloss:<20s} {prob:.3f}")


if __name__ == "__main__":
    main()
