"""Loads extracted keypoint sequences (see extract_keypoints.py) into a
PyTorch dataset for isolated sign recognition."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "data" / "wlasl" / "manifest.json"
KEYPOINTS_DIR = ROOT / "data" / "keypoints"

NUM_LANDMARKS = 33 + 21 + 21
FEATURE_DIM = NUM_LANDMARKS * 3


def _normalize(seq: np.ndarray) -> np.ndarray:
    """Center each frame on the shoulder midpoint and scale by shoulder width
    so the model sees signer-invariant motion rather than absolute position."""
    left_shoulder, right_shoulder = 11, 12
    center = (seq[:, left_shoulder] + seq[:, right_shoulder]) / 2
    scale = np.linalg.norm(seq[:, left_shoulder] - seq[:, right_shoulder], axis=-1)
    scale = np.clip(scale, 1e-6, None)
    return (seq - center[:, None, :]) / scale[:, None, None]


class WLASLDataset(Dataset):
    def __init__(self, split: str, max_frames: int = 96):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.max_frames = max_frames

        self.samples = []
        glosses = set()
        for entry in manifest:
            if entry["split"] != split:
                continue
            kp_path = KEYPOINTS_DIR / entry["gloss"] / f"{entry['video_id']}.npy"
            if kp_path.exists():
                self.samples.append((kp_path, entry["gloss"]))
                glosses.add(entry["gloss"])

        self.classes = sorted(glosses)
        self.gloss_to_idx = {g: i for i, g in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        kp_path, gloss = self.samples[i]
        seq = np.load(kp_path)
        seq = _normalize(seq)

        # pad/truncate to a fixed frame count so batches can be stacked
        if seq.shape[0] >= self.max_frames:
            seq = seq[: self.max_frames]
        else:
            pad = np.zeros((self.max_frames - seq.shape[0], *seq.shape[1:]), dtype=seq.dtype)
            seq = np.concatenate([seq, pad], axis=0)

        x = torch.from_numpy(seq.reshape(self.max_frames, -1)).float()
        y = torch.tensor(self.gloss_to_idx[gloss], dtype=torch.long)
        return x, y
