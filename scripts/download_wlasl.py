"""
Fetch WLASL metadata and download the video clips it references.

WLASL (Word-Level American Sign Language) ships as a JSON index of glosses,
each pointing at a source video (mostly YouTube) plus the start/end frame of
the sign within it. The dataset itself doesn't redistribute video files, so
this script pulls the index and then downloads each clip with yt-dlp.

Usage:
    python scripts/download_wlasl.py --limit 50          # small smoke-test subset
    python scripts/download_wlasl.py --glosses hello,thanks,yes,no
    python scripts/download_wlasl.py                       # full dataset (slow, many GB)
"""
import argparse
import json
import subprocess
from pathlib import Path

import requests

METADATA_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "wlasl"


def fetch_metadata() -> list[dict]:
    meta_path = DATA_DIR / "WLASL_v0.3.json"
    if not meta_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(METADATA_URL, timeout=60)
        resp.raise_for_status()
        meta_path.write_text(resp.text, encoding="utf-8")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def download_instance(gloss: str, inst: dict, out_dir: Path) -> Path | None:
    video_id = inst["video_id"]
    out_path = out_dir / f"{video_id}.mp4"
    if out_path.exists():
        return out_path

    url = inst.get("url", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "mp4",
                "-o", str(out_path),
                url,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return out_path if out_path.exists() else None
    except Exception as e:
        print(f"  [skip] {gloss}/{video_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glosses", type=str, default=None,
                         help="Comma-separated list of glosses to restrict to")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap total number of video clips downloaded")
    args = parser.parse_args()

    metadata = fetch_metadata()
    wanted = set(g.strip().lower() for g in args.glosses.split(",")) if args.glosses else None

    downloaded = 0
    manifest = []
    for entry in metadata:
        gloss = entry["gloss"]
        if wanted and gloss.lower() not in wanted:
            continue
        for inst in entry["instances"]:
            if args.limit and downloaded >= args.limit:
                break
            out_dir = DATA_DIR / "videos" / gloss
            path = download_instance(gloss, inst, out_dir)
            if path:
                manifest.append({
                    "gloss": gloss,
                    "video_id": inst["video_id"],
                    "path": str(path),
                    "frame_start": inst.get("frame_start", 1),
                    "frame_end": inst.get("frame_end", -1),
                    "split": inst.get("split", "train"),
                })
                downloaded += 1
        if args.limit and downloaded >= args.limit:
            break

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {downloaded} clips. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
