"""Gloss -> sign clip lookup, built from whatever WLASL clips have already
been downloaded (see scripts/download_wlasl.py). This is deliberately a
plain dictionary rather than a trained model: text -> sign generation would
need a much larger paired text/motion dataset than is realistically
available here, so we generate by concatenating real recorded clips
instead.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "data" / "wlasl" / "manifest.json"


def build_gloss_dictionary() -> dict[str, list[dict]]:
    """Map each gloss (upper-cased, matching gloss token convention) to the
    list of downloaded clips available for it, each with the [frame_start,
    frame_end] span that actually bounds the sign within the source video
    (WLASL entries often reference the whole source video, not just the
    sign, so callers must crop to this span rather than play the clip whole)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dictionary: dict[str, list[dict]] = {}
    for entry in manifest:
        gloss = entry["gloss"].upper()
        dictionary.setdefault(gloss, []).append({
            "path": entry["path"],
            "frame_start": entry.get("frame_start", 1),
            "frame_end": entry.get("frame_end", -1),
        })
    return dictionary


def lookup_clips(glosses: list[str], dictionary: dict[str, list[dict]] | None = None) -> tuple[list[dict], list[str]]:
    """Resolve a gloss sequence to one clip (path + frame span) per gloss.

    Returns (clips, missing) — `missing` lists glosses with no available
    clip, which the caller should surface (e.g. spell/finger-spell fallback)
    rather than silently drop.
    """
    dictionary = dictionary or build_gloss_dictionary()
    clips, missing = [], []
    for gloss in glosses:
        candidates = dictionary.get(gloss)
        if candidates:
            clips.append(candidates[0])
        else:
            missing.append(gloss)
    return clips, missing
