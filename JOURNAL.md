# Project Journal

## 2026-07-12

**Project defined**: real-time two-way translator between spoken/written
language and sign language.
- Direction A: speech → text → sign gesture generation.
- Direction B: sign language video → text/speech.
- Goal: accessibility for deaf/hard-of-hearing communication.
- Target sign language: **ASL** (chosen over LSF/regional languages for
  public dataset availability — WLASL, How2Sign, MS-ASL all exist for ASL).

**Architecture decision**:
- Speech ↔ text: no training needed, use pretrained Whisper (ASR) + a
  standard TTS engine.
- Text → sign (generation): no training either. Translate text to ASL gloss,
  then look up/concatenate pre-recorded sign clips or skeletal keyframes per
  gloss token. Avoids needing a paired text/motion dataset, which isn't
  realistically available at this project's scale.
- Sign → text (recognition): this is the one component actually trained.
  Pipeline: MediaPipe Holistic extracts pose + hand landmarks per frame →
  transformer encoder classifies the landmark sequence into a gloss (isolated
  sign, word-level) → gloss becomes text/speech output.

**Dataset**: WLASL (Word-Level ASL), ~2000-word vocabulary, videos mostly
hosted on YouTube and referenced by URL in a JSON index (not bundled
directly). Chose isolated word-level recognition as the starting point
before attempting continuous/sentence-level signing (would need How2Sign).

**Environment**: only Python 3.14 available on this machine. Verified
`mediapipe==0.10.35` ships a `py3-none` wheel, so it installs fine despite
being a very recent Python release — flagged as a risk to revisit if a
future mediapipe release drops support or behaves differently.

**Scaffolded** (empty repo → initial pipeline):
- `scripts/download_wlasl.py` — fetches WLASL metadata JSON, downloads
  referenced clips via yt-dlp, writes a manifest (gloss, path, frame
  range, split).
- `src/recognition/extract_keypoints.py` — MediaPipe Holistic over each
  clip's [frame_start, frame_end] span → `.npy` landmark sequences.
- `src/recognition/dataset.py` — PyTorch Dataset; normalizes each sequence
  by centering/scaling on shoulder landmarks (signer-invariant), pads/
  truncates to a fixed frame count.
- `src/recognition/model.py` — transformer encoder (CLS-token pooling)
  over per-frame keypoint features, classifies into gloss vocabulary.
- `src/recognition/train.py`, `src/recognition/infer.py` — training loop
  with checkpointing on best val accuracy; single-clip inference with
  top-5 predictions.
- `README.md` — setup + pipeline usage instructions.

**Not yet started**: `src/speech/` (ASR/TTS wrappers), `src/gloss/`
(English → ASL gloss translation), `src/generation/` (gloss → clip/avatar
lookup). None of these have code yet, only placeholder directories.

**Smoke test run** (small scale, `--glosses hello,thanks,yes,no --limit 20`):
downloaded 20 clips successfully (12 "no" + 8 "yes" — WLASL's metadata isn't
alphabetically ordered, so "hello"/"thanks" weren't reached before the limit
was hit). Confirmed the venv (`.venv/Scripts/python.exe` on Windows,
PowerShell needs backslash paths not `/C:/...`) and downloader work.

**mediapipe API break found and fixed**: `mediapipe==0.10.35`'s wheel no
longer ships `mp.solutions` (the old Holistic API) — only `mp.tasks`, the
newer Tasks API. Rewrote `extract_keypoints.py`/`infer.py` to use
`mp.tasks.vision.HolisticLandmarker`, downloading the model bundle to
`models/holistic_landmarker.task` on first run. Also fixed: (1) landmark
results are flat lists (`result.pose_landmarks` is directly 33 landmarks,
not nested per-person), and (2) VIDEO running mode requires strictly
increasing timestamps across a landmarker's *entire lifetime*, so each clip
now gets its own landmarker instance instead of reusing one across clips
(each clip's timestamp otherwise restarts near 0, which errors on the next
clip).

**Scaled up**: downloaded 300 clips across 27 glosses (195 train / 53 val /
52 test split, per WLASL's own split assignment) via
`python scripts/download_wlasl.py --limit 300` (no gloss filter). Keypoint
extraction succeeded on all 300 clips.

**First real training run** (30 epochs, transformer, `src/recognition/train.py`
defaults): best val accuracy **37.7%** on 27 classes (random baseline ~3.7%),
train accuracy reached 83.6% by epoch 30 — visible overfitting given only
~7 training clips per class on average. Model checkpoint at
`checkpoints/best.pt`, class list at `checkpoints/classes.json`.

**Next steps to consider**: download more clips per gloss (WLASL has up to
~20 instances/gloss for the more common signs) to reduce overfitting;
try data augmentation on keypoint sequences (time-warping, jitter); consider
reducing model capacity (fewer transformer layers) given the small dataset;
eventually test `src/recognition/infer.py` on a fresh clip not in the
manifest.
