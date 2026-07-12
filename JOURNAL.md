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

**Open-sourced**: git initialized, MIT-licensed, `.gitignore` excludes
`data/`, `checkpoints/`, `models/*.task`, `.venv/` (all regenerable via the
scripts). README rewritten for a public audience (architecture table,
current results, roadmap, dataset credit). Repo:
https://github.com/SohaibArbiBakcha/sign-speech-translate — pushed to
`main`.

## Same session, continued: generation direction implemented

**Speech ↔ text wrappers**: `src/speech/asr.py` (Whisper, pretrained,
`transcribe()`), `src/speech/tts.py` (pyttsx3, offline/SAPI5 on Windows,
`speak()`/`save_to_file()`). No training, as planned from the start.

**English → ASL gloss**: `src/gloss/translate.py`, rule-based via spaCy
(`en_core_web_sm`) — drops DET/AUX/PART/PUNCT tokens (articles, copula,
infinitive "to"), lemmatizes and upper-cases the rest. Explicitly documented
as a heuristic approximation, not real ASL grammar (word order, classifiers,
non-manual markers are all unhandled) — good enough to drive clip lookup,
not a linguistic claim.
- Dependency snag: `spacy`'s CLI (`spacy download`) needs `click`, which
  didn't get pulled in as a transitive dependency on this Python 3.14 setup.
  Added `click` to requirements.txt explicitly.

**Gloss → sign generation**: `src/generation/dictionary.py` builds a
gloss→clip lookup straight from the already-downloaded WLASL manifest (no
new data collection); `src/generation/generate.py` concatenates the looked-up
clips into one output video.
- **Bug caught in testing**: first version concatenated each clip's *entire*
  video file. Some WLASL manifest entries reference the whole source
  YouTube video (frame_start/frame_end just marks where the sign is within
  it), so a 3-word test sentence produced an 8-minute video. Fixed by
  cropping each clip to its `[frame_start, frame_end]` span during
  concatenation, same as `extract_keypoints.py` already did — 3-word test
  dropped to a correct 7.68s. Lesson: the frame-span cropping needs to
  happen at *every* point a WLASL clip path is consumed, not just in the
  keypoint extractor.

**End-to-end pipelines**: `src/pipeline_speech_to_sign.py` (audio → text →
gloss → video) and `src/pipeline_sign_to_speech.py` (video → gloss →
speech), the latter via a refactored `src/recognition/infer.py` that now
exposes `predict_gloss()` as a reusable function instead of only a CLI.
Tested `pipeline_sign_to_speech` on a real downloaded clip (`all/01912.mp4`)
with `--no-speak`: predicted "now" (0.396 confidence) — wrong for this
clip, but expected given the model's current ~38% val accuracy; not a
pipeline bug.

**Current limitation, by design**: both directions only work for the 27
glosses that have been downloaded and trained on. Missing glosses are
reported and skipped, not silently dropped — surfaced via the `missing`
return value in `lookup_clips()`/`generate_sign_video()`.

## Same session, continued: scaled up training data, added live demo

**Bigger download**: `python scripts/download_wlasl.py --limit 1000` (no
gloss filter) — went from 300 clips/27 glosses to **1000 clips / 101
glosses** (665 train / 174 val / 161 test, 1-17 clips per gloss). Re-ran
keypoint extraction on the full set (all 1000 succeeded).

**Retrained** (40 epochs, same transformer/hyperparameters): best val
accuracy **36.8%** on 101 classes. Looks flat against the earlier 37.7% on
27 classes, but the random baseline dropped from ~3.7% to ~1% — so relative
to chance the model is now doing meaningfully better on a much harder task.
Overfitting is still pronounced (train accuracy 95-97%+ by epoch 30-40),
consistent with still-thin per-class sample counts; more clips per gloss
remains the biggest lever, followed by augmentation or regularization if
more data isn't available.

**Live webcam demo added**: `src/pipeline_webcam_demo.py` — captures the
default camera, runs `HolisticLandmarker` in VIDEO mode continuously (single
long-lived landmarker for the session, timestamps naturally monotonic since
there's no clip boundary to reset at), keeps a rolling 96-frame buffer,
re-runs the classifier every 15 frames, and overlays the current top
prediction on the window (with an optional `--speak` flag to say new
high-confidence predictions aloud via the existing TTS wrapper). Refactored
`src/recognition/infer.py`'s private `_load()` into a public `load_model()`
so the webcam demo could reuse it cleanly.
- Not testable from this environment (no interactive display/webcam access
  in the tool sandbox) — verified only that it imports and parses cleanly.
  Needs a real run on the user's machine to confirm the live loop works as
  expected.
