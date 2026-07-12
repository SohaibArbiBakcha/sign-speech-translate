# Speech ↔ Sign Language Translator

An open-source, real-time system for two-way translation between spoken/
written language and **American Sign Language (ASL)**, built to make
communication between deaf/hard-of-hearing and hearing people more seamless.

```
  speech/text  ──ASR/gloss──▶  sign clip / avatar generation
  webcam video ──pose+ML──▶  gloss ──▶  text / speech
```

## Why

Most "accessibility" translation tools only work one direction, or treat
sign language as a 1:1 mapping of spoken words to hand shapes — which isn't
how sign languages actually work (ASL has its own grammar, distinct from
English). This project aims for both directions, starting from a solid,
inspectable ML pipeline rather than a demo mapping.

## How it works

| Direction | Approach | Status |
|---|---|---|
| Sign → text/speech | Webcam video → [MediaPipe](https://github.com/google/mediapipe) Holistic landmarks (pose + both hands) → transformer classifier → gloss → text/TTS | **In progress** — training pipeline works end-to-end, see [Results](#current-results) |
| Speech/text → sign | ASR (Whisper) → text → ASL gloss → lookup in a gloss→clip/avatar dictionary | Planned, not yet implemented |

The recognition direction is the one that requires training a model, so it
came first. Generation is designed as a dictionary lookup rather than a
trained generative model — training text→motion generation would need a much
larger paired dataset than is realistically available for this project.

## Current results

Trained on a 300-clip / 27-gloss subset of [WLASL](https://dxli94.github.io/WLASL/)
(Word-Level ASL): **37.7% validation accuracy** (random baseline for 27
classes is ~3.7%). Training accuracy reaches 83%+ by epoch 30, so the model
is clearly learning — the gap is overfitting from small per-class sample
counts (~7 clips/class on average). More data per gloss is the main lever
to close that gap; see [JOURNAL.md](JOURNAL.md) for the full run-by-run log.

## Project layout

```
scripts/download_wlasl.py       fetch WLASL metadata + video clips
src/recognition/
  extract_keypoints.py          video -> MediaPipe landmark sequences
  dataset.py                    PyTorch Dataset over extracted keypoints
  model.py                      transformer encoder classifier
  train.py / infer.py           training loop / single-clip inference
src/speech/                     ASR (Whisper) + TTS wrappers (planned)
src/gloss/                      English -> ASL gloss translation (planned)
src/generation/                 gloss -> sign clip/avatar lookup (planned)
data/                           WLASL metadata, videos, extracted keypoints (gitignored)
checkpoints/                    trained model weights (gitignored)
JOURNAL.md                      dated log of decisions, experiments, and results
```

## Getting started

Requires Python 3.10+ (developed against 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Train the sign recognizer

```bash
# 1. Download a subset of WLASL (start small to validate the pipeline)
python scripts/download_wlasl.py --limit 300

# 2. Extract MediaPipe pose/hand landmarks from each clip
python -m src.recognition.extract_keypoints

# 3. Train the transformer classifier
python -m src.recognition.train --epochs 30

# 4. Run inference on a new clip
python -m src.recognition.infer path/to/clip.mp4
```

Notes:
- WLASL videos are mostly hosted on YouTube (via `yt-dlp`); some links have
  rotted or point at non-YouTube hosts `yt-dlp` can't fetch, so expect the
  downloader to skip a fraction of clips — this is normal.
- MediaPipe's `HolisticLandmarker` model bundle (~a few MB) is downloaded
  automatically to `models/` on first run.

## Roadmap

- [x] WLASL download pipeline
- [x] MediaPipe keypoint extraction
- [x] Transformer-based isolated sign recognition, trained end-to-end
- [ ] Scale up training data (more clips/gloss) to close the train/val gap
- [ ] Continuous (sentence-level) sign recognition (How2Sign)
- [ ] Speech-to-text (Whisper) + text-to-speech wrappers
- [ ] English → ASL gloss translation
- [ ] Gloss → sign clip/avatar generation
- [ ] Real-time webcam demo tying both directions together

## Contributing

Issues and PRs welcome — this is an early-stage, actively-changing project.
If you're picking up a roadmap item, open an issue first so effort doesn't
overlap.

## Dataset & acknowledgments

- [WLASL](https://dxli94.github.io/WLASL/) (Word-Level American Sign
  Language) — Li, Dongxu, et al. "Word-level Deep Sign Language Recognition
  from Video: A New Large-scale Dataset and Methods Comparison." WACV 2020.
- [MediaPipe](https://github.com/google/mediapipe) for pose/hand landmark
  extraction.

## License

MIT — see [LICENSE](LICENSE).
