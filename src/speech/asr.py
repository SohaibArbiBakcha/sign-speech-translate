"""Speech-to-text wrapper around openai-whisper. No training here — the
recognition direction (video -> gloss) is the only component we train;
speech transcription just uses a pretrained model.
"""
import whisper

_model = None


def _get_model(name: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(name)
    return _model


def transcribe(audio_path: str, model_name: str = "base") -> str:
    """Transcribe an audio file (wav/mp3/etc.) to English text."""
    model = _get_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"].strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    parser.add_argument("--model", type=str, default="base")
    args = parser.parse_args()
    print(transcribe(args.audio, args.model))
