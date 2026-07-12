"""Text-to-speech wrapper around pyttsx3 (offline, cross-platform — uses
SAPI5 on Windows). Used for the sign -> speech direction, converting a
recognized gloss/sentence into spoken audio.
"""
import pyttsx3


def speak(text: str, rate: int = 175) -> None:
    """Speak text out loud through the system's default audio output."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()


def save_to_file(text: str, out_path: str, rate: int = 175) -> None:
    """Render text to a wav file instead of speaking it live."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, out_path)
    engine.runAndWait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--out", type=str, default=None, help="Save to a wav file instead of speaking live")
    args = parser.parse_args()
    if args.out:
        save_to_file(args.text, args.out)
    else:
        speak(args.text)
