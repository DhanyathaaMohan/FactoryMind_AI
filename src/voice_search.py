"""
voice_search.py
---------------
CPU-compatible audio transcription using Faster-Whisper.

Supports .wav and .mp3 input files.
The Whisper model is cached at module level so it is only loaded once
per process.

Usage
-----
    from voice_search import transcribe
    text = transcribe("recording.wav")
"""

from pathlib import Path
from typing import Optional

# faster-whisper import — kept inside functions so import errors give a
# clear message rather than crashing the whole backend at startup.
_whisper_model = None
_whisper_model_size = None


# --------------------------------------------------
# SUPPORTED FORMATS
# --------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

# Default model size.  "base" is a good CPU balance of speed vs accuracy.
# Acceptable values: tiny, base, small, medium, large-v2
DEFAULT_MODEL_SIZE = "base"


# --------------------------------------------------
# MODEL CACHE
# --------------------------------------------------

def _get_model(model_size: str = DEFAULT_MODEL_SIZE):
    """
    Lazily load and cache the Faster-Whisper model.
    Re-uses the same model instance on repeated calls.
    """
    global _whisper_model, _whisper_model_size

    if _whisper_model is None or _whisper_model_size != model_size:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            ) from exc

        print(f"Loading Whisper model '{model_size}' on CPU …")
        _whisper_model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",   # fastest CPU quantisation
        )
        _whisper_model_size = model_size
        print("Whisper model ready.")

    return _whisper_model


# --------------------------------------------------
# TRANSCRIPTION
# --------------------------------------------------

def transcribe(
    audio_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe an audio file to text.

    Parameters
    ----------
    audio_path  : str | Path
        Path to a .wav, .mp3 or other supported audio file.
    model_size  : str
        Faster-Whisper model size (default "base").
    language    : str | None
        ISO-639-1 language code (e.g. "en").  None = auto-detect.

    Returns
    -------
    str
        Full transcription of the audio file.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the file extension is not supported.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{audio_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    model = _get_model(model_size)

    transcribe_kwargs = {}
    if language:
        transcribe_kwargs["language"] = language

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        **transcribe_kwargs,
    )

    transcript = " ".join(segment.text.strip() for segment in segments)

    print(
        f"Transcribed [{audio_path.name}]  "
        f"lang={info.language}  "
        f"duration={info.duration:.1f}s"
    )

    return transcript.strip()


# --------------------------------------------------
# MAIN — demonstration
# --------------------------------------------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python voice_search.py <audio_file>")
        print("Supported formats:", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return

    audio_file = sys.argv[1]
    print(f"\nTranscribing: {audio_file}")

    text = transcribe(audio_file)

    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
