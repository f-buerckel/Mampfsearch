import logging
from pathlib import Path

from mampfsearch.core.transcribe import transcribe_lecture

AUDIO_DIR = Path("/home/fbuerckel/Mampfsearch/Resources/real_analysis_lectures/Audio")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not AUDIO_DIR.exists() or not AUDIO_DIR.is_dir():
        raise NotADirectoryError(f"Folder not found: {AUDIO_DIR}")

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        logging.info(f"No .wav files found in: {AUDIO_DIR}")
        return

    logging.info(f"Found {len(wav_files)} .wav file(s) in {AUDIO_DIR}")
    for wav in wav_files:
        logging.info(f"Transcribing {wav.name}...")
        transcribe_lecture(wav)


if __name__ == "__main__":
    main()