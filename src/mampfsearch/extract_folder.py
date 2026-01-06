import logging
from pathlib import Path

from mampfsearch.core.extraction_pipeline.pipeline import extract

# Folder containing .srt files
SRT_DIR = Path("/home/fbuerckel/Mampfsearch/Resources/real_analysis_lectures/Audio")

# Will be used/created in the graph storage
COURSE_NAME = "Real Analysis"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not SRT_DIR.exists() or not SRT_DIR.is_dir():
        raise NotADirectoryError(f"Folder not found: {SRT_DIR}")

    srt_files = sorted(SRT_DIR.glob("*.srt"))
    if not srt_files:
        logging.info(f"No .srt files found in: {SRT_DIR}")
        return

    logging.info(f"Found {len(srt_files)} .srt file(s) in {SRT_DIR}")
    for i, srt in enumerate(srt_files, start=1):
        lecture_name = srt.stem
        lecture_position = i

        logging.info(
            f"Extracting from {srt.name} (lecture='{lecture_name}', pos={lecture_position})..."
        )
        extract(
            file_path=srt,
            course_name=COURSE_NAME,
            lecture_name=lecture_name,
            lecture_position=lecture_position,
            lecture_description=None,
        )


if __name__ == "__main__":
    main()