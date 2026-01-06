import logging
import torch

from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


logger = logging.getLogger(__name__)


def transcribe_lecture(
    audio_file: Path,
):
    if not audio_file.exists():
        logger.error(f"Audio file not found at: {audio_file}")
        raise FileNotFoundError(f"Audio file not found at: {audio_file}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    output_srt_file = audio_file.with_suffix(".srt")
    output_txt_file = audio_file.with_suffix(".txt")
    logger.info(f"Transcription srt will be saved to: {output_srt_file}")
    logger.info(f"Transcription text will be saved to: {output_txt_file}")

    logger.info(f"Starting transcription for {audio_file}...")
    result = pipe(str(audio_file), return_timestamps=True)
    logger.info("Transcription complete.")

    # save as plain text
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    logger.info(f"Successfully created text file at {output_txt_file}")

    to_srt(result["chunks"], output_srt_file)
    logger.info(f"Successfully created SRT file at {output_srt_file}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)  # always 0–999
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def to_srt(segments, output_file="output.srt"):
    """Convert segments into an SRT file.

    Handles missing end timestamps (end=None) by using the next segment start
    as the end, or falling back to start + fallback_duration_s if no next segment is available.
    """
    fallback_duration_s = 1

    with open(output_file, "w", encoding="utf-8") as f:
        out_idx = 1
        for i, seg in enumerate(segments or []):
            start, end = seg.get("timestamp", (None, None))
            text = (seg.get("text") or "").strip()

            if start is None:
                logger.warning("Skipping segment with missing start timestamp.")
                continue

            if end is None:
                # Prefer next segment start as the end time (keeps subtitles aligned).
                next_start = None
                if i + 1 < len(segments or []):
                    next_start = (segments[i + 1].get("timestamp") or (None, None))[0]

                if next_start is not None and next_start >= start:
                    end = next_start
                else:
                    end = start + fallback_duration_s

                logger.warning(
                    "Segment missing end timestamp; filled end=%s for start=%s",
                    end,
                    start,
                )

            # Ensure monotonic timestamps
            if end < start:
                end = start

            f.write(f"{out_idx}\n")
            f.write(f"{format_timestamp(float(start))} --> {format_timestamp(float(end))}\n")
            f.write(f"{text}\n\n")
            out_idx += 1
