import subprocess
import os
import logging
import torch

from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mampfsearch.utils import config

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

    output_srt_file = audio_file.with_suffix('.srt')
    logger.info(f"Transcription will be saved to: {output_srt_file}")

    logger.info(f"Starting transcription for {audio_file}...")
    result = pipe(str(audio_file), return_timestamps=True)
    logger.info("Transcription complete.")

    to_srt(result["chunks"], output_srt_file)
    logger.info(f"Successfully created SRT file at {output_srt_file}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)  # always 0–999
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def to_srt(segments, output_file="output.srt"):
    """Convert segments into an SRT file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start, end = seg["timestamp"]
            text = seg["text"].strip()


            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")