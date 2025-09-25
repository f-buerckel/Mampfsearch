import logging
from fastapi import APIRouter, BackgroundTasks
from mampfsearch.core.chunk import chunk_srt
from mampfsearch.core.lectures.insert_chunks import insert_chunks
from mampfsearch.core.transcribe import transcribe_lecture
from mampfsearch.utils.models import IngestRequest, TranscriptionRequest
from mampfsearch.utils import config

router = APIRouter(
    tags=["Ingest"],
)

logger = logging.getLogger(__name__)

@router.post("/ingest")
async def ingest_transcript(
    request: IngestRequest,
):

    chunks = chunk_srt(
        srt_file=request.srt_file,
        lecture_name=request.lecture_name,
        lecture_position=request.lecture_position,
        min_chunk_size=request.min_chunk_size,
        max_chunk_size=request.max_chunk_size,
        overlap=request.overlap,
    )

    logger.info(f"Generated {len(chunks)} chunks for lecture {request.lecture_name}")

    insert_chunks(
        chunks=chunks,
    ) 

@router.post("/transcribe")
async def transcribe_lecture_endpoint(
    request: TranscriptionRequest,
    background_task: BackgroundTasks
):

    background_task.add_task(transcribe_lecture, audio_file=request.audio_file)
    return {"message": "Transcription started in background"}
