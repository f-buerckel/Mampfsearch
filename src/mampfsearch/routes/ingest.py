import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from mampfsearch.core.chunking import chunk_srt_file
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
    
    try:
        chunks = chunk_srt_file(
            srt_file=request.srt_file,
            course_id=request.course_id,
            lecture_id=request.lecture_id,
            min_chunk_size=request.min_chunk_size,
            max_chunk_size=request.max_chunk_size,
            overlap=request.overlap,
        )

        logger.info(f"Generated {len(chunks)} chunks for lecture {request.lecture_id}")

        insert_chunks(
            chunks=chunks,
        ) 

        return {"message": f"Successfully ingested {len(chunks)} chunks", "chunks": len(chunks)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/transcribe")
async def transcribe_lecture_endpoint(
    request: TranscriptionRequest,
    background_task: BackgroundTasks
):

    background_task.add_task(transcribe_lecture, audio_file=request.audio_file)
    return {"message": "Transcription started in background"}
