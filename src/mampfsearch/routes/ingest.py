import logging
from fastapi import APIRouter
from mampfsearch.commands.chunk import chunk_srt
from mampfsearch.commands.lectures.insert_chunks import insert_chunks
from mampfsearch.utils.models import IngestRequest, Chunk

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
        collection_name=request.collection_name,
    ) 

    logger.info(f"Inserted {len(chunks)} chunks for lecture {request.lecture_name} into collection {request.collection_name}")