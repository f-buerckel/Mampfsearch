import logging
import uuid

from typing import List

from mampfsearch.utils import config
from mampfsearch.utils.models import Segment, Passage

logger = logging.getLogger(__name__)


def insert_chunks(
    chunks: List[Segment],
):
    vectors, payloads = create_embeddings_and_payloads(chunks)
    upload(vectors, payloads, config.LECTURE_COLLECTION_NAME)

    return


def create_embeddings_and_payloads(chunks: List[Segment]):
    payloads = []
    vectors = []

    model = config.get_embedding_model()

    for chunk in chunks:
        payload = {
            "text": chunk.text,
            "course_id": chunk.location.courseId,
            "lecture_id": chunk.location.lectureId,
            "start_time": str(chunk.location.start_time),
            "end_time": str(chunk.location.end_time),
        }

        embedding = model.encode(
            chunk.text, return_dense=True, return_sparse=True, return_colbert_vecs=True
        )

        payloads.append(payload)
        vectors.append(embedding)

    return vectors, payloads


def upload(
    vectors: List[dict],
    payloads: List[dict],
    collection_name: str,
):
    client = config.get_vector_storage()
    client.upload_chunks(collection_name, vectors, payloads)
    logger.info(f"Inserted {len(vectors)} vectors into collection {collection_name}")
