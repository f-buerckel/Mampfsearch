import logging
import uuid

from typing import List
from qdrant_client.models import PointStruct

from mampfsearch.utils import config
from mampfsearch.utils.models import Chunk
from mampfsearch.utils import helpers

logger = logging.getLogger(__name__)

def insert_chunks(
        chunks : List[Chunk],
    ):

    vectors, payloads = create_embeddings_and_payloads(chunks)
    upload(vectors, payloads, config.LECTURE_COLLECTION_NAME)

    return 

def create_embeddings_and_payloads(
        chunks : List[Chunk]
    ):

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

        embedding = model.encode(chunk.text,
                                    return_dense=True,
                                    return_sparse=True,
                                    return_colbert_vecs=True)

        payloads.append(payload)
        vectors.append(embedding)

    return vectors, payloads

def upload(
        vectors : List[dict],
        payloads : List[dict],
        collection_name : str,
    ):

    qdrant_client = config.get_qdrant_client()

    for i, embedding in enumerate(vectors):
        qdrant_client.upsert(
            collection_name=collection_name,
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    payload = payloads[i],
                    vector = {
                        "dense": embedding["dense_vecs"],
                        "colbert": embedding["colbert_vecs"],
                        "sparse": helpers.convert_sparse_vector(embedding["lexical_weights"]),
                    }
                )
            ]
        )

    logger.info(f"Inserted {len(vectors)} vectors into collection {collection_name}")
