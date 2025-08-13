from mampfsearch.utils import config
from mampfsearch.utils.models import Chunk
from mampfsearch.utils import helpers
from typing import List
from qdrant_client.models import PointStruct

def insert_chunks(
        chunks : List[Chunk],
        collection_name=config.LECTURE_COLLECTION_NAME
        ):

    vectors, payloads = create_embeddings_and_payloads(chunks)
    upload(vectors, payloads, collection_name)

    return 

def create_embeddings_and_payloads(chunks):
    model = config.get_bge_embedding_model()

    payloads = []
    vectors = []

    for chunk in chunks:
        payload = {
            "text": chunk.text,
            "start_time": str(chunk.start_time),
            "end_time": str(chunk.end_time),
            "lecture_name": chunk.lecture_name,
            "lecture_position" : chunk.lecture_position,
            "position": chunk.position
        }

        embedding = model.encode(chunk.text,
                                    return_dense=True,
                                    return_sparse=True,
                                    return_colbert_vecs=True)

        payloads.append(payload)
        vectors.append(embedding)

    return vectors, payloads
    
def upload(vectors, payloads, collection_name):
    client = config.get_qdrant_client()

    for i, embedding in enumerate(vectors):
        client.upsert(
            collection_name=collection_name,
            points = [
                PointStruct(
                    id=helpers.name_and_position_to_id(payloads[i]["lecture_name"], payloads[i]["position"], payloads[i]["lecture_position"]),
                    payload = payloads[i],
                    vector = {
                        "dense": embedding["dense_vecs"],
                        "colbert": embedding["colbert_vecs"],
                        "sparse": helpers.convert_sparse_vector(embedding["lexical_weights"]),
                    }
                )
            ]
        )

    print(f"Inserted {len(vectors)} vectors into collection {collection_name}")
