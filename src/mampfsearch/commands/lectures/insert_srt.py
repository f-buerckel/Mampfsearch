from mampfsearch.utils import config
from mampfsearch.utils import helpers
import pysrt
from tqdm import tqdm
from pathlib import Path

def insert_srt(lecture_name, lecture_position, srt_file, collection_name=config.LECTURE_COLLECTION_NAME):

    subs = pysrt.open(srt_file)

    vectors, payloads = create_embeddings_and_payloads(subs, lecture_name, lecture_position)
    upload(lecture_name, lecture_position, vectors, payloads, collection_name)

    return 

def create_embeddings_and_payloads(subs, lecture_name, lecture_position):
    model = config.get_bge_embedding_model()

    payloads = []
    vectors = []

    for sub in tqdm(subs):
        payload = {
            "text": str(sub.text),
            "start_time": str(sub.start),
            "end_time": str(sub.end),
            "lecture_name": lecture_name,
            "lecture_position" : lecture_position,
            "position": int(sub.index)
        }

        embedding = model.encode(str(sub.text),
                                    return_dense=True,
                                    return_sparse=True,
                                    return_colbert_vecs=True)

        payloads.append(payload)
        vectors.append(embedding)

    return vectors, payloads
    
def upload(lecture_name, lecture_position, vectors, payloads, collection_name):
    from qdrant_client.models import PointStruct
    client = config.get_qdrant_client()

    for i, embedding in enumerate(tqdm(vectors)):
        client.upsert(
            collection_name=collection_name,
            points = [
                PointStruct(
                    id=helpers.name_and_position_to_id(lecture_name, payloads[i]["position"], lecture_position),
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
