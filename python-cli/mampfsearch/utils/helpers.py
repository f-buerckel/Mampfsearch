from hashlib import md5

# Converts the bge embeddings into the correct format for qdrant
# https://qdrant.tech/documentation/concepts/vectors/
def convert_sparse_vector(sparse_vectors):
    from qdrant_client.models import SparseVector
    sparse_indices = []
    sparse_values = []
    for token, value in sparse_vectors.items():
        if float(value) > 0:
            sparse_indices.append(token)
            sparse_values.append(float(value))
    
    return SparseVector(
        indices=sparse_indices,
        values=sparse_values
    )

def name_and_position_to_id(name, position, lecture_position):
    """Convert lecture name to a ID using hash"""
    hash = md5(name.encode("utf-8")).hexdigest()
    # Use the first 8 characters of the hash and store as int
    int_hash = int(hash[:8], 16)
    # convert hash and position to string, concat them and turn back into final id
    id = int(str(int_hash) + str(position) + str(lecture_position))
    return id