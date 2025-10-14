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