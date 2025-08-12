import click
from mampfsearch.utils import config

@click.command("init")
def init():
    """Initialize the collection for lectures"""
    create_lectures_collection()
    click.echo("Collection initialized")

def create_lectures_collection(name=config.LECTURE_COLLECTION_NAME):
    from qdrant_client import models
    client = config.get_qdrant_client()

    if client.collection_exists(name):
        click.echo(f"Collection {name} already exists")
        return -1
    
    dimension=config.EMBEDDING_DIMENSION
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": 
                models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE
                ),
            "colbert":
                models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams()
            )
        }
    )
    
    click.echo(f"Created collection {name} (vector dimension={dimension})")
    return 0