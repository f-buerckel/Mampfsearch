import click
from mampfsearch.utils.config import get_qdrant_client

@click.group("collections")
def collection_commands():
    """Manage collections of lectures"""
    pass

@collection_commands.command("delete")
@click.argument("name", required=True)
def delete(name):
    client = get_qdrant_client()
    if not client.collection_exists(name):
        click.echo(f"Collection {name} does not exist")
        return
    
    client.delete_collection(name)
    click.echo(f"Deleted collection {name}")

@collection_commands.command("list")
def list():
    client = get_qdrant_client()
    collections = client.get_collections().collections

    click.echo(f"Found {len(collections)} collections:")

    for collection in collections:
        click.echo(collection.name)
    
    return

@collection_commands.command("get")
@click.argument("name", required=True)
def get(name):
    client = get_qdrant_client()
    if not client.collection_exists(name):
        click.echo(f"Collection {name} does not exist")
        return
    
    collection_info = client.get_collection(name)
    model_info = collection_info.config.params.vectors

    click.echo(f"Status: {collection_info.status}")
    click.echo(f"Indexed vector count: {collection_info.indexed_vectors_count}")
    click.echo(f"Embedding dimension: {model_info.size}")
    click.echo(f"Distance metric: {model_info.distance}")

    return
