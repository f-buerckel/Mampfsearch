import click
from .init import init
from .insert_srt import insert_srt_command
from .search import search_lectures_command
from .ask import ask


@click.group("lectures")
def lectures_group():
    """Manage and search lecture materials."""
    pass

lectures_group.add_command(init)
lectures_group.add_command(insert_srt_command)
lectures_group.add_command(search_lectures_command)
lectures_group.add_command(ask)