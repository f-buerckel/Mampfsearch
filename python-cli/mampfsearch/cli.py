import click

from mampfsearch.commands import chunk, collections, lectures_group, transcribe_lecture, benchmark
@click.group()
def main():
    """mampfsearch: Search and manage lecture transcripts"""
    pass

main.add_command(chunk.chunk_srt_command)
main.add_command(collections.collection_commands)
main.add_command(benchmark.benchmark)
main.add_command(lectures_group)
main.add_command(transcribe_lecture)

if __name__ == "__main__":
    main()