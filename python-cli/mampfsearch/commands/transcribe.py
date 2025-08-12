from mampfsearch.utils import config
import click
from pathlib import Path
import subprocess
import os

@click.command("transcribe")
@click.argument(
    "audio_filename",
    type=str,
    required=True
)
@click.option(
    "--model",
    default="medium",
    show_default=True,
    help="Whisper model size (e.g., tiny, base, small, medium, large)."
)
def transcribe_lecture(audio_filename: str, model: str):
    """
    Transcribes an audio file to SRT using whisper.

    AUDIO_FILENAME is the name of the audio file (e.g., 'lecture1.wav')
    located in the '<Lecture_Path>/audio/' directory.
    Transcripts will be saved to '<Lecture_Path>/transcripts/'.
    """
    host_data_dir = Path(config.get_lectures_path()).resolve()

    host_audio_dir = host_data_dir / "audio"
    host_transcripts_dir = host_data_dir / "transcripts"

    host_audio_file_path = host_audio_dir / audio_filename

    if not host_audio_file_path.exists():
        click.echo(f"Error: Audio file not found at '{host_audio_file_path}'.", err=True)
        click.echo(f"Please ensure '{audio_filename}' exists in '{host_audio_dir}'.")
        raise click.Abort()

    # Ensure the host transcripts directory exists
    os.makedirs(host_transcripts_dir, exist_ok=True)

    # Determine paths within the container
    container_audio_file_path = Path("/data/audio") / audio_filename
    container_transcripts_dir = Path("/data/transcripts")

    srt_filename_stem = Path(audio_filename).stem
    expected_srt_host_path = host_transcripts_dir / (srt_filename_stem + ".srt")

    click.echo(f"Host Docker volume base: '{host_data_dir}'")
    click.echo(f"Host audio file: '{host_audio_file_path}'")
    click.echo(f"Host transcripts output directory: '{host_transcripts_dir}'")
    click.echo(f"Container audio file: '{container_audio_file_path}'")
    click.echo(f"Container transcripts output directory: '{container_transcripts_dir}'")
    click.echo(f"Using Whisper model: '{model}'")

    docker_command = [
        "docker", "exec",
        "-t",
        "whisper-container",
        "whisper",
        str(container_audio_file_path),
        "--model", model,
        "--output_format", "srt",
        "--output_dir", str(container_transcripts_dir),
    ]

    click.echo(f"\nExecuting Docker command: {' '.join(docker_command)}\n")

    try:
        process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                click.echo(line, nl=False)
        process.wait()

        if process.returncode == 0:
            if expected_srt_host_path.exists():
                click.echo(f"\nTranscription successful! SRT file saved to: {expected_srt_host_path}")
            else:
                click.echo(f"\nTranscription command completed, but expected SRT file not found at '{expected_srt_host_path}'.", err=True)
        else:
            click.echo(f"\nError during transcription. Docker command exited with code {process.returncode}.", err=True)

    except FileNotFoundError:
        click.echo("Error: 'docker' command not found. Is Docker installed and in your PATH?", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        raise click.Abort()
