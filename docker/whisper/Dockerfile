FROM python:3.11

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install openai-whisper

RUN apt-get update && apt-get install git ffmpeg -y

RUN mkdir -p /data

# ENTRYPOINT ["whisper"]
CMD ["tail", "-f", "/dev/null"]
# CMD ["--help"]