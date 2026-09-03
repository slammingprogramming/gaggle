FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ffmpeg provides both the `ffmpeg` and `ffprobe` binaries this project
# shells out to for real media metadata extraction, motion/audio analysis,
# and derived-clip extraction. tesseract-ocr is the OCR engine used for
# license-plate text recognition (enrichment/plate.py) -- both are real,
# fully offline, no model-download-required capabilities. Nothing here
# calls a cloud API by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY docs ./docs
COPY examples ./examples

RUN pip install --no-cache-dir -e .[dev,vision,transcription,cloud]

# The workspace is a bind mount target (see docker-compose.yml); it is not
# baked into the image so evidence never lives inside a container layer.
VOLUME ["/workspace"]

ENTRYPOINT ["gaggle"]
CMD ["--help"]
