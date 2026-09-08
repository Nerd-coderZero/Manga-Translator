FROM python:3.10-slim

# opencv needs libgl1 and libglib2.0-0 even in the headless build.
# the placement stage renders text with PIL and looks for a bold TrueType
# face on disk (pipeline_core.find_available_font); without these font
# packages it falls back to a bitmap font that cannot be scaled, so every
# translated line would render at one fixed size.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fonts-dejavu-core \
        fonts-liberation \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000. model weights are downloaded at
# first use into the home directory (~/.paddleocr for PaddleOCR,
# ~/.cache/huggingface for manga-ocr), so HOME has to be writable by that
# user or the first request fails with a permission error rather than a
# model error, which is a confusing thing to debug.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=user requirements.txt ./
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user backend/ ./backend/
COPY --chown=user frontend/ ./frontend/

ENV FRONTEND_DIR=/app/frontend \
    MANGA_TRANSLATOR_DATA_DIR=/home/user/data \
    PORT=7860

RUN mkdir -p /home/user/data

EXPOSE 7860

# single worker. the OCR model instances in ocr.py are process-global and
# not safe to share across threads, and the batch queue in batch_store.py
# is in-process state that a second worker would not see.
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
