---
title: Manga Translator
emoji: 📖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
python_version: "3.10"
startup_duration_timeout: 1h
pinned: false
short_description: Japanese and Chinese manga translation with OCR and an LLM
---

# Manga Translator

Translates Japanese and Chinese manga pages into English: detect the text
regions, read them, translate each line, then erase the original text and
paint the translation back into the same region.

## Pipeline

```
image
  -> PaddleOCR DB detector          text region boxes
  -> redundant-box filter           drop duplicate oversized detections (ocr.py)
  -> vertical fragment merge        reassemble sliced vertical columns (ja only)
  -> manga-ocr (ja) / PaddleOCR rec (zh)    recognised source text
  -> Nemotron via NVIDIA NIM        english line
  -> placement                      erase source text, fit and paint translation
image
```

Japanese and Chinese take deliberately different paths. Detection is
PaddleOCR for both, with per-language thresholds, but Japanese recognition
switches to manga-ocr, which is trained on vertical manga text and handles
it far better than PaddleOCR's general recogniser. Japanese also needs the
fragment-merge step because the DB detector slices vertical columns into
pieces; Chinese does not, and runs the generic `merge_text_boxes` instead.

## Layout

```
app.py               Gradio UI, the Hugging Face Space entry point
backend/
  ocr.py             detection and recognition, per-language
  translation.py     NVIDIA NIM client, prompts, retry loop
  placement.py       erase and repaint translated text
  pipeline_core.py   geometry, colour, text-fitting helpers
  pipeline.py        orchestration for one page
  runner.py          seam between the HTTP layer and the pipeline
  stub_pipeline.py   no-ML stand-in used for integration testing
  batch_store.py     job queue, worker thread, result storage
  main.py            FastAPI app, routes, CORS
frontend/
  index.html         upload and batch progress
  reader.html        paged reader with per-region inspection
  app.js             shared API client
  style.css
tests/
  test_api.py        HTTP-level suite, no browser
  test_browser.py    real Chromium driving the HTML frontend
  test_gradio.py     real Chromium driving the Gradio app
docs/
  BUGS.md            bugs found, evidence, fixes
  DECISIONS.md       why the API and architecture are shaped this way
  LIMITATIONS.md     known defects and unverified areas
notebook/            the original Kaggle notebook the backend came from
```

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | config and worker state, does not load models |
| POST | `/api/translate` | one image, synchronous, returns regions + image URL |
| GET | `/api/image/{token}` | rendered PNG |
| POST | `/api/batch` | N images, returns `batch_id`, responds 202 |
| GET | `/api/batch/{id}` | poll status |
| GET | `/api/batch/{id}/page/{n}` | one finished page: image URL + regions |
| GET | `/api/batch/{id}/download` | zip of finished pages |
| DELETE | `/api/batch/{id}` | drop batch and free disk |

See `docs/DECISIONS.md` for why each of those is shaped the way it is.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | unset | NVIDIA NIM key. Translation fails without it. |
| `CORS_ALLOW_ORIGINS` | `*` | comma-separated origin allowlist |
| `USE_STUB_PIPELINE` | unset | `1` swaps in the no-ML stub |
| `MANGA_TRANSLATOR_DATA_DIR` | temp dir | where rendered pages are written |
| `FRONTEND_DIR` | `../frontend` | static files to serve |

## Running locally

```
pip install -r requirements.txt
cd backend
NVIDIA_API_KEY=... python -m uvicorn main:app --port 8000
```

Then open `http://127.0.0.1:8000/`.

To work on the frontend without the ML dependencies installed:

```
pip install fastapi uvicorn python-multipart pillow
cd backend
USE_STUB_PIPELINE=1 python -m uvicorn main:app --port 8000
```

Those four packages are the complete requirement for stub mode. Pillow is
the stub path's only third-party import, and `tests/test_api.py` asserts
that numpy, cv2, torch and paddleocr stay out of it.

The stub performs no detection, recognition or translation. It draws its
own placeholder regions and stamps the output image so a stub render can
never be mistaken for a real one.

## Tests

```
python tests/test_api.py       # 44 checks, no browser needed
python tests/test_browser.py   # 33 checks, needs: pip install playwright && playwright install chromium
python tests/test_gradio.py    # 13 checks, same playwright requirement
```

90 checks in total. All three run against the stub pipeline. They test the
HTTP layer, the job queue, CORS, and both front ends; they do not test
detection, recognition or translation quality. `docs/LIMITATIONS.md` is
explicit about that boundary.

## Two front ends

`app.py` is a Gradio UI and is what the Hugging Face Space serves.
`backend/main.py` is the FastAPI service that serves `frontend/`, and is
how the project runs locally.

Both drive the same `batch_store` and `runner`, so the job queue, the
single-worker invariant and the result storage are identical under either.
The Space does not serve `frontend/index.html` or `frontend/reader.html`.

To run the Gradio app locally:

```
pip install gradio
USE_STUB_PIPELINE=1 python app.py
```

## Deployment

The Space uses the Gradio SDK, which is free; Docker Spaces require
billing. `Dockerfile` is retained and still correct if billing is ever
added. `docs/DEPLOY.md` has the exact steps. The NVIDIA key is set as a
Space secret and never appears in any file in this repository.
