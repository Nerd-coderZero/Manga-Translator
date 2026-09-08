# Deploying to Hugging Face Spaces

The Space runs on the **Gradio SDK**, which is free. `app.py` at the
repository root is the entry point.

## Correction to an earlier version of this document

An earlier revision of this file told you to create a **Docker** Space.
That was wrong. Docker Spaces are not available on a free account. The
Space creation page shows a "Paid" badge on the Docker SDK with the
tooltip:

> Add billing to your account (credits or subscribe to PRO) to unlock
> Docker Spaces

The project was moved to the Gradio SDK as a result. `Dockerfile` is kept
in the repository and is still correct, so if you later add billing you can
switch `sdk: docker` back in `README.md` and use it unchanged. Nothing was
thrown away.

## What runs where

The repository has two front ends over one backend:

- `app.py` — Gradio UI. This is what the Space serves.
- `backend/main.py` — the FastAPI service with the HTML frontend in
  `frontend/`. This is how the project runs locally and is what the API and
  browser test suites exercise.

Both drive the same `batch_store` and `runner`, so the job queue, the
single-worker invariant and the result storage are identical under either.
The Space does not serve `frontend/index.html` or `frontend/reader.html`.

Nothing below has been executed. No account was created, no Space was
created, and nothing was pushed. These are the steps you run.

## Before you start

You need a Hugging Face account and an NVIDIA NIM API key. The NVIDIA key
is never written to a file in this repository and must not be committed; it
goes in as a Space secret in step 4.

## 1. Create the Space

On huggingface.co: **New** > **Space**.

- Owner: your account
- Space name: `manga-translator`
- License: your choice
- SDK: **Gradio** > **Blank**
- Hardware: **ZeroGPU**. As of this writing, CPU Basic on the Space
  creation page is greyed out with the tooltip "On the free tier, Gradio
  Spaces run on ZeroGPU. Subscribe to PRO for unlocking free cpu-basic
  flavor." ZeroGPU is the free tier now, not an alternative to it.
- Visibility: Public or Private. "Protected" requires PRO.

### Why ZeroGPU is fine for this app

ZeroGPU only attaches a real GPU to functions decorated with `@spaces.GPU`.
Nothing in `app.py` or `backend/` imports the `spaces` package or uses that
decorator, so nothing in this codebase ever requests a GPU. The Space
should run on CPU exactly as tested locally, and the free account's 5
minutes/day GPU quota is not something this app can hit, since it never
asks for GPU time.

The one part of this that has not been verified: HF's docs state that
outside a `@spaces.GPU` function, a "CUDA emulation mode" is active so
`torch` calls do not error even without a real GPU attached. `manga-ocr`
uses `torch` and typically auto-detects a CUDA device if `torch.cuda.is_available()`
reports one. Whether that emulation causes it to report `True` and whether
`manga-ocr` then behaves correctly under it has never been observed,
because the real pipeline has never run in any environment available while
building this project. If the Space's translate call fails in a way that
looks device-related, that is the first place to look. See UNV-7 in
`docs/LIMITATIONS.md`.

Do not pick a Gradio template other than Blank; `app.py` in this repository
is the app.

## 2. Point this repository at the Space

From `C:\Users\Admin\Desktop\Claude Code\Projects\Manga Translator`:

```
git init
git add .
git commit -m "Manga translation pipeline: backend, Gradio Space, tests, docs"
git remote add space https://huggingface.co/spaces/<your-username>/manga-translator
```

## 3. Check what you are about to push

```
git status
```

`notebook/` contains the original Kaggle notebook with embedded output
images of real manga pages. Decide whether you want that in a public Space
before pushing. If not:

```
echo "notebook/" >> .gitignore
git rm -r --cached notebook
git commit -m "Exclude source notebook from the deployed Space"
```

There is also a duplicate copy of the notebook at the repository root
(`image-translator-cn-en (6).ipynb`) alongside the one in `notebook/`.
Delete whichever you do not want.

## 4. Set the secret

In the Space: **Settings** > **Variables and secrets** > **New secret**.

- Name: `NVIDIA_API_KEY`
- Value: your key

Add it as a **secret**, not a variable. Variables are visible in the Space
UI; secrets are not. Set this before the first push so the container has it
on its first boot.

Do not set `USE_STUB_PIPELINE` in the Space. If it is set to `1` the Space
will run the stub and translate nothing.

## 5. Push

```
git push space main
```

Hugging Face will ask for your username and an access token with write
scope (Settings > Access Tokens on huggingface.co). Your account password
will not work.

## 6. Watch the build

The Space page shows build logs. Expect a slow first build: paddlepaddle,
paddleocr and torch are on the order of 2 GB of wheels.

`packages.txt` installs the Debian packages the pipeline needs: `libgl1`
and `libglib2.0-0` for opencv, and `fonts-dejavu-core` plus
`fonts-liberation` because `pipeline_core.find_available_font` looks for a
bold TrueType face on disk. Without those fonts, placement falls back to a
bitmap font that cannot be scaled and every translated line renders at one
fixed size.

`startup_duration_timeout: 1h` is set in the README config block because
the default is 30 minutes and a cold first run downloads model weights on
top of installing them.

## 7. Verify

The Space has no `/api/health` endpoint. The Gradio app reports the same
information in the UI instead, as a line under the title. It is written by
`_startup_notice()` in `app.py` and says one of three things:

- **"Backend is running the STUB pipeline"** — `USE_STUB_PIPELINE` is set
  in the Space environment. Remove it.
- **"NVIDIA_API_KEY is not configured"** — step 4 did not take. Detection
  will run but every translation call will fail.
- **"Ready. The first page after a cold start is slow"** — correct.

Once you see the third message, upload one page and translate it.

If you want the JSON health check as well, run the FastAPI service instead
of, or alongside, the Gradio app. It is unchanged and still serves
`/api/health`.

## Cold starts

Models are not loaded at startup. The cost lands on the first translation
request, which downloads model weights on top of loading them, so that
first request can take several minutes. Subsequent requests reuse the
process-global model instances.

On a free Space this compounds: Spaces sleep when idle, and a woken Space
has an empty model cache again, so the first request after every sleep pays
the download cost. Persistent storage is no longer offered on Spaces, so
the options are paid hardware that does not sleep, or accepting it.

`preload_from_hub` in the README config block can pull manga-ocr's weights
at build time rather than first-request time. It is not set, because it has
not been tested here and a wrong entry fails the build.

## What has been verified and what has not

**Verified**, by running it:

- `app.py` serves, accepts a multi-file upload, publishes intermediate
  progress before completion, renders the gallery, fills the region table
  and produces a downloadable zip. 13 checks in `tests/test_gradio.py`,
  driven by real headless Chromium against the stub pipeline.
- The FastAPI service still passes 44 API checks and 33 browser checks.
- All backend modules compile.

**Not verified:**

- The Space has never been built. The pinned versions in
  `requirements.txt` are chosen from what the notebook used and are not
  proven to resolve together on the Space's Python 3.10 image.
- `packages.txt` has never been applied.
- The real pipeline has still never run in this environment. PaddleOCR,
  manga-ocr, the NVIDIA NIM call and `placement.py` rendering are evidenced
  only by the original Kaggle notebook's recorded output.
- Memory headroom on CPU Basic with both OCR models resident is unknown.

The dependency resolution is the first thing to expect trouble from. One
conflict has already been found and fixed locally: `gr.Dataframe`
postprocesses through pandas, which required a newer jinja2 than was
present, so the region table is rendered as plain HTML instead. Expect
more of that class of problem on the first build, and read the log rather
than guessing.
