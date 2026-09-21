# Known limitations and unverified areas

Split into two kinds of entry. **Known defects** are diagnosed problems
that are not fixed. **Unverified** areas are things that may well be
correct but have not been tested, and are listed so nobody mistakes
absence of a failure report for evidence of correctness.

---

# Known defects

## LIM-1 Translated text is drastically oversized on vertical Japanese regions

**Status:** diagnosed, not fixed. Parked deliberately: the data needed to
verify a fix is not available.

### Symptom

On Japanese pages, translations painted into tall narrow vertical regions
render at an enormous font size and wrap into near-vertical stacks of one
or two characters per line. In the recovered end-to-end run:

- `「おしり」っ！！` rendered as `B / ut / t!!` in letters roughly a third
  of the panel height
- `店長！` rendered as `M / an / ag / er!`
- `試してみたいでしょうッ！？` overflowed its region across the artwork

Meanwhile text in wide oval speech balloons, which merge horizontally
rather than vertically, renders at a sensible size on the same page. That
contrast is the strongest single piece of evidence: the defect tracks
region shape, not the text.

### Where it comes from

`placement.py` sizes the font from the height of the original text:

```python
original_height = box.get("avg_original_height", box_height)
target_font_size = max(int(original_height * 0.85), min_acceptable_font_size)
```

The name says what the value is supposed to be: the average height of the
individual source text fragments. For the Chinese path that is what it is.
`pipeline_core.merge_text_boxes` computes it correctly:

```python
individual_heights = [(bounds_list[i][3] - bounds_list[i][1]) for i in indices]
avg_original_height = sum(individual_heights) / len(individual_heights)
```

The Japanese path never reaches that code. `pipeline.py` bypasses
`merge_text_boxes` for `source_lang == "ja"`, because `ocr.py` has already
done its own merging, and fills the field in with a fallback:

```python
b.setdefault("avg_original_height", b["bounds"][3] - b["bounds"][1])
```

That is the full height of the merged region, not the average height of a
character in it. For a vertical column of six characters spanning 270 px,
the font size requested is `270 * 0.85`, about 229 px, when the correct
basis is a single character of roughly 45 px.

The reason the fallback exists at all is upstream:
`ocr.py._merge_vertical_fragments` discards the per-fragment bounds when it
builds a merged region. It keeps only the union rectangle and
`source_fragment_count`. The information needed to compute a real average
is destroyed before `pipeline.py` runs, so `pipeline.py` has nothing better
to fall back on.

### What is confirmed and what is not

Confirmed:

- the code path is exactly as described, read directly from the recovered
  source
- the rendered output in the recovered notebook shows the symptom on a real
  page
- the symptom correlates with region shape in the direction the diagnosis
  predicts: tall narrow regions are wrong, wide regions on the same page
  are right

Not confirmed:

- that fixing `avg_original_height` alone produces correct output. The
  placement stage has further logic after the font-size decision, including
  region expansion and a shrink-to-fit loop, and that logic has never been
  observed running against a corrected value.

### What a fix would involve

Carry per-fragment heights through the merge in
`ocr.py._merge_vertical_fragments`. It already has the fragment bounds in
hand when it builds each region; it would attach the median fragment
height to the region alongside `source_fragment_count`, and
`_detect_and_read_japanese` would pass it through to the returned box.
`pipeline.py` would then read that value instead of falling back to the
region height. Median rather than mean, because a vertical column often
picks up one or two much larger fragments from adjacent art or a stray
detection, and a mean would be dragged upward by them.

### Why it is not fixed

Verifying it requires rendering a real page and looking at the result. That
needs the clean source image, which was not recovered: the notebook
contains only rendered and annotated derivatives with boxes already drawn
on them. Writing the change without being able to run it would produce
something that looks plausible and is documented as fixed, which is worse
than a documented defect.

To pick this up: supply a clean Japanese page image and, ideally, the
Chinese equivalent to confirm the change does not regress the path that
currently works.

---

## LIM-2 Batch state does not survive a restart

A batch lives in a Python dict in one process. Restarting the container, or
a Space going to sleep, loses every in-flight and completed batch. Rendered
pages on disk become unreachable because the tokens that address them are
gone with the registry.

This is a deliberate trade, not an oversight. See `DECISIONS.md`. It is
listed here because it is a real user-visible behaviour: a user who leaves
a tab open across a Space sleep gets a 404 on their batch, and the frontend
reports it as "batch not found or expired".

## LIM-3 Throughput is one page at a time

One worker thread by design, because the OCR models are process-global and
not thread-safe. A 20-page batch is 20 sequential page translations, each
of which is itself a serial loop of one LLM call per text region. Real
throughput on the recovered run was roughly one region per several seconds
once NVIDIA's 503 backoff is included.

The obvious speedup is not more workers, which the model instances forbid,
but batching the translation calls: one request carrying all of a page's
lines instead of one request per line. That changes the prompt contract and
has not been attempted.

`MAX_PAGES_PER_BATCH` was raised from 40 to 100 (see `docs/DECISIONS.md`)
on the strength of a real 90-page run completing correctly, but that
change does not touch this limitation at all: pages still translate one at
a time, so a 100-page batch is a proportionally longer wait, not faster
throughput. The two were deliberately kept as separate, independent
decisions rather than solved together.

## LIM-6 The deployed Space's UI has no jump-to-page, no working fullscreen, and no zoom

**Status:** understood, not fixed. The reader built for this (single page /
long strip / fullscreen modes, thumbnail jump-to-page -- see BUGS.md and
the reader entry in DECISIONS.md) exists and is tested, but it is not
reachable from the live Space at all.

### Why

The Hugging Face Space deploys as a Gradio SDK Space, which runs `app.py`
directly (`demo.launch(...)`). `app.py` has no reference anywhere to
`frontend/`, `main.py`, or `reader.html` -- confirmed by grep, zero
matches. The reader and its three modes are served only by
`backend/main.py`'s FastAPI static mount, which is a different way of
running this project (`uvicorn main:app`) than what the Space actually
runs. A user on the deployed Space has only ever seen `gr.Gallery`, which
has no jump-to-page, no long-strip mode, and no dedicated fullscreen of
this project's own design -- what looks like a stuck, unresponsive
"fullscreen" is Gradio's own built-in image-preview lightbox
(`gr.Gallery(preview=True)`), a third-party component this project does
not control the behaviour of.

### What was fixed instead, in this same investigation

A real, separate, structural performance bug was found and fixed while
diagnosing this: `app.py`'s `translate()` generator was rebuilding and
re-sending the *entire* gallery (full-resolution PNGs) and the *entire*
region table on every 0.5-second poll tick for the whole batch duration,
regardless of whether anything had actually finished since the last tick.
Confirmed by reading `gradio/components/gallery.py`'s `postprocess`
directly: `gr.Gallery` and `gr.HTML` have no incremental-update path, so
every yield with a real (non-`gr.update()`) value tears down and rebuilds
the whole component client-side. At 40-100 pages this got worse as the
batch progressed, matching the reported symptom ("lag once past 10
images, still slow after done translating"). Three changes, all verified
with a real 12-page batch through a real headless browser session:
gallery images are now capped to a 1100px-long-edge JPEG thumbnail
(`_thumbnail_path`, cached per source file) rather than the full 1-2 MB
rendered PNG; the region table shows only the 5 most recently completed
pages rather than growing without bound, with a note that every page's
data is still in the downloadable zip; and the gallery/table are only
rebuilt when the completed-page count has actually changed since the
last tick, using `gr.update()` (a genuine no-op) otherwise. See BUGS.md
and DECISIONS.md for the full write-up.

This fixes the lag. It does not fix the missing jump-to-page or the
Gradio lightbox getting stuck, because those are inherent to `gr.Gallery`
as a component, not something tunable in `app.py`.

### What a real fix would involve

Either (a) reworking the Space to run FastAPI with Gradio mounted inside
it (`gr.mount_gradio_app`), so the existing, tested `reader.html` becomes
reachable on the deployed Space -- previously considered and rejected
once already for being an unverified deployment mechanism (see "Why the
Space uses the Gradio SDK and not Docker" in DECISIONS.md), so revisiting
it would need to actually prove the mount works under ZeroGPU before
relying on it, not assume it does this time; or (b) building genuine
jump-to-page and a project-controlled fullscreen directly out of Gradio
primitives (for example, a hidden `gr.Number` driven by JS thumbnail
clicks, and a custom fullscreen overlay via `elem_id`-targeted CSS/JS
rather than `gr.Gallery`'s own preview). Neither was attempted this pass;
the user asked for whichever fixes the lag fastest and with least risk,
and the lag was the one fixable within that constraint.

## LIM-4 Uploads are not validated beyond extension and size

`main.py` checks the file extension and the byte length. It does not verify
that the bytes are actually a decodable image. A file named `.png`
containing arbitrary data reaches the pipeline and fails there, surfacing
as a 500 on the single endpoint or a failed page in a batch. The failure is
contained and reported, but the error message is a decoder exception rather
than something useful.

## LIM-5 `is_garbage_text` cannot distinguish real repeated syllables from OCR noise

**Status:** diagnosed against real data (see BUG-8 in `docs/BUGS.md`), not
fixed. Parked because no safe rule was found, not because it was not
attempted.

After BUG-8 excluded punctuation from the repeated-character-dominance
check, 2 of the original 42 real empty regions from a 90-page bulk run
were still flagged: a repeated sound effect
(`嗷嗷嗷嗷！！！咳！！！嗷嗷嗷！！`) and repeated mocking laughter
(`他女儿~哼哼哼哼哼哼哼~谁知道呢`). Both are genuine manga content that
deliberately repeats one real CJK character for emphasis. Structurally
that is indistinguishable from a detector or recognizer misreading the
same character repeatedly across a noisy region -- both produce a string
dominated by one repeated, valid, real character.

A curated whitelist of known onomatopoeia syllables (哼, 呼, 喝, 嗷, and
so on) was considered and not implemented: writing one without a larger
real sample of both genuine sound effects and genuine repeated-character
OCR noise to validate against would be guessing, and a wrong guess in
either direction is a new false positive or false negative added without
evidence. This is deliberately left as an open limitation rather than
"fixed" with an unverified heuristic.

To pick this up: a larger real sample of both failure shapes -- confirmed
onomatopoeia and confirmed repeated-character misreads, from real pipeline
output -- would make a whitelist or a frequency-based rule verifiable
rather than guessed.

---

# Unverified

## UNV-1 The real pipeline has never run in this environment

Everything verified during this rebuild ran against the stub pipeline. The
stub reproduces `translate_page`'s return contract exactly and is enough to
test HTTP behaviour, the job queue, CORS and the frontend, but it performs
no detection, no recognition and no translation.

Not exercised end to end here:

- PaddleOCR detection and recognition
- manga-ocr recognition
- the NVIDIA NIM translation call and its retry loop
- `placement.py` rendering

These did run in the original Kaggle notebook, and its recorded output is
the evidence that they work: 23 merged regions, 0 skipped, and a full set
of Japanese-to-English translations on a real page. That evidence is from
2026-09-02 and predates this repository's restructuring. The restructuring
did not change any of those modules' contents, but nothing has run them
since.

## UNV-8 paddlepaddle version has drifted from what was proven working

**Status: the crash this caused is fixed (BUG-5). What remains here is
narrower than the original entry.**

The original Kaggle notebook proved `paddlepaddle==2.6.1` and
`paddleocr==2.7.3` working against real manga pages -- that is the entire
evidentiary basis for BUG-1 and BUG-2 in `docs/BUGS.md`. That exact pair no
longer exists on PyPI (`paddlepaddle==2.6.1` specifically), so
`requirements.txt` leaves both unpinned and the Space resolves
`paddlepaddle==3.3.1` / `paddleocr==3.7.0` instead -- confirmed from the
real build log, not assumed. That turned out to be a full class rewrite
(`paddleocr` 3.x is built on a new `paddlex` dependency), which broke both
the Japanese and Chinese detection paths. BUG-5 in `docs/BUGS.md` covers
the fix and what was verified directly against the installed 3.7.0
package.

What BUG-5 did *not* verify, because it is a question about output
quality rather than whether the code runs, and requires a real manga page
rather than synthetic text:

- **The detection thresholds may need retuning.** `_LANG_CONFIG`'s
  `text_det_thresh` / `text_det_box_thresh` / `text_det_unclip_ratio`
  values are carried over unchanged from the 2.6.1-era notebook, tuned
  against that version's detector model (`ch_PP-OCRv4_det`, confirmed from
  the notebook's own download log). The Space's resolved version loads a
  different model generation (`PP-OCRv6_medium_det`, confirmed from a real
  local run). Whether thresholds tuned for a v4-era detector still produce
  well-shaped boxes on a v6-era one -- neither too tight nor too loose --
  is unverified. BUG-1's containment filter operates on whatever boxes
  come out of detection, so a change in the detector's own box-shaping
  behaviour could shift how often that filter's area/containment
  thresholds fire, independent of whether the filter's own logic is
  correct.
- **manga-ocr's version also drifted.** The Space resolves
  `manga-ocr==0.1.16`; the notebook's evidence is against `0.1.14`. Its
  recognition call has not run anywhere in an environment available while
  building this project, so whether its output quality or interface
  matches what BUG-1 and the placement logic were verified against is
  unknown.
- **The Japanese path now runs PaddleOCR's own recognizer and discards
  it**, since `.predict()` no longer offers a detection-only call (see
  BUG-5). This is a performance regression, not a correctness one, but it
  means every Japanese page now pays for two recognition passes (PaddleOCR's,
  thrown away, then manga-ocr's) where one used to run.

None of this is fixed by getting the code to run without crashing. The
next real verification step is a real manga page through the deployed
Space, read for whether detection boxes and recognized text look right --
not just whether an exception is raised.

## UNV-2 The Space has never been built

The project deploys as a Gradio SDK Space. That build has never run. The
pinned versions in `requirements.txt` are chosen from what the notebook
used and are not proven to resolve together on the Space's Python 3.10
image, and `packages.txt` has never been applied.

One conflict in this class has already been found and fixed locally:
`gr.Dataframe` postprocesses through pandas, which required a newer jinja2
than was installed, so the region table is hand-rendered HTML instead.
That is evidence the risk is real rather than theoretical.

`Dockerfile` has never been built either. It is retained as the migration
path if billing is ever added to the account, since Docker Spaces are not
free. See `DEPLOY.md`.

## UNV-6 The Gradio UI has only run against the stub

`tests/test_gradio.py` drives the real `app.py` in real headless Chromium
and passes 13 checks: the page serves, a multi-file upload is accepted,
intermediate progress is published before completion, the gallery renders
and decodes images, the region table fills, and the zip download appears.

Every one of those ran against the stub pipeline. The Gradio layer is
verified; what it displays is not. In particular, no real OCR output has
ever been rendered into that region table, so how it handles long Japanese
source lines or unusual characters is unknown.

## UNV-7 Behaviour of torch-based code under ZeroGPU's CUDA emulation

Partially resolved. A real deploy revealed a harder requirement than the
one this entry originally described: ZeroGPU Spaces refuse to serve
traffic at all unless at least one `@spaces.GPU`-decorated function exists
in the app, independent of whether it is ever called. The Space failed
with "No @spaces.GPU function detected during startup" even though the
container had started correctly and was bound to its port. `app.py` now
imports `spaces` and defines `_zerogpu_probe`, decorated with
`@spaces.GPU`, to satisfy that requirement. See `docs/DECISIONS.md` for
the reasoning.

What is still open: `_zerogpu_probe` is called once at startup and its
result (`torch.cuda.is_available()`) is printed to the container logs, but
that call is wrapped in a broad `except Exception` specifically because
torch is not installed in stub mode, so a failure there is swallowed
rather than surfaced. Whether `manga-ocr`'s own torch usage — which is not
routed through `@spaces.GPU` at all — behaves correctly under whatever
CUDA visibility ZeroGPU presents outside a decorated call is still
unverified, because the real pipeline has never run in any environment
used while building this project (the same gap as UNV-1). Once the Space
is confirmed running, the fix is to read the "ZeroGPU probe:" line in the
container logs first; if it says `True`, that is new information worth
following up on, since nothing in this codebase currently routes actual
OCR or recognition work through the decorator, and a translation failure
that looks device-related would be the next thing to investigate.

## UNV-3 The Chinese path has not been re-run

The recovered notebook's end-to-end evidence is a Japanese page. The
Chinese path uses different detection thresholds, PaddleOCR's own
recogniser instead of manga-ocr, and `merge_text_boxes` instead of
`_merge_vertical_fragments`. It is the older of the two paths and was
presumably working, but there is no recorded output in the recovered
notebook demonstrating it.

## UNV-4 TTL eviction has not been observed firing

`BatchStore.evict_expired` is called on every batch submission and drops
batches older than three hours. The logic is simple and the deletion path
it calls is tested, but a three-hour TTL was not waited out, and no test
overrides the TTL to force an eviction. The code path that removes an
expired batch has therefore never executed.

## UNV-5 Behaviour under concurrent submissions is untested

The registry is lock-protected and the queue is a `queue.Queue`, so the
design should be correct under concurrent submits. Every test in the suite
submits from a single client in sequence. Nothing has verified the locking
under actual contention.
