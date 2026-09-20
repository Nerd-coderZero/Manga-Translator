# Design decisions

Why things are the way they are. Written to be defended out loud, so each
entry states the alternative that was rejected and what it would have cost.

---

## Recovery: the backend was reconstructed from notebook cells

The backend previously lived in a sandbox that no longer exists. What
survived was a Kaggle notebook, `image-translator-cn-en (6).ipynb`.

Five of the six modules were recoverable because the notebook wrote them
with `%%writefile` cells, so the file bodies were intact in the source:
`pipeline_core.py`, `ocr.py`, `translation.py`, `placement.py`,
`pipeline.py`. They were extracted verbatim, with only the magic line
stripped, and each compiles.

`main.py` and `batch_store.py` were not in the notebook. Every cell was
searched for `FastAPI`, `CORSMiddleware`, `uvicorn`, `batch_store` and
`<html`, with no matches. They were rebuilt from scratch rather than
guessed at, on the reasoning that a fresh design that can be explained is
worth more than a reconstruction of a design nobody remembers.

---

## Why single translate is synchronous and batch is not

One page costs roughly 20 to 60 seconds: detection, then one LLM call per
text region, serially. The recovered run log shows those calls returning
503 from NVIDIA regularly and going through exponential backoff, which
stretches the tail considerably.

One page fits inside an HTTP request. Twenty do not, and would hit proxy
and gateway timeouts. So the split is drawn at the point where the work
stops fitting in a request.

The rejected alternative is making everything asynchronous for uniformity.
That would force a client to poll for a single-image translation, which is
the common case and the one a demo goes through, and gains nothing.

---

## Why the endpoints return JSON with regions rather than image bytes

`/api/translate` and `/api/batch/{id}/page/{n}` return the detected regions
with source text and translation, plus a URL for the image, instead of just
returning the PNG.

The rendered image alone discards everything that makes the pipeline
inspectable. The reader view needs the region data, and so does anyone
trying to understand why a particular line came out the way it did. It also
means a failure can be localised: if the image looks wrong you can see
whether OCR read the wrong text or the translation was wrong or placement
put it in the wrong place.

Cost: two requests instead of one for a single translation. Worth it.

---

## Why polling instead of WebSockets

A batch runs for minutes, and per-second progress granularity is more than
enough. Polling is four lines of client code, works through any proxy
without special configuration, and has no reconnection story to get wrong.

A WebSocket would buy sub-second updates that nobody needs and add a
failure mode to explain.

---

## Why one worker thread

The OCR model instances in `ocr.py` are process-global singletons
(`_ocr_models`, `_manga_ocr_model`) and are not safe to call concurrently.
A container also has one model's worth of memory to spend, and both
PaddleOCR and manga-ocr are resident at once on the Japanese path.

Concurrency of one is therefore the honest capacity of this service.
Putting the serialisation in `batch_store` makes it a stated invariant of
the store rather than something that happens to be true because requests
arrive slowly.

The rejected alternative is a worker pool, which would require per-thread
model instances and multiply memory by the pool size. On free Space
hardware that is not available.

`uvicorn --workers 1` in the Dockerfile is part of the same decision: a
second process would have its own model instances and its own copy of the
in-memory registry, so a poll could land on a process that has never heard
of the batch.

---

## Why metadata is in memory and images are on disk

Rendered manga pages are roughly 1 to 2 MB each and a batch can be forty of
them. Holding those in the process is a direct route to an out-of-memory
kill on small hardware. They go to disk.

Metadata is small, is read on every poll, and is worthless once the process
dies, so there is nothing to gain from persisting it. A database would add
a dependency and a schema in exchange for a durability guarantee that Space
storage cannot honour anyway, since it is ephemeral.

The consequence is recorded as LIM-2: a restart loses all batches. That is
the accepted cost, not an oversight.

---

## Why batch IDs and image tokens are random

`secrets.token_urlsafe(16)`, not sequential integers.

A Space can be public. Sequential IDs would let any visitor enumerate other
visitors' uploads and translated output by counting. The tokens are the
only thing standing between one user's pages and another's, so they are
unguessable.

Deleting a batch also revokes its image tokens, which the API suite
verifies: after a delete, a previously working image URL returns 404.

---

## Why CORS is configured from an environment variable

In the deployed Space the frontend is served from the same origin as the
API, so CORS is not involved at all. It matters for two other cases:
local development, where the frontend often runs on a different port, and
any future separately hosted frontend.

The allowlist comes from `CORS_ALLOW_ORIGINS` rather than being hardcoded
so the deployed Space does not have to carry development origins in its
source.

`allow_credentials` is deliberately `False`. The API has no cookies or
sessions so nothing needs it, and leaving it off means a wildcard origin
cannot be escalated into a credentialed cross-site read. This is why the
default of `*` is acceptable rather than reckless.

The browser suite runs the frontend on a genuinely different origin from
the API so that real preflights happen. The API suite additionally asserts
the negative case: a request from an origin outside the allowlist gets no
`Access-Control-Allow-Origin` header back. Testing only the positive case
would pass just as well against middleware that allowed everything.

---

## Why `/api/health` does not touch the models

Loading PaddleOCR and manga-ocr takes tens of seconds and most of the
container's memory. A health check that triggered a load would be a denial
of service against the thing it is meant to be checking, and on a platform
that health-checks on a timer it could hold the container permanently busy.

So health reports configuration and worker liveness: pipeline mode, whether
the API key is present, whether the worker thread is alive, queue depth. It
answers "is this deployed correctly" rather than "can it translate right
now", and the distinction is stated in `DEPLOY.md` so the check is not
mistaken for the latter.

---

## Why there is a stub pipeline

The real pipeline imports paddleocr, manga-ocr and torch, and downloads
model weights on first use. That is not available in every environment
where the HTTP layer, the job queue and the frontend need to be exercised,
and it makes an integration test slow enough that nobody runs it.

`runner.py` is a single seam. It picks the real or stub implementation
based on `USE_STUB_PIPELINE` and normalises the result. `batch_store` and
`main` import `runner` and never import a pipeline directly, so neither of
them contains any test-related branching.

Two properties make the stub safe:

- it reproduces `translate_page`'s return contract exactly, so it cannot
  hide a shape mismatch that the real pipeline would trip over
- every image it renders is stamped "STUB PIPELINE - NO TRANSLATION
  PERFORMED", and `/api/health` reports `pipeline_mode: stub`, which the
  frontend surfaces as a warning banner

A stub render cannot be mistaken for a real one, which is the failure mode
that makes stubs dangerous.

The boundary is recorded honestly in `LIMITATIONS.md` as UNV-1: these tests
prove the plumbing, not the machine learning.

---

## Why the static mount is registered last

`app.mount("/", StaticFiles(...))` is the final statement in `main.py`.
Starlette matches routes in registration order, so mounting the frontend at
the root before the API routes were declared would shadow every one of
them. The API suite checks that both `/index.html` and the API routes are
reachable, which would fail if this were reordered.

---

## Why the zip entries are index-prefixed

`0000_page1.png`, `0001_page2.png`, and so on. Zip readers and file
managers sort lexicographically, not by insertion order, so without the
prefix a 12-page batch would present as 1, 10, 11, 12, 2, 3 and the reading
order would be destroyed. The original notebook hit exactly this and had a
`natural_sort_key` helper to work around it after the fact; prefixing at
write time is the simpler fix.

---

## Why the Space uses the Gradio SDK and not Docker

The original plan was a Docker Space, because Docker gives exact control
over the base image, the apt packages and the entry point, and the
`Dockerfile` was written first.

That plan was wrong on a fact: Docker Spaces require billing. The Space
creation page marks the Docker SDK "Paid", with the tooltip "Add billing to
your account (credits or subscribe to PRO) to unlock Docker Spaces". Of the
three SDKs, only `gradio` and `static` are free, and `static` has no
backend.

Gradio covers what Docker was being used for: `requirements.txt` for Python
packages and `packages.txt` for apt packages, both documented for Gradio
Spaces. What it does not give is a custom entry point, since `app_port` is
documented as Docker-only and Gradio Spaces run `app.py` on 7860.

Rejected alternative: keep FastAPI as the Space's server by mounting a
token Gradio block onto it with `gr.mount_gradio_app` and running uvicorn
from `app.py`. That would have preserved `frontend/index.html` and
`reader.html` as the deployed UI. It was rejected because whether the
Gradio SDK's health checking tolerates an app whose root is not a Gradio
interface was unknown, and shipping an unverified deployment mechanism to
fix a deployment problem is circular. A Gradio UI that was actually tested
beats a FastAPI mount that was not.

`Dockerfile` is kept. It is still correct, and switching `sdk` back to
`docker` in the README config block is the whole migration if billing is
ever added.

## Why app.py drives batch_store instead of calling the pipeline directly

The Gradio handler could have called `runner.run_page` in a loop. It goes
through `store.create_batch` and polls instead.

The reason is specific to a public Space: several visitors can submit at
the same moment. The OCR model instances in `ocr.py` are process-global and
not thread-safe, so two concurrent translations would touch the same models
from different threads. The store's single worker serialises them.

Gradio's `default_concurrency_limit=1` is set as well. The two do different
jobs and both are wanted: the Gradio limit bounds how many requests are in
flight, and the worker thread bounds how many translations actually reach
the models. Relying on the Gradio limit alone would put the invariant in
the UI layer, where a later change to that number would silently break it.

The side benefit is that per-page progress in the Gradio UI comes from the
same `batch.to_dict()` the HTTP API returns, so the two front ends cannot
disagree about what a batch is doing.

## Why the region table is hand-rendered HTML

`gr.Dataframe` is the obvious component for four string columns. It
postprocesses through pandas, which requires jinja2 3.1.2 or newer, and
that import failed during local verification.

A table of four string columns does not justify a pandas dependency,
especially on a Space whose build has not been proven. `gr.HTML` with
escaped cells has no dependencies at all. `html.escape` is applied to every
cell because the source column contains OCR output, which is untrusted
text.

## Why app.py defines a @spaces.GPU function it never really needs

A real deploy failed at the platform level with "No @spaces.GPU function
detected during startup," even though the container had started correctly
and Gradio was bound to its port. ZeroGPU Spaces, it turns out, refuse to
serve traffic unless at least one function decorated with `@spaces.GPU`
exists somewhere in the app — a requirement independent of whether this
codebase actually wants GPU acceleration, which it does not: paddlepaddle
is the CPU build, and detection, recognition and placement all run on CPU
regardless.

The alternative to satisfying this honestly would have been decorating
something load-bearing purely to make the platform happy, which risks
tangling the requirement into code that has other jobs. Instead,
`_zerogpu_probe` is a small standalone function with no callers from the
actual translation path. Its only job is to exist so the platform's
startup scan finds it.

Since a function had to exist anyway, it was made to do one small useful
thing rather than nothing: return `torch.cuda.is_available()`. This
directly answers a question `docs/LIMITATIONS.md` had previously left open
under UNV-7 — what CUDA visibility looks like under ZeroGPU's emulation
outside a decorated call — for the cost of one log line at startup. The
call is wrapped in a broad `except Exception` because torch is not
installed in stub mode, and a diagnostic that crashes local development is
worse than one that occasionally prints nothing.

Confirmed locally, separate from the platform requirement: importing
`spaces` and calling an `@spaces.GPU`-decorated function outside any HF
Space infrastructure executes normally rather than erroring, which is
consistent with HF's own documentation that the decorator is "effect-free
in non-ZeroGPU environments." That was verified by running it directly in
a sandbox with no GPU present, before trusting it not to break local
development or the test suites.

## Why translation has a free fallback and thinking is off for Nemotron

Two changes to `translate_text_nemotron`'s request, made together and kept
together because the second only became necessary once the first was
tried: `enable_thinking` was switched to `False`, `temperature` to 0.3,
`top_p` to 0.9, and `max_tokens` was capped at 512.

The original call ran with thinking on and no token cap, which is
correct for a model meant to reason before answering but is the wrong
shape for this task: a real bulk run showed it adding meaningful latency
per line, and worse, a fixed token budget with thinking on risks the
model spending the entire budget on reasoning tokens and returning empty
content, which reads identically to a hard failure downstream. Lower
temperature and top_p were kept for the same reason a translation task
generally wants them: less variance line to line, matching the terse,
consistent register the system prompt already asks for.

That alone does not solve the empty-content risk, only makes it less
likely. `TRANSLATION_PROVIDER` (env var, default `auto`) adds a real
fallback chain: Nemotron first, with retries capped at 2 in auto mode
rather than the old 5, since a fallback existing makes patiently retrying
a flaky line worse than handing it to a different translator; then
`deep-translator`'s keyless `GoogleTranslator`; then its `MyMemoryTranslator`
if Google also fails. `_is_valid()` treats a blank result or one that
still contains CJK characters (the model echoing the source instead of
translating it) as a miss, not just an exception, so a silent wrong
answer gets the same fallback treatment as a hard error.

Verified on a real 90-page Chinese bulk run: `sent to fallback: 4, google:
0, mymemory: 4` -- every real Nemotron failure in that run was rescued by
the fallback chain, and Google specifically returned nothing usable any
of the 4 times it was tried, which is why MyMemory is second in the chain
rather than the only fallback: a single free provider was not enough on
its own, confirmed rather than assumed. `provider="fast"` (skip Nemotron
entirely) and `provider="nemotron"` (old behaviour, no fallback) are kept
as explicit options rather than removed, for bulk runs that want to
avoid the NVIDIA rate limit entirely or that need the old patient-retry
behaviour for comparison.

Rejected alternative: raising `max_tokens` instead of turning thinking
off, to give the model room for both reasoning and an answer. Not tried,
because the latency cost of thinking is the more important problem for a
bulk pipeline translating hundreds of short lines, and a much larger
token cap makes the empty-content failure mode rarer but not impossible,
where turning thinking off removes the mechanism entirely.

## Why the batch cap was raised from 40 to 100, and why that isn't a throughput change

`MAX_PAGES_PER_BATCH` was 40, chosen (per the "metadata in memory / images
on disk" decision above) against a rough disk-space budget: rendered pages
are 1-2 MB each, and a batch sits on disk for up to `DEFAULT_TTL_SECONDS`
(3 hours) before `evict_expired` reclaims it. That reasoning was re-checked,
not just raised on request: rendered pages still live on disk, not in the
process, so a bigger cap is a disk-space and TTL question, not a memory
one. A 90-page real Kaggle bulk run (see `docs/BUGS.md` BUG-7 and BUG-8)
confirmed the pipeline itself handles a batch that size correctly, which
is real evidence the old ceiling was conservative relative to genuine use.
100 was chosen as headroom above that real test, not an arbitrary round
number picked without one.

This is explicitly not a throughput fix. LIM-3 (one worker thread, pages
translate strictly serially) is unchanged by this: a 100-page batch is a
longer wait, not a faster one. Raising the cap without also addressing
LIM-3 was a deliberate, separate decision -- the two are independent
constraints and conflating them would have meant either not raising the
cap (leaving real bulk use worse off for no memory reason) or attempting a
worker-pool change with much higher risk in the same pass as a one-line
constant change. The pre-existing comment in `batch_store.py` that said
"holding them in the process is... a route to an out-of-memory kill" was
also corrected while touching this -- rendered images were never actually
held in the process; only metadata is, and metadata scales with page
*count*, not page bytes, so it was not the real constraint even at 40.

## Why the reader has three modes instead of improving the single-page view alone

The static frontend's reader was strictly one-page-at-a-time
(Previous/Next, no overview), which was fine at a page count of a handful
but did not scale once the batch cap went up: reaching page 40 of 100
meant 39 clicks with no way to see where you were headed. Three real gaps
existed, not one, so the fix addresses all three rather than picking the
most obvious:

- **No overview or jump-to-page.** Fixed with a thumbnail rail, present in
  both single and long-strip modes, that loads each thumbnail lazily
  (reusing whatever the page cache already has, or fetching on first
  render) and lets a click jump straight to any page.
- **No mode suited to fast, continuous reading of many pages in a row.**
  Fixed with a long-strip mode -- every readable page stacked vertically,
  each image loaded only when it scrolls near the viewport via
  `IntersectionObserver` with a 600px lookahead margin, not all up front.
  For a 100-page batch, loading every image eagerly would be exactly the
  kind of unbounded-request-fan-out this project has been careful to
  avoid elsewhere (see the batch/worker design); lazy loading keeps the
  strip's real cost proportional to how far the user has actually
  scrolled.
- **No immersive, distraction-free page for actually reading the art.**
  Fixed with a fullscreen mode -- the page fills the viewport, the
  regions panel and thumbnail rail are hidden, and left/right click zones
  plus arrow keys and Escape handle navigation and exit.

Rejected alternative: replacing the one-page reader outright with a grid
view showing every page at once, closer to the Gradio Space's gallery.
Rejected because a manga reader's primary job is sequential reading in
order, which a flat grid actively works against, and because the existing
single-page mode with prefetch (see below) already serves that case well
-- the actual gaps were the lack of an overview and the lack of a
continuous-scroll option, which the thumbnail rail and strip mode solve
directly without discarding what already worked.

The single-page mode's existing prefetch-one-ahead behaviour, the
Previous/Next buttons, and arrow-key navigation are unchanged; the
regions panel, error banners, and the "N page(s) failed and are not
shown" warning all continue to work identically across all three modes,
since all three read from the same `readablePages`/`pageCache` state
rather than each maintaining their own.

Two real bugs were found and fixed while building this, both caught by
running a real Playwright session against the real stub-backed API rather
than by inspection: the thumbnail rail did not appear at all on initial
load, because visibility was only ever toggled inside the mode-switch
handler and `init()` renders the first page directly without going
through it -- fixed by having the rail's own render function own its
visibility. Separately, in fullscreen mode the right-side click-to-advance
zone and the exit button shared the same z-index and sat in DOM order
after it, so the zone intercepted every click intended for the exit
button -- fixed by raising the button's z-index above the zones and
clearing the top 56px of the click-zones so the two can never overlap
regardless of z-index. Both are now covered by regression checks in
`tests/test_browser.py` rather than only having been observed once.

## Why the reader prefetches one page ahead

Manga is read page by page in order, so the next page is nearly always the
one wanted next. Prefetching one ahead makes navigation feel immediate at
the cost of one extra request. Prefetch failures are swallowed, because a
failed prefetch is retried naturally when the user actually navigates and
surfacing it would be noise.
