# Bugs

Each entry records what broke, why, how the diagnosis was confirmed against
real data, what the fix does, and how the fix was verified. Bugs that are
diagnosed but not fixed live in `LIMITATIONS.md`, not here.

---

## BUG-1 Redundant oversized detection boxes in Japanese OCR

**Status:** fixed and verified against real page data.

### Symptom

On dense vertical Japanese text, a single text cluster produced two
competing detections: the correct set of small per-fragment boxes, and one
large box covering the whole cluster. Both survived into the merge stage.
The large box was then treated as a text region in its own right, so the
same text was recognised twice and the oversized box distorted the merge.

### Root cause

PaddleOCR's DB detector emits both. They are separate entries in the raw
`dt_boxes` output and nothing downstream reconciles them. This is detector
behaviour, not a bug in our merge logic, so it has to be handled before
merging rather than inside it.

### How it was confirmed

Confirmed on a real 1280x1780 Japanese page, not by reasoning about the
algorithm.

The full-page raw detection run produced 58 boxes. Box 14 was
`(654,1458)-(888,1729)`, 234x271 px, against a page where the typical
fragment is roughly 30x30.

That region was then cropped out and re-run in isolation. The crop produced
11 boxes: 10 small ones tracking the actual characters, plus one at
`(15,12)-(248,286)` spanning the whole crop. The same run repeated with a
tighter `det_db_unclip_ratio` of 1.2 still produced 11 boxes with the same
oversized eleventh, which ruled out box expansion as the cause and
established that the detector genuinely emits a duplicate region rather
than an over-dilated version of a real one.

### The fix

`ocr.py`, `_drop_redundant_oversized_boxes`, called from
`_detect_and_read_japanese` before `_merge_vertical_fragments`.

A box is dropped only when both conditions hold:

1. its area is at least 8x the page's median box area
2. it contains at least 3 other independently detected boxes at a
   containment ratio of 0.9 or more

`_containment_ratio(inner, outer)` is the fraction of the inner box's own
area that falls inside the outer box, so it is not confused by a large box
overlapping a small one only partially.

Both conditions are required, and that is the substance of the fix. Area
alone would delete a legitimately large single box, such as one long
horizontal line of text. Containment alone would delete a legitimate
nested detection. Only the conjunction identifies a duplicate.

It runs before fragment merging so the merge step never sees a spurious
box competing with the real per-fragment detections it exists to
reassemble.

### Verification

First against the 11-box crop it was designed on:

```
input boxes: 11
kept: 10
dropped indices (from original list): [10]
  dropped box 10: (15, 12, 248, 286)
```

Then, separately, the shipped implementation was re-run against the full
58-box detection list from the real page, which is a harder test because
the median area is computed across the whole page rather than a single
dense cluster:

```
median box area on page: 1652   (drop threshold = 13216)
kept 57 of 58
  DROPPED [14] (654,1458,888,1729) area=63414 (38.4x median) contains=7
area outliers that were KEPT:
  kept [35] (1124,585,1187,932) area=21861 (13.2x median) contains=0
```

Box 35 is the useful result. It is a genuine area outlier at 13.2x the
median and the area test alone would have deleted it, but it contains no
other detections and is correctly kept. That is direct evidence on real
data that the two-signal design is not over-dropping, which a test on the
known-bad box alone could not have shown.

---

## BUG-2 PaddleOCR's `.ocr(rec=False)` wrapper raises on multi-box pages

**Status:** fixed. Found in the recovered source; the workaround was
already in place and is documented here because it is load-bearing and
looks like a mistake if you do not know why it is there.

### Symptom

Calling `model.ocr(image_path, rec=False)` to get detection-only output
raises `ValueError` on any page with more than one detected box.

### Root cause

PaddleOCR's wrapper performs `if not dt_boxes:` internally. `dt_boxes` is a
numpy array, and the truthiness of an array with more than one element is
ambiguous in numpy, so the check raises rather than returning.

### The fix

`ocr.py`, `_run_paddle_detection` calls `det_model.text_detector(img)`
directly, bypassing the wrapper, and does its own `is None or len(...) == 0`
check.

### Verification

Every Japanese detection run in the recovered notebook goes through this
path and returns box lists successfully, including the 58-box page. The
Japanese pipeline could not produce any output at all without it.

---

## BUG-3 CSS colour values corrupted during authoring

**Status:** fixed during this rebuild.

### Symptom

Two declarations in `frontend/style.css` were written with invalid values:
`background: #3a4considerable` on `button:disabled` and
`color: #6d7governmental` on `.region .meta`.

### Root cause

Authoring error, not a logic bug. Both are syntactically invalid colour
tokens.

### Why it mattered

A browser discards an invalid declaration and keeps the previous cascade
value, so this fails silently. `button:disabled` would have kept the
enabled button's background, making disabled Previous/Next controls in the
reader look clickable. That is exactly the kind of defect that survives to
a demo because nothing errors.

### The fix

`#3a4258` for the disabled button. The `.region .meta` rule was removed
outright: no element in `reader.html` carries that class, so the rule was
dead.

### Verification

Both the disabled state and the region panel render correctly in the
browser suite, and the reader screenshot in `tests/screenshots/reader.png`
shows a visually distinct greyed-out Previous button on page 1 of 3 while
Next remains active.

---

## BUG-4 Stub pipeline transitively required numpy

**Status:** fixed and verified in the environment that failed.

### Symptom

Running the server in stub mode with only `fastapi`, `uvicorn`,
`python-multipart` and `pillow` installed accepted an upload, queued it,
and then failed the page with:

```
ModuleNotFoundError: No module named 'numpy'
```

reported in the frontend's progress table against the uploaded file, at
0.16s.

### Root cause

`stub_pipeline.py` imported one helper:

```python
from pipeline_core import find_available_font
```

`pipeline_core.py` begins with `import numpy as np` at module scope.
Python has no way to import a single function without executing the whole
module, so importing that one helper pulled numpy into the stub's
dependency set.

The helper itself uses only `os` and `glob` and never touches numpy, which
is what made the dependency invisible on inspection.

### Why every existing test missed it

Both suites ran in an interpreter where numpy was already installed, as a
dependency of Pillow's test environment and of the pipeline itself. A
transitive import that resolves successfully is indistinguishable from no
import at all. The property under test is not "does it import" but "does
it import *without numpy available*", and nothing was asserting that.

This is also why it survived to a user-facing failure: 76 passing checks
covered every behaviour of the stub except the one reason the stub exists.

### The fix

`stub_pipeline.py` no longer imports `pipeline_core`. The font lookup is
duplicated locally as `_find_available_font`, with Windows font paths added
alongside the Linux ones, since the stub is the seam most likely to be run
on a developer's own machine.

The duplication is deliberate and is commented as such. The alternative,
moving `find_available_font` into a shared dependency-free module, would be
tidier in the abstract but changes a file recovered verbatim from the
notebook, which is a worse trade than sixteen duplicated lines.

Pillow is now the stub path's only third-party import.

### Verification

Reproduced first, in a clean virtualenv containing only the four packages
from the light install:

```
REPRODUCED: ModuleNotFoundError No module named 'numpy'
```

After the fix, in that same virtualenv:

- the stub imports, and `'numpy' in sys.modules` is `False` afterwards
- it renders a `.webp` input end to end, returning 4 regions and a valid
  900x1300 PNG
- the full server boots and a real multipart upload of a `.webp` reaches
  `status: done`, `error: None`, `region_count: 4`

### Regression test

`tests/test_api.py::check_stub_dependency_isolation` imports `runner` and
`stub_pipeline` in a subprocess with an import hook that raises on numpy,
cv2, torch and paddleocr. A subprocess is required: once numpy is loaded in
the parent interpreter the transitive import cannot be observed.

The test was confirmed to actually catch the bug by reverting the fix and
re-running:

```
[FAIL] stub path imports without numpy, cv2, torch or paddleocr
       --  ImportError: numpy must not be imported by the stub path
```

A regression test that has never been seen failing against the original
defect is not evidence of anything.

---

## BUG-5 PaddleOCR's entire class was replaced under this project, not just a version

**Status:** fixed and verified against the real library, installed at the
exact versions the Space actually resolves to.

### Symptom

On the deployed Space, every Japanese translation failed with:

```
AttributeError: 'PaddleOCR' object has no attribute 'text_detector'
```

### Root cause

UNV-8 (see `docs/LIMITATIONS.md`) had already established that
`paddlepaddle==2.6.1`, the version this project's code was written and
verified against, no longer exists on PyPI, and that `requirements.txt`
was changed to let `paddlepaddle` and `paddleocr` resolve unpinned as a
result. What actually resolves on the Space is `paddlepaddle==3.3.1` and
`paddleocr==3.7.0` (confirmed by reading the real build log). That is not
a compatible upgrade: `paddleocr` 3.x is built on a new dependency,
`paddlex`, and replaces the entire class this pipeline was written
against.

`ocr.py._run_paddle_detection` called `det_model.text_detector(img)`
directly -- a documented workaround (BUG-2) for a bug in the old
`.ocr(rec=False)` wrapper. That internal attribute does not exist on
3.7.0. Confirmed directly: `dir(PaddleOCR)` on the installed 3.7.0 package
lists only `close, export_paddlex_config_to_yaml,
get_cli_subcommand_executor, ocr, predict, predict_iter` -- no
`text_detector`, and no detector-only method of any name. The nested
`paddlex_pipeline` object was also inspected directly and exposes nothing
detector-shaped either.

A second, undiscovered break existed in the same commit:
`ocr.py.detect_and_read_text` (the Chinese path) calls
`model.ocr(image_path, cls=True)` and parses the result as
`[coords, (text, confidence)]` tuples, the 2.x output shape. Calling
`.ocr()` on 3.7.0 prints `DeprecationWarning: Please use predict instead`
and returns the same dict-based `OCRResult` object `.predict()` does, not
the old tuple format. This was never reported because the user had only
tested the Japanese path; it was found by testing the Chinese path
directly against the real library before assuming it still worked.

### How the fix was designed

The installed constructor was inspected directly rather than guessed at:
`inspect.signature(PaddleOCR.__init__)` was compared against every keyword
argument `ocr.py` was passing. Every one of `use_angle_cls`,
`det_limit_side_len`, `det_limit_type`, `det_db_thresh`,
`det_db_box_thresh`, `det_db_unclip_ratio` was absent from the new
signature. The replacements were found in the same signature: a systematic
`text_det_` prefix replaces `det_db_`/`det_` (`text_det_thresh`,
`text_det_box_thresh`, `text_det_unclip_ratio`, `text_det_limit_side_len`,
`text_det_limit_type`), and `use_angle_cls` is replaced by
`use_textline_orientation`. Two new stages that did not exist in 2.x,
`use_doc_orientation_classify` and `use_doc_unwarping`, were explicitly
set to `False` rather than left at their defaults, since enabling
undocumented new pipeline stages this project never verified would be a
silent behaviour change, not a version bump.

A second, unrelated failure surfaced while probing the new constructor: a
call to `.predict()` raised `NotImplementedError:
(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]` from paddle's own oneDNN
backend. This was isolated, not assumed: the same image was run twice,
once with `enable_mkldnn` at its default and once with
`enable_mkldnn=False`, with every other variable held constant. Only the
flag changed the outcome, confirming a real backend bug independent of
image content, so `enable_mkldnn=False` is now part of the model
constructor for both languages.

With the correct constructor established, `.predict()` was found to
return one dict-like result per image carrying `dt_polys` (detection
polygons), `rec_texts`, `rec_scores`, and precomputed axis-aligned
`rec_boxes` -- detection and recognition together, in one call, with no
separate detection-only entry point. This makes BUG-2's original
workaround unnecessary as well as impossible: the numpy-truthiness bug
that workaround existed for was specific to the old `.ocr(rec=False)`
code path, and `.predict()` does not share it.

The fix: `_run_paddle_predict` replaces `_run_paddle_detection`, calling
`.predict()` instead of the removed internal attribute.
`_coords_from_dt_polys` converts `dt_polys`'s numpy arrays to the plain
`[[x, y], ...]` list format the rest of the file already expects. Both
`detect_and_read_text` (Chinese) and `_detect_and_read_japanese` were
rewritten to call `_run_paddle_predict`, but consume different fields:
Chinese uses `dt_polys` together with `rec_texts`/`rec_scores` from
PaddleOCR's own recognizer, unchanged from the original design. Japanese
uses only `dt_polys` and discards `rec_texts`/`rec_scores`, since
recognition still comes from manga-ocr -- this is less efficient than the
old detection-only call, since PaddleOCR's own recognizer now runs and is
thrown away, and is recorded as a limitation rather than silently
accepted.

Critically, `_drop_redundant_oversized_boxes` (BUG-1) and
`_merge_vertical_fragments` were **not modified**. Both operate purely on
the plain coordinate-list format, independent of which PaddleOCR version
produced it, so the fix was scoped to exactly the two functions that
touched the removed API.

### Verification

Run directly against the real installed `paddleocr==3.7.0` /
`paddlepaddle==3.3.1` -- the exact versions confirmed from the Space's
build log, not a version chosen for convenience:

- **Chinese path, full and unmodified call**: `detect_and_read_text('...',
  source_lang='zh')` against a real rendered-text image returned one box,
  `text='Hello World Test'`, `confidence=0.9998`, correctly shaped
  coordinates. No crash, no deprecation-path surprises.
- **Japanese path, through BUG-1 and the merge step**: a three-region
  synthetic image produced 3 raw detected boxes, all 3 survived
  `_drop_redundant_oversized_boxes` unchanged (correctly -- none were
  duplicates), and `_merge_vertical_fragments` correctly kept them as 3
  separate regions rather than merging spatially distant text. This
  confirms BUG-1's fix continues to function correctly fed from the new
  detection source, not just that it still compiles.
- **The remaining gap was confirmed, not assumed**: calling
  `get_manga_ocr_model()` in this environment fails with exactly
  `ModuleNotFoundError: No module named 'manga_ocr'` and nothing else --
  proving the Japanese path's detection and merge logic run correctly up
  to the one dependency genuinely not installed here, rather than failing
  earlier for an unrelated reason.
- All 90 existing stub-mode checks (44 API, 33 browser, 13 Gradio) were
  re-run after this change and still pass, confirming nothing outside
  `ocr.py` was disturbed.

### What is still not verified

manga-ocr's own recognition call was not exercised, since manga-ocr and
torch are not installed in the environment this fix was written in. The
build log shows the Space resolves `manga-ocr==0.1.16`, not the
notebook-verified `0.1.14`; whether its recognition behaves the same way
is unconfirmed. See the updated UNV-8 in `docs/LIMITATIONS.md`.

---

## BUG-6 `is_garbage_text` let short OCR misreads through unflagged

**Status:** fixed for the two reported failure shapes, with two accepted
tradeoffs documented rather than silently shipped.

### Symptom

Reported by the separate evaluation-harness project run against this
pipeline: short manga-ocr misreads on the Japanese path, for example `CSB`
and `d00`, were passed on to translation instead of being caught as
garbage first.

### Root cause

`is_garbage_text` (`backend/pipeline_core.py`) had two checks, both gated
by a minimum length: a repeated-character-dominance check active only at
`len(normalized) >= 8`, and a digit-majority check active only at
`len(normalized) >= 6`. Nothing examined strings shorter than 6 characters
at all. Confirmed directly by extracting and running the function before
making any change: `is_garbage_text("CSB")` and `is_garbage_text("d00")`
both returned `False`.

### The fix

Two narrower checks were added, active only at `len(stripped) <= 5`, using
the original (not O-to-0-normalized) text -- the existing normalization
exists for the digit-majority check and would misfire here, for example
turning `"OK"` into `"0K"`:

- a short token mixing letters with digits, where digits are not a small
  minority (`len(digit_chars) >= len(letters)`), is flagged. This catches
  `d00` and the same shape with other characters (`3d0`, `9x2`).
- a short (3+ character) all-ASCII, all-consonant token with no digits is
  flagged. This catches `CSB` and `XKQ`.

Both checks were designed against, and verified against, the two reported
strings plus a set of common short real words and abbreviations the fix
must not flag (`OK`, `Hi`, `TV`, `Mr`, `SOS`, `ID`, and 20 more) to confirm
the fix is not simply over-broad.

### Two accepted tradeoffs, not fixed

Both were found by testing, not predicted from reading the code, and both
are left as-is rather than hidden:

- **`3D` (and similarly shaped real alphanumeric tokens) is a false
  positive.** The letter/digit-mix rule cannot distinguish a real short
  alphanumeric token from a misread using digit-vs-letter ratio alone at
  this length. Making the rule stricter (for example requiring 2+ digits)
  would let `d00`-shaped misreads with only one digit back through, which
  is the more common real failure shape reported. The tradeoff was chosen
  deliberately in that direction.
- **A digit-minority letter/digit mix, for example `O0O`, is not caught.**
  This is the direct inverse of the `3D` case: the same rule that would
  catch it would also flag `3D`-shaped real content, so it is left
  unflagged rather than trading one false positive for another.
- **The all-consonant rule will also flag genuine all-consonant short
  tokens**, most plausibly manga sound effects (`SHH`, `TSK`, `GRR`).
  Whether this actually collides with real sound-effect text has not been
  tested, since no real sample of manga sound-effect OCR output was
  available while writing this fix. `SOS` was checked and does not
  collide, because `O` counts as a vowel here; three-letter effects
  without any of `AEIOU` are the ones at risk.

### Verification

`tests/test_pipeline_core.py`, 40 checks, all passing: the two reported
strings and two same-shape variants are flagged; 27 common short real
words and abbreviations are not flagged; both documented tradeoffs above
are asserted as-is (`3D` flagged, `O0O` not flagged) so a future change
to either rule shows up as a deliberate test change, not a silent
behaviour shift; the pre-existing longer-string checks (repeated-character
at length >= 8, digit-majority at length >= 6) and the original edge cases
(empty string, whitespace, over-length string) were re-run unchanged and
still pass, confirming the fix only added a new branch and did not modify
existing logic.

Confirmed by reading the import graph (`grep` across `stub_pipeline.py`,
`runner.py`, `main.py`, `batch_store.py`) that nothing in the stub-mode
path imports `pipeline_core`, so the existing 90 stub-mode checks could
not have been affected by this change and were not re-run for that reason.

Not verified: this function's behaviour against real manga-ocr output.
Everything above was checked against a hand-picked set of strings, not
against real recognition results from actual pages, since manga-ocr and
torch remain uninstalled in this environment. The evaluation-harness
project that reported this gap is the intended source of that evidence.
