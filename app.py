import html
import os
import sys
import time

import gradio as gr
import spaces
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import runner
from batch_store import STATUS_DONE, STATUS_FAILED, store

LANG_CHOICES = [("Japanese", "ja"), ("Chinese", "zh")]
POLL_INTERVAL_SECONDS = 0.5

# rendered pages are full-resolution PNGs (1-2 MB, per docs/DECISIONS.md).
# gr.Gallery has no separate thumbnail/full-size pair per item (confirmed
# by reading gradio.data_classes.ImageData -- one path per item, used for
# both the grid and the click-to-zoom preview), so this one size has to
# serve both. 1100px keeps translated dialogue legible when a user clicks
# to zoom, while still being a real cut from a 1-2 MB original: this is
# the dominant fix for the reported lag, since gr.Gallery has no
# incremental-update path and rebuilds the whole grid client-side on every
# completed page, which gets expensive fast at full resolution past
# roughly ten pages.
THUMBNAIL_MAX_EDGE = 1100

# region table rows are appended forever otherwise, and gr.HTML has no
# incremental update path either -- the whole string is re-sent and
# re-parsed on every yield. capping to the most recently completed pages
# keeps that string bounded regardless of batch size; the full data for
# every page is still in the downloadable zip once the batch finishes.
REGION_TABLE_MAX_PAGES = 5

def _thumbnail_path(image_path):
    # the thumbnail path is deterministic from the source path, so
    # existence on disk is the cache -- no separate in-memory dict needed.
    # an in-memory cache keyed by image_path would grow for the whole life
    # of the process (batches get deleted, on TTL or explicitly, but a
    # dict entry pointing at a since-deleted path never would), which is
    # exactly the kind of slow leak worth avoiding on a long-running Space.
    # a completed page's rendered image never changes after translation,
    # so the file-exists check below is sufficient reuse: on every
    # subsequent poll tick this is one stat() call, not a re-resize.
    thumb_path = image_path + f".thumb{THUMBNAIL_MAX_EDGE}.jpg"
    if os.path.exists(thumb_path):
        return thumb_path

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
            img.save(thumb_path, "JPEG", quality=80)
        return thumb_path
    except Exception:
        # a thumbnail failure must not break the gallery -- fall back to
        # the original image rather than showing nothing for that page
        return image_path

# Entry point for the Hugging Face Gradio Space.
#
# The FastAPI service in backend/main.py is unchanged and is still the way
# this runs locally and the way the HTTP API is exercised by the test
# suites. This module is a second front end onto the same machinery, not a
# replacement for it: it drives batch_store and runner directly, so the job
# queue, the single-worker invariant and the result storage are identical
# under both.
#
# Reusing batch_store rather than calling runner.run_page in a loop matters
# on a public Space specifically. Several visitors can submit at the same
# moment, and the OCR model instances in ocr.py are process-global and not
# thread-safe. The store's single worker thread serialises them. Gradio's
# own concurrency limit is set to 1 as well, which is belt and braces: the
# limit bounds how many requests are in flight, the worker bounds how many
# translations actually touch the models.
#
# _zerogpu_probe exists for a platform requirement, not for acceleration.
# A real deploy failed at the platform level with "No @spaces.GPU function
# detected during startup" -- the app itself had started and bound to its
# port correctly, but ZeroGPU Spaces refuse to serve traffic unless at
# least one function decorated with @spaces.GPU exists anywhere in the
# app. Nothing in this pipeline otherwise needs or requests a real GPU:
# paddlepaddle is the CPU build, and no other code path is routed through
# this decorator. Calling it once at startup, rather than only defining
# it, also answers a question that was previously open (see UNV-7 in
# docs/LIMITATIONS.md): whether torch reports a CUDA device as available
# under ZeroGPU's CUDA emulation outside a decorated call. The result is
# logged so that answer is visible in the container logs without having
# to add diagnostic code later.


@spaces.GPU
def _zerogpu_probe():
    import torch
    return torch.cuda.is_available()


def _describe_failure(batch):
    failures = [p for p in batch.pages if p.status == STATUS_FAILED]
    if not failures:
        return ""
    lines = ["", "Failed pages:"]
    for page in failures:
        lines.append(f"- {page.filename}: {page.error}")
    return "\n".join(lines)


def _gallery_items(batch):
    items = []
    for page in batch.pages:
        if page.status != STATUS_DONE or not page.image_token:
            continue
        path = store.image_path(page.image_token)
        if path and os.path.exists(path):
            items.append((_thumbnail_path(path), f"{page.index + 1}. {page.filename}"))
    return items


def _region_table(batch):
    # rendered as HTML rather than gr.Dataframe on purpose. Dataframe
    # postprocesses through pandas, which in turn requires jinja2 >= 3.1.2;
    # that combination failed to import during local verification. a table
    # of four string columns does not justify a pandas dependency on a
    # Space whose build has not been proven yet.
    #
    # capped to the most recently completed REGION_TABLE_MAX_PAGES pages,
    # not all of them. gr.HTML has no incremental-update path -- the whole
    # string is re-sent and re-parsed by the browser on every yield -- so
    # an uncapped table grows without bound over a long batch and was part
    # of the reported lag. every page's regions are still in the
    # downloadable zip once the batch finishes; this table is a live
    # preview, not the only place the data exists.
    completed_pages = [p for p in batch.pages if p.status == STATUS_DONE]
    shown_pages = completed_pages[-REGION_TABLE_MAX_PAGES:]
    omitted = len(completed_pages) - len(shown_pages)

    rows = []
    for page in shown_pages:
        for region in page.regions:
            rows.append((
                page.index + 1,
                region.get("source_text", ""),
                region.get("translation", ""),
                str(region.get("bounds", "")),
            ))

    if not rows:
        return "<p>No regions yet.</p>"

    cells = "".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(source)}</td>"
        f"<td>{html.escape(translation)}</td>"
        f"<td><code>{html.escape(bounds)}</code></td>"
        "</tr>"
        for index, source, translation, bounds in rows
    )
    note = (
        f"<p style='color:#888;font-size:0.85em'>Showing the {len(shown_pages)} most "
        f"recently completed page(s); {omitted} earlier page(s) omitted from this "
        "live preview. Every page's regions are included in the downloadable zip.</p>"
        if omitted > 0 else ""
    )
    return (
        note +
        "<table style='width:100%;border-collapse:collapse' id='regions-table'>"
        "<thead><tr>"
        "<th style='text-align:left'>Page</th>"
        "<th style='text-align:left'>Source</th>"
        "<th style='text-align:left'>Translation</th>"
        "<th style='text-align:left'>Bounds</th>"
        "</tr></thead>"
        f"<tbody>{cells}</tbody></table>"
    )


def translate(files, source_lang):
    if not files:
        raise gr.Error("Select at least one page image first.")

    payloads = []
    for item in files:
        path = item if isinstance(item, str) else item.name
        with open(path, "rb") as handle:
            payloads.append((os.path.basename(path), handle.read()))

    try:
        batch = store.create_batch(payloads, source_lang)
    except ValueError as exc:
        raise gr.Error(str(exc))

    # generator: gradio re-renders the outputs on every yield, so progress
    # is driven by the same per-page status the HTTP API exposes rather
    # than by a separate progress mechanism that could disagree with it.
    #
    # the status text is cheap (a short string) and updates every tick for
    # live feedback. the gallery and region table are not cheap: gr.Gallery
    # and gr.HTML both have no incremental-update path, so every yield with
    # a real value rebuilds the whole component client-side and, in the
    # gallery's case, re-requests every image in it. rebuilding those on
    # every 0.5s tick regardless of whether a new page had actually finished
    # was the reported lag -- most ticks find nothing new to show. gr.update()
    # with no value is a genuine no-op on the frontend, unlike re-sending an
    # unchanged list, so only building fresh values when finished actually
    # increases is what removes the wasted rebuilds.
    last_finished = -1
    while True:
        snapshot = batch.to_dict()
        finished = snapshot["completed"] + snapshot["failed"]
        total = snapshot["total"]
        status = (
            f"{finished} / {total} pages processed"
            + (f", {snapshot['failed']} failed" if snapshot["failed"] else "")
            + f" -- batch {snapshot['status']}"
            + _describe_failure(batch)
        )

        if finished != last_finished:
            gallery_update = _gallery_items(batch)
            table_update = _region_table(batch)
            last_finished = finished
        else:
            gallery_update = gr.update()
            table_update = gr.update()

        yield status, gallery_update, table_update, gr.update()

        if snapshot["status"] == STATUS_DONE:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    zip_path = None
    buffer = store.build_zip(batch.id)
    if buffer is not None:
        zip_path = os.path.join(batch.directory, f"translated_{batch.id[:8]}.zip")
        with open(zip_path, "wb") as handle:
            handle.write(buffer.read())

    final = batch.to_dict()
    summary = (
        f"Done. {final['completed']} of {final['total']} pages translated"
        + (f", {final['failed']} failed" if final["failed"] else "")
        + _describe_failure(batch)
    )
    yield summary, _gallery_items(batch), _region_table(batch), gr.update(value=zip_path, visible=zip_path is not None)


def _startup_notice():
    mode = runner.pipeline_mode()
    if mode == "stub":
        return (
            "Backend is running the STUB pipeline. No detection, recognition or "
            "translation is performed and every output image is stamped as such."
        )
    if not os.environ.get("NVIDIA_API_KEY"):
        return (
            "NVIDIA_API_KEY is not configured. Detection will run but every "
            "translation call will fail."
        )
    return (
        "Ready. The first page after a cold start is slow: OCR model weights "
        "are downloaded and loaded on first use."
    )


with gr.Blocks(title="Manga Translator") as demo:
    gr.Markdown(
        "# Manga Translator\n"
        "PaddleOCR detection, manga-ocr recognition for Japanese, "
        "Nemotron via NVIDIA NIM for translation."
    )
    gr.Markdown(_startup_notice())

    with gr.Row():
        files_input = gr.File(
            label="Page images",
            file_count="multiple",
            file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            elem_id="files-input",
        )
        lang_input = gr.Radio(
            choices=LANG_CHOICES,
            value="ja",
            label="Source language",
            elem_id="lang-input",
        )

    translate_button = gr.Button("Translate", variant="primary", elem_id="translate-button")
    status_output = gr.Textbox(label="Status", lines=3, elem_id="status-output", interactive=False)

    gallery_output = gr.Gallery(
        label="Translated pages",
        columns=2,
        height=620,
        preview=True,
        elem_id="gallery-output",
    )

    gr.Markdown("### Detected regions")
    regions_output = gr.HTML(value="<p>No regions yet.</p>", elem_id="regions-output")

    zip_output = gr.File(label="Download all pages", visible=False, elem_id="zip-output")

    translate_button.click(
        fn=translate,
        inputs=[files_input, lang_input],
        outputs=[status_output, gallery_output, regions_output, zip_output],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    # best-effort: torch is not installed in stub mode (see runner.py and
    # requirements.txt's light local-dev install), and the platform's
    # requirement is satisfied by the decorated function existing in the
    # module, not by this call succeeding. A failure here must never block
    # startup.
    try:
        print(f"ZeroGPU probe: torch.cuda.is_available() = {_zerogpu_probe()}")
    except Exception as exc:
        print(f"ZeroGPU probe skipped: {type(exc).__name__}: {exc}")
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
