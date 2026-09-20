import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

API_PORT = 8012
STATIC_PORT = 8013
API_BASE = f"http://127.0.0.1:{API_PORT}"
STATIC_BASE = f"http://127.0.0.1:{STATIC_PORT}"

SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SHOT_DIR, exist_ok=True)

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def wait_for(url, timeout=45):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def make_page(path, label):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (800, 1150), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 30, 770, 1120], outline=(170, 170, 170), width=3)
    draw.text((50, 50), label, fill=(0, 0, 0))
    image.save(path)
    return path


def start_backend(data_dir):
    env = dict(os.environ)
    env["USE_STUB_PIPELINE"] = "1"
    env["MANGA_TRANSLATOR_DATA_DIR"] = data_dir
    # the static server below is a genuinely different origin, so the browser
    # will run real CORS preflights against the API. that is the only way to
    # exercise the middleware the way a browser actually does.
    env["CORS_ALLOW_ORIGINS"] = STATIC_BASE
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=BACKEND_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def start_static():
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(STATIC_PORT), "--bind", "127.0.0.1"],
        cwd=FRONTEND_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run(page, pages, console_errors, failed_responses):
    page.goto(f"{STATIC_BASE}/index.html?api={API_BASE}", wait_until="networkidle")

    check("index.html renders its title", page.title() == "Manga Translator", page.title())

    page.wait_for_function(
        "document.getElementById('health-summary').textContent !== 'checking backend...'",
        timeout=15000,
    )
    health_text = page.inner_text("#health-summary")
    check("cross-origin health call succeeded", "backend ok" in health_text, health_text)
    check("stub mode is surfaced to the user", "stub" in health_text, health_text)

    banner_text = page.inner_text("#banner")
    check("stub warning banner is shown", "stub pipeline" in banner_text.lower(), banner_text)

    page.set_input_files("#files", pages)
    check("file count is reflected in the ui",
          "3 files selected" in page.inner_text("#file-summary"),
          page.inner_text("#file-summary"))

    check("progress card is hidden before submit", page.is_hidden("#progress-card"))

    page.click("#submit")
    page.wait_for_selector("#progress-card:not(.hidden)", timeout=15000)
    check("progress card appears after submit", page.is_visible("#progress-card"))

    page.wait_for_function("document.querySelectorAll('#page-table tbody tr').length === 3", timeout=15000)
    check("progress table lists every page",
          page.eval_on_selector_all("#page-table tbody tr", "rows => rows.length") == 3)

    # the batch must actually pass through a non-terminal state, otherwise
    # the asynchronous design is not being exercised at all
    observed_running = page.evaluate("""
      () => new Promise(resolve => {
        let seen = false;
        const timer = setInterval(() => {
          const text = document.getElementById('progress-text').textContent;
          if (/running/.test(text) || document.querySelector('.pill.running')) seen = true;
          if (/batch done/.test(text)) { clearInterval(timer); resolve(seen); }
        }, 120);
        setTimeout(() => { clearInterval(timer); resolve(seen); }, 90000);
      })
    """)
    check("batch was observed in a running state before completing", observed_running)

    page.wait_for_function(
        "/batch done/.test(document.getElementById('progress-text').textContent)", timeout=90000
    )
    progress_text = page.inner_text("#progress-text")
    check("all pages processed", "3 / 3 pages processed" in progress_text, progress_text)
    check("no failures reported", "failed" not in progress_text, progress_text)

    done_pills = page.eval_on_selector_all(".pill.done", "els => els.length")
    check("every row shows a done pill", done_pills == 3, str(done_pills))

    bar_width = page.eval_on_selector("#bar-fill", "el => el.style.width")
    check("progress bar reached 100%", bar_width == "100%", bar_width)

    page.screenshot(path=os.path.join(SHOT_DIR, "upload.png"), full_page=True)

    check("open reader is enabled", page.is_enabled("#open-reader"))
    check("download is enabled", page.is_enabled("#download"))

    page.click("#open-reader")
    page.wait_for_url("**/reader.html*", timeout=15000)
    check("reader url carries the api override", "api=" in page.url, page.url)

    page.wait_for_selector("#stage img", timeout=20000)
    check("reader loaded a page image", page.is_visible("#stage img"))

    # naturalWidth is the only assertion that proves the browser actually
    # fetched and decoded the image, rather than just inserting a tag
    natural_width = page.eval_on_selector("#stage img", "img => img.naturalWidth")
    check("page image decoded in the browser", natural_width > 0, f"naturalWidth={natural_width}")

    check("counter shows first of three", page.inner_text("#counter").strip() == "1 / 3",
          page.inner_text("#counter"))
    check("previous is disabled on the first page", page.is_disabled("#prev"))

    page.screenshot(path=os.path.join(SHOT_DIR, "reader.png"), full_page=True)

    region_count = page.eval_on_selector_all("#regions .region", "els => els.length")
    check("regions panel is populated", region_count > 0, str(region_count))
    first_region = page.inner_text("#regions .region")
    check("region shows source text", "source" in first_region, first_region.replace("\n", " ")[:80])
    check("region shows a bounds tuple", "[" in first_region and "]" in first_region)

    first_src = page.eval_on_selector("#stage img", "img => img.src")
    page.click("#next")
    page.wait_for_function(
        "document.getElementById('counter').textContent.trim() === '2 / 3'", timeout=20000
    )
    check("next advances the counter", page.inner_text("#counter").strip() == "2 / 3")
    page.wait_for_function(
        f"document.querySelector('#stage img') && document.querySelector('#stage img').src !== {first_src!r}",
        timeout=20000,
    )
    check("next loads a different image",
          page.eval_on_selector("#stage img", "img => img.src") != first_src)

    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "document.getElementById('counter').textContent.trim() === '3 / 3'", timeout=20000
    )
    check("arrow key navigation works", page.inner_text("#counter").strip() == "3 / 3")
    check("next is disabled on the last page", page.is_disabled("#next"))

    page.keyboard.press("ArrowLeft")
    page.wait_for_function(
        "document.getElementById('counter').textContent.trim() === '2 / 3'", timeout=20000
    )
    check("arrow key back navigation works", page.inner_text("#counter").strip() == "2 / 3")

    # reader modes: single (default, already exercised above), long strip,
    # and fullscreen. reload the reader on the same batch id directly rather
    # than going through "Back to upload" -- that resets index.html's own
    # in-memory batch state and disables #open-reader again, which is
    # index.html's behaviour to test separately, not something this reader
    # mode check should depend on.
    reader_url = page.url
    page.goto(reader_url, wait_until="networkidle")
    page.wait_for_selector("#stage img", timeout=20000)

    thumb_count = page.eval_on_selector_all(".thumb-rail .thumb", "els => els.length")
    check("thumbnail rail shows one thumbnail per readable page", thumb_count == 3, str(thumb_count))
    check("thumbnail rail is visible in single mode", page.is_visible("#thumb-rail"))

    page.click('.thumb-rail .thumb:nth-child(3)')
    page.wait_for_function(
        "document.getElementById('counter').textContent.trim() === '3 / 3'", timeout=20000
    )
    check("clicking a thumbnail jumps to that page", page.inner_text("#counter").strip() == "3 / 3")

    page.click('button[data-mode="strip"]')
    page.wait_for_selector(".strip-page", timeout=10000)
    check("strip mode hides the single-page view", page.is_hidden("#single-view"))
    check("strip mode shows the strip view", page.is_visible("#strip-view"))
    strip_page_count = page.eval_on_selector_all(".strip-page", "els => els.length")
    check("strip mode renders one container per readable page", strip_page_count == 3, str(strip_page_count))
    page.wait_for_function("document.querySelectorAll('.strip-page img').length > 0", timeout=10000)
    strip_img_count = page.eval_on_selector_all(".strip-page img", "els => els.length")
    check("strip mode lazy-loads at least the first page's image", strip_img_count > 0, str(strip_img_count))

    page.click('button[data-mode="fullscreen"]')
    page.wait_for_selector("#fullscreen-view:not(.hidden)", timeout=10000)
    check("fullscreen mode shows the overlay", page.is_visible("#fullscreen-view"))
    check("fullscreen mode hides the thumbnail rail", page.is_hidden("#thumb-rail"))
    fs_src = page.eval_on_selector("#fs-image", "img => img.src")
    check("fullscreen mode loads an image", bool(fs_src), fs_src)

    # this click failing to register (timing out) is exactly the symptom of
    # the fixed bug: the right click-zone used to sit above this button and
    # swallow the click. reaching the check below is the real assertion.
    page.click("#fs-exit")
    page.wait_for_function(
        "document.getElementById('fullscreen-view').classList.contains('hidden')", timeout=10000
    )
    check("exiting fullscreen returns to single-page mode", page.is_visible("#single-view"))

    page.click("#back")
    page.wait_for_url("**/index.html*", timeout=15000)
    check("back returns to the upload page", page.url.endswith("index.html") or "index.html" in page.url)

    # error path: reader with no batch parameter
    page.goto(f"{STATIC_BASE}/reader.html?api={API_BASE}", wait_until="networkidle")
    page.wait_for_selector("#banner:not(.hidden)", timeout=10000)
    check("reader without a batch shows an error",
          "No batch specified" in page.inner_text("#banner"), page.inner_text("#banner"))

    # error path: reader with an unknown batch id
    page.goto(f"{STATIC_BASE}/reader.html?batch=doesnotexist&api={API_BASE}", wait_until="networkidle")
    page.wait_for_selector("#banner:not(.hidden)", timeout=10000)
    check("reader with an unknown batch surfaces the api error",
          "not found" in page.inner_text("#banner").lower(), page.inner_text("#banner"))

    # Deliberate 404 probes above make the browser log console errors, so the
    # assertion is on failed responses with their URLs rather than on console
    # text. Anything failing outside the two endpoints this test intentionally
    # asks for a missing batch on is a real defect.
    expected_404_substrings = ("/api/batch/doesnotexist", "favicon.ico")
    unexpected = [
        (url, status) for url, status in failed_responses
        if not any(fragment in url for fragment in expected_404_substrings)
    ]
    check("no unexpected failed requests during the run",
          len(unexpected) == 0, "; ".join(f"{s} {u}" for u, s in unexpected[:5]))

    page_errors = [e for e in console_errors if not e.startswith("Failed to load resource")]
    check("no uncaught javascript exceptions", len(page_errors) == 0, "; ".join(page_errors[:3]))


def main():
    tmp_dir = tempfile.mkdtemp(prefix="mt-browsertest-")
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pages = [make_page(os.path.join(tmp_dir, f"page{i}.png"), f"browser test page {i}") for i in (1, 2, 3)]

    backend = start_backend(data_dir)
    static = start_static()
    try:
        if not wait_for(f"{API_BASE}/api/health"):
            raise RuntimeError("backend did not start")
        if not wait_for(f"{STATIC_BASE}/index.html"):
            raise RuntimeError("static server did not start")

        console_errors = []
        failed_responses = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))
            page.on("response", lambda r: failed_responses.append((r.url, r.status)) if r.status >= 400 else None)
            try:
                run(page, pages, console_errors, failed_responses)
                page.screenshot(path=os.path.join(tmp_dir, "final.png"), full_page=True)
                print(f"\nscreenshot: {os.path.join(tmp_dir, 'final.png')}")
            finally:
                context.close()
                browser.close()
    finally:
        for process in (backend, static):
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
