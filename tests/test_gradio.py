import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8014
BASE = f"http://127.0.0.1:{PORT}"

SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SHOT_DIR, exist_ok=True)

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def wait_for(url, timeout=90):
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


def start_app(data_dir):
    env = dict(os.environ)
    env["USE_STUB_PIPELINE"] = "1"
    env["MANGA_TRANSLATOR_DATA_DIR"] = data_dir
    env["PORT"] = str(PORT)
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    return subprocess.Popen(
        [sys.executable, "app.py"], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def run(page, pages, console_errors):
    page.goto(BASE, wait_until="networkidle")

    check("gradio app serves a page", "Manga Translator" in page.content())

    page.wait_for_selector("#translate-button", timeout=30000)
    check("translate button rendered", page.is_visible("#translate-button"))
    check("language selector rendered", page.is_visible("#lang-input"))
    check("stub notice is shown to the user",
          "STUB" in page.inner_text("body"), "stub warning present")

    # gradio's File component hides the real input; set files on it directly
    file_input = page.locator("#files-input input[type='file']")
    file_input.set_input_files(pages)
    page.wait_for_timeout(1500)

    page.click("#translate-button")

    # the generator must publish at least one intermediate state before the
    # final one, otherwise progress reporting is not actually working
    saw_progress = page.evaluate("""
      () => new Promise(resolve => {
        let seen = false;
        const timer = setInterval(() => {
          const box = document.querySelector('#status-output textarea, #status-output input');
          const text = box ? box.value : '';
          if (/pages processed/.test(text) && !/^Done\\./.test(text)) seen = true;
          if (/^Done\\./.test(text)) { clearInterval(timer); resolve(seen); }
        }, 100);
        setTimeout(() => { clearInterval(timer); resolve(seen); }, 120000);
      })
    """)
    check("intermediate progress was published before completion", saw_progress)

    page.wait_for_function(
        "(() => { const b = document.querySelector('#status-output textarea, #status-output input');"
        " return b && /^Done\\./.test(b.value); })()",
        timeout=120000,
    )
    status = page.eval_on_selector("#status-output textarea, #status-output input", "el => el.value")
    check("all three pages translated", "3 of 3 pages translated" in status, status.split("\n")[0])
    check("no failures reported", "failed" not in status, status.split("\n")[0])

    page.wait_for_selector("#gallery-output img", timeout=30000)
    images = page.eval_on_selector_all("#gallery-output img", "els => els.length")
    check("gallery rendered images", images > 0, str(images))

    decoded = page.eval_on_selector_all(
        "#gallery-output img", "els => els.filter(i => i.naturalWidth > 0).length"
    )
    check("at least one gallery image decoded in the browser", decoded > 0, str(decoded))

    page.screenshot(path=os.path.join(SHOT_DIR, "gradio.png"), full_page=True)

    table_text = page.inner_text("#regions-output")
    check("region table is populated", "stub region" in table_text,
          table_text.replace("\n", " ")[:70])
    check("region table shows the source column", "source" in table_text.lower())

    check("zip download became visible", page.is_visible("#zip-output"))

    page_errors = [e for e in console_errors if "favicon" not in e.lower()]
    check("no uncaught javascript exceptions", len(page_errors) == 0, "; ".join(page_errors[:3]))


def run_lag_fix_checks(page, pages, console_errors):
    # separate batch, larger than REGION_TABLE_MAX_PAGES (5), to prove two
    # real things rather than just the presence of code: the gallery serves
    # downscaled thumbnails instead of the full-resolution rendered PNGs,
    # and the region table stays capped rather than growing with every page.
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector("#translate-button", timeout=30000)

    file_input = page.locator("#files-input input[type='file']")
    file_input.set_input_files(pages)
    page.wait_for_timeout(1500)
    page.click("#translate-button")

    page.wait_for_function(
        "(() => { const b = document.querySelector('#status-output textarea, #status-output input');"
        f" return b && /of {len(pages)} pages translated/.test(b.value); }})()",
        timeout=180000,
    )

    page.wait_for_selector("#gallery-output img", timeout=30000)
    page.wait_for_timeout(500)

    gallery_src = page.eval_on_selector("#gallery-output img", "el => el.src")
    check("gallery image url points at a thumbnail file, not the original",
          ".thumb" in gallery_src, gallery_src)

    natural_width = page.eval_on_selector("#gallery-output img", "el => el.naturalWidth")
    check("gallery thumbnail is downscaled below the 1100px cap",
          0 < natural_width <= 1100, str(natural_width))

    table_text = page.inner_text("#regions-output")
    check(f"region table caps at 5 pages for a {len(pages)}-page batch",
          "5 most recently completed" in table_text, table_text[:120])
    check("region table names how many earlier pages were omitted",
          f"{len(pages) - 5} earlier page(s) omitted" in table_text, table_text[:200])

    page_errors = [e for e in console_errors if "favicon" not in e.lower()]
    check("no uncaught javascript exceptions during the larger batch",
          len(page_errors) == 0, "; ".join(page_errors[:3]))


def main():
    tmp_dir = tempfile.mkdtemp(prefix="mt-gradio-")
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pages = [make_page(os.path.join(tmp_dir, f"page{i}.png"), f"gradio test page {i}") for i in (1, 2, 3)]
    # 12 pages: past the reported "lag starts around 10 images" threshold,
    # and past REGION_TABLE_MAX_PAGES (5), so the cap actually gets exercised
    larger_batch = [
        make_page(os.path.join(tmp_dir, f"lagtest{i}.png"), f"lag test page {i}") for i in range(1, 13)
    ]

    process = start_app(data_dir)
    try:
        if not wait_for(BASE):
            output = ""
            if process.poll() is not None:
                output = process.stdout.read().decode(errors="replace")[-2000:]
            raise RuntimeError("gradio app did not start\n" + output)

        console_errors = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1280, "height": 1000})
            page = context.new_page()
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))
            try:
                run(page, pages, console_errors)
                run_lag_fix_checks(page, larger_batch, console_errors)
            finally:
                context.close()
                browser.close()
    finally:
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
