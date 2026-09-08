import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

BASE = "http://127.0.0.1:8011"
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def request(method, path, body=None, headers=None):
    req = urllib.request.Request(BASE + path, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def multipart(fields, files):
    # minimal multipart encoder so the suite has no third-party dependency
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    for name, filename, content, content_type in files:
        head = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        parts.append(head + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def make_page(path, label):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (900, 1300), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 860, 1260], outline=(180, 180, 180), width=2)
    draw.text((60, 60), label, fill=(0, 0, 0))
    image.save(path)
    return path


def start_server(data_dir):
    env = dict(os.environ)
    env["USE_STUB_PIPELINE"] = "1"
    env["MANGA_TRANSLATOR_DATA_DIR"] = data_dir
    env["CORS_ALLOW_ORIGINS"] = "http://localhost:5500,http://127.0.0.1:5500"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8011"],
        cwd=BACKEND_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        try:
            status, _, _ = request("GET", "/api/health")
            if status == 200:
                return process
        except Exception:
            pass
        if process.poll() is not None:
            print(process.stdout.read().decode(errors="replace"))
            raise RuntimeError("server exited during startup")
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy")


def check_stub_dependency_isolation():
    # The stub exists so the HTTP, job and frontend layers can run without
    # the pipeline's dependencies installed. Importing it in a subprocess
    # with numpy blocked is the only way to assert that property: once numpy
    # is present in the parent interpreter, a transitive import of it is
    # invisible. BUG-4 was exactly this, and passed every other test.
    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] in ('numpy', 'cv2', 'torch', 'paddleocr') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(name + ' must not be imported by the stub path')\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, %r)\n"
        "import runner\n"
        "import stub_pipeline\n"
        "print('ok')\n" % BACKEND_DIR
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    check(
        "stub path imports without numpy, cv2, torch or paddleocr",
        result.returncode == 0 and "ok" in result.stdout,
        (result.stderr.strip().splitlines() or ["-"])[-1],
    )


def run(tmp_dir):
    check_stub_dependency_isolation()

    pages = [
        make_page(os.path.join(tmp_dir, "p1.png"), "test page 1"),
        make_page(os.path.join(tmp_dir, "p2.png"), "test page 2"),
        make_page(os.path.join(tmp_dir, "p3.png"), "test page 3"),
    ]

    status, headers, body = request("GET", "/api/health")
    health = json.loads(body)
    check("health returns 200", status == 200)
    check("health reports stub pipeline", health.get("pipeline_mode") == "stub", health.get("pipeline_mode"))
    check("health reports live worker thread", health["store"]["worker_alive"] is True)

    # CORS preflight from an allowed origin
    status, headers, _ = request(
        "OPTIONS", "/api/batch",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    allow_origin = headers.get("access-control-allow-origin")
    allow_methods = headers.get("access-control-allow-methods", "")
    check("preflight from allowed origin succeeds", status in (200, 204), f"status={status}")
    check("preflight echoes the allowed origin", allow_origin == "http://localhost:5500", str(allow_origin))
    check("preflight advertises POST", "POST" in allow_methods, allow_methods)

    # CORS from a disallowed origin must not receive the allow header
    status, headers, _ = request(
        "OPTIONS", "/api/batch",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    check(
        "preflight from disallowed origin gets no allow-origin header",
        headers.get("access-control-allow-origin") is None,
        str(headers.get("access-control-allow-origin")),
    )

    # simple GET carries the allow header for an allowed origin
    status, headers, _ = request("GET", "/api/health", headers={"Origin": "http://127.0.0.1:5500"})
    check(
        "GET carries allow-origin for allowed origin",
        headers.get("access-control-allow-origin") == "http://127.0.0.1:5500",
        str(headers.get("access-control-allow-origin")),
    )

    # validation
    body, content_type = multipart(
        [("source_lang", "ko")],
        [("files", "p1.png", open(pages[0], "rb").read(), "image/png")],
    )
    status, _, payload = request("POST", "/api/batch", body, {"Content-Type": content_type})
    check("unsupported source_lang rejected with 422", status == 422, f"status={status}")

    body, content_type = multipart(
        [("source_lang", "ja")],
        [("files", "notes.txt", b"not an image", "text/plain")],
    )
    status, _, payload = request("POST", "/api/batch", body, {"Content-Type": content_type})
    check("unsupported file extension rejected with 422", status == 422, f"status={status}")

    body, content_type = multipart(
        [("source_lang", "ja")],
        [("files", "empty.png", b"", "image/png")],
    )
    status, _, _ = request("POST", "/api/batch", body, {"Content-Type": content_type})
    check("empty file rejected with 422", status == 422, f"status={status}")

    status, _, _ = request("GET", "/api/batch/nonexistent")
    check("unknown batch id returns 404", status == 404, f"status={status}")

    status, _, _ = request("GET", "/api/image/nonexistent")
    check("unknown image token returns 404", status == 404, f"status={status}")

    # happy path batch
    files = [("files", os.path.basename(p), open(p, "rb").read(), "image/png") for p in pages]
    body, content_type = multipart([("source_lang", "ja")], files)
    status, _, payload = request("POST", "/api/batch", body, {"Content-Type": content_type})
    check("batch submit returns 202", status == 202, f"status={status}")
    created = json.loads(payload)
    batch_id = created["batch_id"]
    check("batch submit reports the page count", created["total"] == 3, str(created["total"]))

    # the first poll should show work still outstanding, which is the whole
    # reason the endpoint is asynchronous
    _, _, payload = request("GET", f"/api/batch/{batch_id}")
    first_poll = json.loads(payload)
    check(
        "first poll shows work outstanding",
        first_poll["completed"] + first_poll["failed"] < first_poll["total"],
        f"{first_poll['completed'] + first_poll['failed']}/{first_poll['total']}",
    )

    # a page that is not finished must not be readable
    status, _, _ = request("GET", f"/api/batch/{batch_id}/page/2")
    check("unfinished page returns 409", status == 409, f"status={status}")

    deadline = time.time() + 120
    batch = first_poll
    while time.time() < deadline:
        _, _, payload = request("GET", f"/api/batch/{batch_id}")
        batch = json.loads(payload)
        if batch["status"] == "done":
            break
        time.sleep(0.5)

    check("batch reaches done", batch["status"] == "done", batch["status"])
    check("all three pages completed", batch["completed"] == 3, str(batch["completed"]))
    check("no pages failed", batch["failed"] == 0, str(batch["failed"]))
    durations = [p["duration_seconds"] for p in batch["pages"]]
    check("every page recorded a duration", all(d is not None for d in durations), str(durations))

    status, _, payload = request("GET", f"/api/batch/{batch_id}/page/0")
    page = json.loads(payload)
    check("completed page is readable", status == 200, f"status={status}")
    check("page carries regions", len(page["regions"]) > 0, str(len(page["regions"])))
    region = page["regions"][0]
    check("region has integer bounds of length 4",
          isinstance(region["bounds"], list) and len(region["bounds"]) == 4
          and all(isinstance(v, int) for v in region["bounds"]), str(region["bounds"]))
    check("region has source and translation keys",
          "source_text" in region and "translation" in region)

    status, headers, image_bytes = request("GET", page["image_url"])
    check("rendered image is served", status == 200, f"status={status}")
    check("rendered image is a PNG", image_bytes[:8] == b"\x89PNG\r\n\x1a\n")
    check("image content type is image/png", headers.get("content-type") == "image/png",
          str(headers.get("content-type")))

    status, _, page_out_of_range = request("GET", f"/api/batch/{batch_id}/page/99")
    check("page index out of range returns 404", status == 404, f"status={status}")

    status, headers, zip_bytes = request("GET", f"/api/batch/{batch_id}/download")
    check("zip download returns 200", status == 200, f"status={status}")
    check("zip has an attachment disposition",
          "attachment" in headers.get("content-disposition", ""),
          headers.get("content-disposition", ""))
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
        names = archive.namelist()
        check("zip contains all three pages", len(names) == 3, str(names))
        check("zip entries are ordered by index",
              names == sorted(names) and names[0].startswith("0000"), str(names))
        check("zip is not corrupt", archive.testzip() is None)
    except zipfile.BadZipFile as exc:
        check("zip is a valid archive", False, str(exc))

    # single page endpoint
    body, content_type = multipart(
        [("source_lang", "zh")],
        [("file", "single.png", open(pages[0], "rb").read(), "image/png")],
    )
    status, _, payload = request("POST", "/api/translate", body, {"Content-Type": content_type})
    check("single translate returns 200", status == 200, f"status={status}")
    single = json.loads(payload)
    check("single translate returns regions", len(single["regions"]) > 0, str(len(single["regions"])))
    status, _, single_image = request("GET", single["image_url"])
    check("single translate image is served", status == 200 and single_image[:8] == b"\x89PNG\r\n\x1a\n")

    # deletion
    status, _, _ = request("DELETE", f"/api/batch/{batch_id}")
    check("delete returns 200", status == 200, f"status={status}")
    status, _, _ = request("GET", f"/api/batch/{batch_id}")
    check("deleted batch is gone", status == 404, f"status={status}")
    status, _, _ = request("GET", page["image_url"])
    check("image token revoked after delete", status == 404, f"status={status}")
    status, _, _ = request("DELETE", f"/api/batch/{batch_id}")
    check("deleting twice returns 404", status == 404, f"status={status}")

    # static frontend must still be reachable behind the api routes
    status, _, index_html = request("GET", "/index.html")
    check("index.html is served", status == 200 and b"Manga Translator" in index_html, f"status={status}")
    status, _, reader_html = request("GET", "/reader.html")
    check("reader.html is served", status == 200 and b"Reader" in reader_html, f"status={status}")


def main():
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="mt-apitest-")
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    process = start_server(data_dir)
    try:
        run(tmp_dir)
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
