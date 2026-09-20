import io
import os
import queue
import secrets
import shutil
import tempfile
import threading
import time
import zipfile

import runner

# Job registry and result storage for translation work.
#
# Metadata lives in memory under a lock; rendered page images live on disk,
# not in the process, so a large batch is a disk-space and TTL question, not
# a memory one. Rendered manga pages are roughly 1-2 MB each, so a 100-page
# batch is on the order of 100-200 MB on disk for up to DEFAULT_TTL_SECONDS,
# after which evict_expired reclaims it. Metadata is small, is read on every
# poll, and has no value once the process dies, so persisting it would buy
# nothing.
#
# There is exactly one worker thread. The OCR models in ocr.py are
# process-global singletons (_ocr_models, _manga_ocr_model) and are not
# safe to call concurrently, and a single container has one model's worth
# of memory to spend. Serialising here makes that a stated invariant of
# the store rather than an accident of how requests happen to arrive. A
# larger MAX_PAGES_PER_BATCH does not change this: pages still translate
# strictly one at a time (see docs/LIMITATIONS.md LIM-3), so a bigger batch
# is a proportionally longer wait, not a throughput gain.

DEFAULT_TTL_SECONDS = 3 * 60 * 60
# raised from 40 to 100 after a real 90-page bulk translation run (Kaggle,
# 2026-09) confirmed the pipeline handles a batch that size correctly; 100
# leaves headroom above that real test rather than sitting right at its edge.
MAX_PAGES_PER_BATCH = 100

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class PageJob:
    def __init__(self, index, filename, input_path, source_lang):
        self.index = index
        self.filename = filename
        self.input_path = input_path
        self.source_lang = source_lang
        self.status = STATUS_PENDING
        self.error = None
        self.image_token = None
        self.regions = []
        self.duration_seconds = None

    def to_dict(self):
        return {
            "index": self.index,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "image_token": self.image_token,
            "region_count": len(self.regions),
            "duration_seconds": self.duration_seconds,
        }


class Batch:
    def __init__(self, batch_id, source_lang, directory):
        self.id = batch_id
        self.source_lang = source_lang
        self.directory = directory
        self.created_at = time.time()
        self.pages = []

    @property
    def status(self):
        states = [p.status for p in self.pages]
        if not states:
            return STATUS_DONE
        if any(s == STATUS_RUNNING for s in states):
            return STATUS_RUNNING
        if all(s in (STATUS_DONE, STATUS_FAILED) for s in states):
            return STATUS_DONE
        if any(s in (STATUS_DONE, STATUS_FAILED) for s in states):
            return STATUS_RUNNING
        return STATUS_PENDING

    def to_dict(self):
        states = [p.status for p in self.pages]
        return {
            "batch_id": self.id,
            "source_lang": self.source_lang,
            "status": self.status,
            "created_at": self.created_at,
            "total": len(self.pages),
            "completed": sum(1 for s in states if s == STATUS_DONE),
            "failed": sum(1 for s in states if s == STATUS_FAILED),
            "pages": [p.to_dict() for p in self.pages],
        }


class BatchStore:
    def __init__(self, root=None, ttl_seconds=DEFAULT_TTL_SECONDS):
        self.root = root or os.environ.get("MANGA_TRANSLATOR_DATA_DIR") or tempfile.mkdtemp(
            prefix="manga-translator-"
        )
        os.makedirs(self.root, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._batches = {}
        self._images = {}
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._work_loop, daemon=True, name="translation-worker")
        self._worker.start()

    def _work_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            batch_id, page_index = item
            try:
                self._process(batch_id, page_index)
            except Exception as exc:
                # a failure here must not take the worker thread down, or every
                # subsequent job in the process would silently never run
                with self._lock:
                    batch = self._batches.get(batch_id)
                    if batch and page_index < len(batch.pages):
                        page = batch.pages[page_index]
                        page.status = STATUS_FAILED
                        page.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def _process(self, batch_id, page_index):
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return
            page = batch.pages[page_index]
            page.status = STATUS_RUNNING
            input_path = page.input_path
            source_lang = page.source_lang
            output_path = os.path.join(batch.directory, f"page_{page_index:04d}_out.png")

        started = time.time()
        try:
            result = runner.run_page(input_path, output_path, source_lang)
        except Exception as exc:
            with self._lock:
                page.status = STATUS_FAILED
                page.error = f"{type(exc).__name__}: {exc}"
                page.duration_seconds = round(time.time() - started, 2)
            return

        token = secrets.token_urlsafe(16)
        with self._lock:
            self._images[token] = output_path
            page.image_token = token
            page.regions = result["regions"]
            page.status = STATUS_DONE
            page.duration_seconds = round(time.time() - started, 2)

    def create_batch(self, files, source_lang):
        # files: list of (filename, bytes)
        if len(files) > MAX_PAGES_PER_BATCH:
            raise ValueError(f"batch exceeds the {MAX_PAGES_PER_BATCH} page limit")

        self.evict_expired()

        batch_id = secrets.token_urlsafe(16)
        directory = os.path.join(self.root, batch_id)
        os.makedirs(directory, exist_ok=True)
        batch = Batch(batch_id, source_lang, directory)

        for index, (filename, data) in enumerate(files):
            extension = os.path.splitext(filename)[1].lower() or ".png"
            input_path = os.path.join(directory, f"page_{index:04d}_in{extension}")
            with open(input_path, "wb") as handle:
                handle.write(data)
            batch.pages.append(PageJob(index, filename, input_path, source_lang))

        with self._lock:
            self._batches[batch_id] = batch

        for page in batch.pages:
            self._queue.put((batch_id, page.index))

        return batch

    def register_image(self, path):
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._images[token] = path
        return token

    def image_path(self, token):
        with self._lock:
            return self._images.get(token)

    def get_batch(self, batch_id):
        with self._lock:
            return self._batches.get(batch_id)

    def page_regions(self, batch_id, page_index):
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None or page_index >= len(batch.pages):
                return None
            return list(batch.pages[page_index].regions)

    def build_zip(self, batch_id):
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            entries = [
                (p.index, p.filename, self._images.get(p.image_token))
                for p in batch.pages
                if p.status == STATUS_DONE and p.image_token
            ]

        if not entries:
            return None

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, filename, path in entries:
                if not path or not os.path.exists(path):
                    continue
                stem = os.path.splitext(os.path.basename(filename))[0]
                # index prefix preserves reading order; zip readers and file
                # managers sort lexicographically, not by insertion order
                archive.write(path, arcname=f"{index:04d}_{stem}.png")
        buffer.seek(0)
        return buffer

    def delete_batch(self, batch_id):
        with self._lock:
            batch = self._batches.pop(batch_id, None)
            if batch is None:
                return False
            for page in batch.pages:
                if page.image_token:
                    self._images.pop(page.image_token, None)
            directory = batch.directory
        shutil.rmtree(directory, ignore_errors=True)
        return True

    def evict_expired(self):
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired = [bid for bid, b in self._batches.items() if b.created_at < cutoff]
        for batch_id in expired:
            self.delete_batch(batch_id)
        return len(expired)

    def stats(self):
        with self._lock:
            return {
                "batches": len(self._batches),
                "queued": self._queue.qsize(),
                "worker_alive": self._worker.is_alive(),
                "storage_root": self.root,
                "ttl_seconds": self.ttl_seconds,
            }


store = BatchStore()
