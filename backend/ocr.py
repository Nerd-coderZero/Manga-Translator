from paddleocr import PaddleOCR

# separate model instances per language: PaddleOCR requires a distinct
# detection/recognition model to be loaded per language, they cannot
# share one instance. detection params are tuned per language too, since
# a threshold set tuned for Chinese manga has no reason to be correct
# for Japanese, which has different text density and furigana/ruby text
# that Chinese does not.
#
# paddleocr 3.7.0 / paddlepaddle 3.3.1 (the versions that actually resolve
# from PyPI now -- the notebook's 2.6.1/2.7.3 pair no longer exists there
# at all) replaced the entire class this pipeline was built against. the
# constructor keyword names below, the removal of the old use_angle_cls
# flag in favour of use_textline_orientation, and enable_mkldnn=False were
# all confirmed directly: PaddleOCR(**old_kwarg_names) raises immediately
# since none of det_limit_side_len / det_db_thresh / etc. are accepted
# parameters on this version, and without enable_mkldnn=False, .predict()
# raised NotImplementedError from paddle's oneDNN backend
# (ConvertPirAttribute2RuntimeAttribute not support ...) on a real image,
# independent of image content -- confirmed by isolating the same image
# with and without that flag before deciding it was the cause.
_ocr_models = {}
_manga_ocr_model = None

_LANG_CONFIG = {
    "zh": {
        "paddle_lang": "ch",
        "text_det_limit_side_len": 2560,
        "text_det_limit_type": "max",
        "text_det_thresh": 0.15,
        "text_det_box_thresh": 0.3,
        "text_det_unclip_ratio": 2.0,
    },
    "ja": {
        "paddle_lang": "japan",
        "text_det_limit_side_len": 2560,
        "text_det_limit_type": "max",
        "text_det_thresh": 0.1,
        "text_det_box_thresh": 0.15,
        "text_det_unclip_ratio": 1.8,
    },
}


def get_ocr_model(lang="zh"):
    config = _LANG_CONFIG.get(lang)
    if config is None:
        raise ValueError(f"unsupported language: {lang}. use 'zh' or 'ja'.")

    if lang not in _ocr_models:
        _ocr_models[lang] = PaddleOCR(
            lang=config["paddle_lang"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            enable_mkldnn=False,
            text_det_limit_side_len=config["text_det_limit_side_len"],
            text_det_limit_type=config["text_det_limit_type"],
            text_det_thresh=config["text_det_thresh"],
            text_det_box_thresh=config["text_det_box_thresh"],
            text_det_unclip_ratio=config["text_det_unclip_ratio"],
        )
    return _ocr_models[lang]


def get_manga_ocr_model():
    global _manga_ocr_model
    if _manga_ocr_model is None:
        from manga_ocr import MangaOcr
        # force_cpu=True is required on the deployed Space, not optional
        # hardening. manga-ocr auto-selects torch.cuda.is_available() as
        # its device, and on ZeroGPU that reports True outside any
        # @spaces.GPU-decorated function (that is what "CUDA emulation
        # mode" means -- code sees a GPU without one actually being
        # attached). manga-ocr then genuinely tries to move its model to
        # a real "cuda" device from inside batch_store's worker thread,
        # which is not decorated, and that real (non-emulated) CUDA call
        # is what raised "Low-level CUDA init reached" on a real deploy.
        # this pipeline is CPU-only by design regardless -- paddlepaddle
        # is the CPU build and nothing here is meant to use ZeroGPU for
        # acceleration (see docs/DECISIONS.md) -- so forcing CPU here is
        # consistent with that design, not a workaround around it.
        _manga_ocr_model = MangaOcr(force_cpu=True)
    return _manga_ocr_model


def _box_bounds(coords):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _box_area(bounds):
    x1, y1, x2, y2 = bounds
    return max(0, x2 - x1) * max(0, y2 - y1)


def _containment_ratio(inner, outer):
    # fraction of the inner box's own area that overlaps the outer box.
    # 1.0 means inner sits fully within outer.
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    overlap_x1, overlap_y1 = max(ix1, ox1), max(iy1, oy1)
    overlap_x2, overlap_y2 = min(ix2, ox2), min(iy2, oy2)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    inner_area = _box_area(inner)
    if inner_area == 0:
        return 0.0
    return overlap_area / inner_area


def _drop_redundant_oversized_boxes(
    dt_boxes,
    area_ratio_thresh=8.0,
    containment_thresh=0.9,
    min_contained_count=3,
):
    # on dense vertical text clusters, PaddleOCR's DB detector can emit
    # one large bounding box covering a whole region IN ADDITION TO the
    # correct small per-fragment boxes for that same region -- both
    # survive in the raw output, they are not merged together. verified
    # against real manga pages: the small boxes account for the actual
    # text regions, the large box is a redundant duplicate detection.
    #
    # a box is dropped only if it is both a strong area outlier relative
    # to the rest of the page AND almost fully contains several other
    # independently-detected boxes. either signal alone is insufficient:
    # a genuinely large single box (e.g. one long line of text) would
    # trip the area check alone, and a box containing one smaller box
    # could be a legitimate nested detection rather than a duplicate.
    if len(dt_boxes) < min_contained_count + 1:
        return dt_boxes

    bounds_list = [_box_bounds(coords) for coords in dt_boxes]
    areas = [_box_area(b) for b in bounds_list]
    sorted_areas = sorted(areas)
    n = len(sorted_areas)
    median_area = sorted_areas[n // 2]
    if median_area == 0:
        return dt_boxes

    keep = []
    for i, coords in enumerate(dt_boxes):
        area = areas[i]
        if area < median_area * area_ratio_thresh:
            keep.append(coords)
            continue

        contained_count = 0
        for j, other_bounds in enumerate(bounds_list):
            if i == j:
                continue
            if _containment_ratio(other_bounds, bounds_list[i]) >= containment_thresh:
                contained_count += 1

        if contained_count < min_contained_count:
            keep.append(coords)
        # else: dropped as a redundant oversized detection

    return keep


def _merge_vertical_fragments(raw_results, x_overlap_ratio_thresh=0.3, x_gap_thresh=15, y_gap_thresh=40):
    # PaddleOCR's Japanese detection frequently slices a single vertical
    # text column into multiple narrow boxes, either overlapping or sitting
    # immediately adjacent with a small gap, each capturing only part of
    # the column. this groups boxes that are close/overlapping in x AND
    # close/overlapping in y into a single merged region before recognition
    # runs, rather than letting each fragment get read independently.
    items = []
    for line in raw_results:
        coords = line[0]
        x1, y1, x2, y2 = _box_bounds(coords)
        items.append({"coords": coords, "bounds": (x1, y1, x2, y2)})

    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            x1a, y1a, x2a, y2a = items[i]["bounds"]
            x1b, y1b, x2b, y2b = items[j]["bounds"]

            overlap_x = max(0, min(x2a, x2b) - max(x1a, x1b))
            gap_x = max(0, max(x1a, x1b) - min(x2a, x2b))
            narrower_width = min(x2a - x1a, x2b - x1b)

            x_close_enough = False
            if narrower_width > 0 and (overlap_x / narrower_width) >= x_overlap_ratio_thresh:
                x_close_enough = True
            elif gap_x <= x_gap_thresh:
                x_close_enough = True

            y_gap = max(0, max(y1a, y1b) - min(y2a, y2b))
            y_close_enough = y_gap <= y_gap_thresh

            if x_close_enough and y_close_enough:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged_regions = []
    for indices in groups.values():
        group_bounds = [items[i]["bounds"] for i in indices]
        x1 = min(b[0] for b in group_bounds)
        y1 = min(b[1] for b in group_bounds)
        x2 = max(b[2] for b in group_bounds)
        y2 = max(b[3] for b in group_bounds)
        merged_regions.append({
            "coords": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "source_fragment_count": len(indices),
        })

    return merged_regions


def _run_paddle_predict(model, image_path):
    # PaddleOCR 2.x's .ocr(rec=False) wrapper did `if not dt_boxes:`
    # internally, which raised ValueError once more than one box was found
    # (numpy array truthiness is ambiguous for arrays with >1 element) --
    # that was the reason this pipeline used to bypass the wrapper and call
    # .text_detector directly. Neither the bug nor the workaround exists on
    # 3.7.0: .text_detector is gone entirely (confirmed -- dir(PaddleOCR)
    # no longer lists it), and .predict() is a different interface that
    # returns a list of dict-like results rather than raising on truthiness
    # checks. .ocr() still exists but is now a deprecated wrapper around
    # .predict() with the identical dict-based return shape, confirmed by
    # calling it directly and inspecting the result type, so there is no
    # reason to keep two code paths -- everything calls .predict().
    results = model.predict(image_path)
    if not results:
        return None
    return results[0]


def _coords_from_dt_polys(dt_polys):
    # dt_polys entries are numpy arrays (confirmed dtype int16, shape
    # (4, 2)); downstream code (_box_bounds, _drop_redundant_oversized_boxes,
    # _merge_vertical_fragments) expects plain python [[x, y], ...] lists,
    # matching what the old dt_boxes format provided.
    return [coords.tolist() if hasattr(coords, "tolist") else list(coords) for coords in dt_polys]


def detect_and_read_text(image_path, source_lang="zh"):
    if source_lang == "ja":
        return _detect_and_read_japanese(image_path)

    model = get_ocr_model(source_lang)
    result = _run_paddle_predict(model, image_path)
    if result is None:
        return []

    coords_list = _coords_from_dt_polys(result.get("dt_polys", []))
    rec_texts = result.get("rec_texts", [])
    rec_scores = result.get("rec_scores", [])

    boxes = []
    for coords, text, confidence in zip(coords_list, rec_texts, rec_scores):
        boxes.append({"coords": coords, "text": text, "confidence": confidence})
    return boxes


def _detect_and_read_japanese(image_path):
    # detection stays on PaddleOCR (it locates text regions reasonably,
    # even if it fragments vertical columns), but recognition switches to
    # manga-ocr, which is purpose-built for vertical Japanese manga text
    # and handles it far better than PaddleOCR's general recognition model.
    #
    # .predict() always runs detection and recognition together on this
    # version -- there is no longer a detection-only entry point (confirmed:
    # dir(PaddleOCR) has no detector-only method, and the nested
    # paddlex_pipeline object exposes no detector attribute either). rec_texts
    # and rec_scores from this call are discarded; only dt_polys is used.
    # this means PaddleOCR's own recognizer runs on every region even though
    # its output is thrown away, which is wasted compute compared to the old
    # detection-only call -- see docs/LIMITATIONS.md.
    from PIL import Image
    from pipeline_core import is_garbage_text

    det_model = get_ocr_model("ja")
    result = _run_paddle_predict(det_model, image_path)
    if result is None:
        return []

    dt_boxes = _coords_from_dt_polys(result.get("dt_polys", []))

    if not dt_boxes:
        return []

    # remove redundant oversized boxes before fragment merging runs, so
    # the merge step never sees a spurious box competing with the real
    # per-fragment detections it is meant to reassemble
    dt_boxes = _drop_redundant_oversized_boxes(dt_boxes)

    if not dt_boxes:
        return []

    # wrap into the [coords, ...] line format _merge_vertical_fragments expects
    raw_lines = [[coords] for coords in dt_boxes]
    merged_regions = _merge_vertical_fragments(raw_lines)

    mocr = get_manga_ocr_model()
    full_img = Image.open(image_path).convert("RGB")

    boxes = []
    for region in merged_regions:
        x1, y1, x2, y2 = _box_bounds(region["coords"])
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(full_img.width, int(x2)), min(full_img.height, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = full_img.crop((x1, y1, x2, y2))

        # manga-ocr's recognizer struggles on very small crops (e.g. tiny
        # 2-3 character bubbles) since there's little pixel signal to work
        # with at native resolution. upscaling before recognition gives it
        # more to work with and often recovers text that'd otherwise come
        # back blank or garbled.
        min_dim = min(crop.width, crop.height)
        if min_dim < 60:
            scale = max(2, int(60 / max(min_dim, 1)))
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)

        text = mocr(crop)

        # loosened detection thresholds (needed to recover faint/short
        # japanese text columns) also let through non-text regions like
        # screentone or hair shading. manga-ocr's output on those is a
        # far more reliable "was there real text here" signal than the
        # detector's own confidence score, so filter post-recognition.
        if is_garbage_text(text):
            continue

        boxes.append({
            "coords": region["coords"],
            "text": text,
            "confidence": 1.0,  # manga-ocr does not expose a confidence score
        })

    return boxes
