import os


def _use_stub():
    return os.environ.get("USE_STUB_PIPELINE", "").strip() in ("1", "true", "yes")


def _to_int_bounds(bounds):
    x1, y1, x2, y2 = bounds
    return [int(x1), int(y1), int(x2), int(y2)]


def normalise_result(result):
    # translate_page returns model-layer objects: bounds may hold numpy
    # scalars or floats depending on which detection path produced them,
    # and merged_boxes carries fields the HTTP layer has no use for.
    # flattening here keeps json serialisation out of the route handlers
    # and gives both the single and batch paths one identical shape.
    regions = []
    boxes = result.get("merged_boxes", [])
    translations = result.get("translations", [])
    for i, box in enumerate(boxes):
        bounds = box.get("bounds")
        if bounds is None:
            coords = box.get("coords", [])
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            bounds = (min(xs), min(ys), max(xs), max(ys)) if coords else (0, 0, 0, 0)
        regions.append({
            "bounds": _to_int_bounds(bounds),
            "source_text": box.get("text", ""),
            "translation": translations[i] if i < len(translations) else "",
        })
    return {
        "regions": regions,
        "skipped_boxes": [str(s) for s in result.get("skipped_boxes", [])],
    }


def run_page(input_path, output_path, source_lang):
    # single seam between the HTTP/job layer and the ML pipeline. the real
    # pipeline pulls in paddleocr, manga-ocr and torch and downloads model
    # weights on first use, which is not available in every environment the
    # HTTP layer needs to be exercised in. selecting the implementation here
    # keeps that concern out of batch_store and main entirely.
    if _use_stub():
        from stub_pipeline import translate_page
    else:
        from pipeline import translate_page

    result = translate_page(input_path, output_path, source_lang=source_lang)
    return normalise_result(result)


def pipeline_mode():
    return "stub" if _use_stub() else "real"
