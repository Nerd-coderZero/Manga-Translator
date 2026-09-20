import os

from ocr import detect_and_read_text
from translation import translate_text
from placement import erase_and_paste_text
from pipeline_core import merge_text_boxes, is_garbage_text


def translate_page(input_path, output_path, source_lang="zh"):
    raw_boxes = detect_and_read_text(input_path, source_lang)

    # Japanese detection already merges fragmented vertical text columns
    # into full regions internally (see ocr.py's _detect_and_read_japanese),
    # so running the generic merge pass again here would be redundant and
    # risks incorrectly merging separate, unrelated balloons that happen to
    # sit close together. only Chinese needs this merge step.
    if source_lang == "ja":
        merged_boxes = raw_boxes
        for b in merged_boxes:
            b.setdefault("bounds", _coords_to_bounds(b["coords"]))
            b.setdefault("avg_original_height", b["bounds"][3] - b["bounds"][1])
            b.setdefault("original_area", (b["bounds"][2] - b["bounds"][0]) * (b["bounds"][3] - b["bounds"][1]))
    else:
        merged_boxes = merge_text_boxes(raw_boxes)

    translations = []
    for b in merged_boxes:
        if is_garbage_text(b["text"], check_length=False):
            translations.append("")
        else:
            translations.append(translate_text(b["text"], source_lang))

    result_img, skipped_boxes = erase_and_paste_text(input_path, merged_boxes, translations, output_path)

    return {
        "merged_boxes": merged_boxes,
        "translations": translations,
        "skipped_boxes": skipped_boxes,
        "output_path": output_path,
    }


def _coords_to_bounds(coords):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
