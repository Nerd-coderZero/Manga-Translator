import glob
import os
import time

from PIL import Image, ImageDraw

# Stand-in for pipeline.translate_page used to exercise the HTTP, job and
# frontend layers without loading paddleocr, manga-ocr or torch. It performs
# no detection, recognition or translation. It reproduces translate_page's
# return contract exactly: merged_boxes with bounds and text, a translations
# list of equal length, skipped_boxes, and an output image written to
# output_path. Region positions are derived from image dimensions so the
# output varies with the input rather than being fixed.
#
# Pillow is the only third-party import here on purpose. The font lookup
# below duplicates pipeline_core.find_available_font rather than importing
# it, because pipeline_core imports numpy at module scope, and importing a
# single helper from it pulls numpy into the stub's dependency set. That
# defeats the point of a seam whose reason for existing is to run without
# the pipeline's dependencies installed.


def _find_available_font():
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    for pattern in ("/usr/share/fonts/**/*Bold*.ttf", "/usr/share/fonts/**/*.ttf",
                    "C:\\Windows\\Fonts\\*.ttf"):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    return None

_STUB_LINES = [
    "stub region one",
    "stub region two",
    "stub region three",
    "stub region four",
]


def translate_page(input_path, output_path, source_lang="zh"):
    img = Image.open(input_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    font_path = _find_available_font()

    merged_boxes = []
    translations = []

    columns = 2
    rows = 2
    pad = max(8, min(width, height) // 40)
    cell_w = width // columns
    cell_h = height // rows

    for index in range(columns * rows):
        col = index % columns
        row = index // columns
        x1 = col * cell_w + pad
        y1 = row * cell_h + pad
        x2 = (col + 1) * cell_w - pad
        y2 = (row + 1) * cell_h - pad
        if x2 <= x1 or y2 <= y1:
            continue

        text = _STUB_LINES[index % len(_STUB_LINES)]
        merged_boxes.append({
            "bounds": (x1, y1, x2, y2),
            "text": f"[{source_lang}] source {index}",
            "avg_original_height": (y2 - y1) // 4,
            "original_area": (x2 - x1) * (y2 - y1),
        })
        translations.append(text)

        draw.rectangle([x1, y1, x2, y2], outline=(220, 40, 40), width=3)
        _draw_label(draw, x1 + 6, y1 + 6, text, font_path, max(12, cell_h // 20))

    _draw_label(draw, pad, height - pad - 24, "STUB PIPELINE - NO TRANSLATION PERFORMED",
                font_path, max(14, height // 60), fill=(220, 40, 40))

    # deliberate delay so batch progress polling has something to observe
    time.sleep(1.2)

    img.save(output_path)

    return {
        "merged_boxes": merged_boxes,
        "translations": translations,
        "skipped_boxes": [],
        "output_path": output_path,
    }


def _draw_label(draw, x, y, text, font_path, size, fill=(20, 20, 20)):
    from PIL import ImageFont
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4], fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=fill)
