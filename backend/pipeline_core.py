import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import os


def find_available_font():
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    search_patterns = [
        "/usr/share/fonts/**/*Bold*.ttf",
        "/usr/share/fonts/**/*bold*.ttf",
        "/usr/share/fonts/**/*.ttf",
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    return None


def get_box_bounds(coords):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def boxes_overlap_or_close(b1, b2, gap_threshold=15):
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    expanded_a = (x1a - gap_threshold, y1a - gap_threshold, x2a + gap_threshold, y2a + gap_threshold)
    if expanded_a[0] > x2b or x1b > expanded_a[2]:
        return False
    if expanded_a[1] > y2b or y1b > expanded_a[3]:
        return False
    return True


def get_contrast_ratio(color1, color2):
    def luminance(c):
        r, g, b = [x / 255.0 for x in c]
        r, g, b = [((x / 12.92) if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4) for x in (r, g, b)]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    l1, l2 = luminance(color1), luminance(color2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def ensure_readable_text_color(text_color, bg_color, min_contrast=4.5):
    if get_contrast_ratio(text_color, bg_color) >= min_contrast:
        return text_color
    black_contrast = get_contrast_ratio((0, 0, 0), bg_color)
    white_contrast = get_contrast_ratio((255, 255, 255), bg_color)
    return (0, 0, 0) if black_contrast >= white_contrast else (255, 255, 255)


def get_dominant_text_color(img_np, x1, y1, x2, y2, bg_color, threshold=60):
    region = img_np[y1:y2, x1:x2].reshape(-1, 3)
    bg_arr = np.array(bg_color)
    distances = np.linalg.norm(region.astype(int) - bg_arr.astype(int), axis=1)
    text_pixels = region[distances > threshold]
    if len(text_pixels) == 0:
        return (0, 0, 0)
    return tuple(int(v) for v in np.median(text_pixels, axis=0))


def wrap_text_to_fit(draw, text, font, max_width):
    words = text.split(" ")
    if len(words) == 1:
        words = list(text)

    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line and " " in text else "") + word
        test_bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = test_bbox[2] - test_bbox[0]
        if test_width <= max_width or current_line == "":
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def is_garbage_text(text, max_reasonable_length=80):
    if not text or len(text.strip()) == 0:
        return True

    stripped = text.strip()

    if len(stripped) > max_reasonable_length:
        return True

    normalized = stripped.replace("O", "0").replace("〇", "0").replace("○", "0")

    if len(normalized) >= 8:
        most_common_char_count = max(normalized.count(c) for c in set(normalized))
        if most_common_char_count / len(normalized) > 0.4:
            return True

    digit_count = sum(1 for c in normalized if c.isdigit())
    if len(normalized) >= 6 and digit_count / len(normalized) > 0.5:
        return True

    return False


def merge_text_boxes(boxes, gap_threshold=35):
    bounds_list = [get_box_bounds(b["coords"]) for b in boxes]
    n = len(boxes)
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
            if boxes_overlap_or_close(bounds_list[i], bounds_list[j], gap_threshold):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for indices in groups.values():
        group_bounds = [bounds_list[i] for i in indices]
        x1 = min(b[0] for b in group_bounds)
        y1 = min(b[1] for b in group_bounds)
        x2 = max(b[2] for b in group_bounds)
        y2 = max(b[3] for b in group_bounds)

        sorted_indices = sorted(indices, key=lambda i: bounds_list[i][0], reverse=True)
        combined_text = "".join(boxes[i]["text"] for i in sorted_indices)

        individual_heights = [(bounds_list[i][3] - bounds_list[i][1]) for i in indices]
        avg_original_height = sum(individual_heights) / len(individual_heights)

        merged.append({
            "bounds": (x1, y1, x2, y2),
            "text": combined_text,
            "source_indices": indices,
            "avg_original_height": avg_original_height,
            "original_area": (x2 - x1) * (y2 - y1),
        })

    merged.sort(key=lambda m: m["bounds"][0], reverse=True)
    return merged


def fit_text_in_box(draw, text, box_width, box_height, font_path, max_font_size, min_font_size=8):
    font_size = max_font_size

    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            break

        lines = wrap_text_to_fit(draw, text, font, box_width)

        line_heights = []
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            line_heights.append(line_height)
            max_line_width = max(max_line_width, line_width)

        line_spacing = int(font_size * 0.2)
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0

        if max_line_width <= box_width and total_height <= box_height:
            return font, lines, line_heights, line_spacing

        font_size -= 1

    font = ImageFont.truetype(font_path, min_font_size) if font_path else ImageFont.load_default()
    lines = wrap_text_to_fit(draw, text, font, box_width)
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    line_spacing = int(min_font_size * 0.2)
    return font, lines, line_heights, line_spacing


def compute_expansion_direction(x1, y1, x2, y2, img_width, img_height):
    space_left = x1
    space_right = img_width - x2
    space_top = y1
    space_bottom = img_height - y2

    horizontal_space = max(space_left, space_right)
    vertical_space = max(space_top, space_bottom)

    if horizontal_space >= vertical_space:
        return "horizontal", space_left, space_right
    else:
        return "vertical", space_top, space_bottom


def expand_box(x1, y1, x2, y2, img_width, img_height, needed_width, needed_height):
    direction, space_a, space_b = compute_expansion_direction(x1, y1, x2, y2, img_width, img_height)
    current_width = x2 - x1
    current_height = y2 - y1

    if direction == "horizontal":
        extra_needed = max(0, needed_width - current_width)
        if space_a >= space_b:
            new_x1 = max(0, x1 - extra_needed)
            new_x2 = x2
        else:
            new_x1 = x1
            new_x2 = min(img_width, x2 + extra_needed)
        return new_x1, y1, new_x2, y2
    else:
        extra_needed = max(0, needed_height - current_height)
        if space_a >= space_b:
            new_y1 = max(0, y1 - extra_needed)
            new_y2 = y2
        else:
            new_y1 = y1
            new_y2 = min(img_height, y2 + extra_needed)
        return x1, new_y1, x2, new_y2
