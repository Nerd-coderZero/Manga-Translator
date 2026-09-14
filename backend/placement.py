import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline_core import (
    get_dominant_text_color,
    ensure_readable_text_color,
    fit_text_in_box,
    wrap_text_to_fit,
    expand_box,
    compute_expansion_direction,
    find_available_font,
)

FONT_PATH = find_available_font()


def erase_and_paste_text(image_path, merged_boxes, translations, output_path,
                          min_acceptable_font_size=10, max_area_multiplier=2.5,
                          max_chars_per_source_char=6):
    img = Image.open(image_path).convert("RGB")
    img_np_original = np.array(img)
    draw = ImageDraw.Draw(img)

    font_path = FONT_PATH
    img_height, img_width = img_np_original.shape[:2]

    skipped_boxes = []

    for box, translated in zip(merged_boxes, translations):
        if not translated:
            continue

        source_len = max(len(box["text"]), 1)
        if len(translated) > source_len * max_chars_per_source_char and len(translated) > 60:
            print(f"skipped box at {box['bounds']}: translated output implausibly long "
                  f"({len(translated)} chars for {source_len}-char source), likely not a "
                  f"clean translation. left untranslated.")
            skipped_boxes.append({"bounds": box["bounds"], "text": box["text"], "reason": "implausible_length"})
            continue

        x1, y1, x2, y2 = box["bounds"]
        box_width = x2 - x1
        box_height = y2 - y1
        original_area = box.get("original_area", box_width * box_height)

        edge_pixels = np.concatenate([
            img_np_original[y1:y2, max(0, x1-3):x1].reshape(-1, 3) if x1 > 3 else np.array([[255, 255, 255]]),
            img_np_original[y1:y2, x2:x2+3].reshape(-1, 3) if x2+3 < img_width else np.array([[255, 255, 255]])
        ])
        bg_color = tuple(int(v) for v in np.median(edge_pixels, axis=0))

        text_color = get_dominant_text_color(img_np_original, x1, y1, x2, y2, bg_color)
        text_color = ensure_readable_text_color(text_color, bg_color)

        original_height = box.get("avg_original_height", box_height)
        target_font_size = max(int(original_height * 0.85), min_acceptable_font_size)
        ceiling_font_size = min(box_height, box_width)
        max_font_size = min(target_font_size, ceiling_font_size)
        max_font_size = max(max_font_size, min_acceptable_font_size)

        font, lines, line_heights, line_spacing = fit_text_in_box(
            draw, translated, box_width, box_height, font_path, max_font_size, min_acceptable_font_size
        )

        fits_at_readable_size = font.size >= min_acceptable_font_size and \
            all((draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]) <= box_width for line in lines)

        if fits_at_readable_size:
            total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0
            max_line_width = max(
                (draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0
            )
            width_slack = box_width - max_line_width
            height_slack = box_height - total_text_height

            slack_ratio_width = width_slack / box_width if box_width else 0
            slack_ratio_height = height_slack / box_height if box_height else 0

            if slack_ratio_width > 0.35 and slack_ratio_height > 0.35:
                grow_ceiling = min(int(max_font_size * 1.3), ceiling_font_size)
                grown_font, grown_lines, grown_heights, grown_spacing = fit_text_in_box(
                    draw, translated, box_width, box_height, font_path, grow_ceiling, font.size
                )
                if grown_font.size > font.size:
                    font, lines, line_heights, line_spacing = grown_font, grown_lines, grown_heights, grown_spacing

        final_x1, final_y1, final_x2, final_y2 = x1, y1, x2, y2
        final_box_width, final_box_height = box_width, box_height

        if not fits_at_readable_size:
            max_area = original_area * max_area_multiplier

            test_width = box_width
            test_height = box_height
            needed_font = min_acceptable_font_size
            for candidate_font_size in range(min_acceptable_font_size, ceiling_font_size + 20):
                try:
                    candidate_font = ImageFont.truetype(font_path, candidate_font_size)
                except:
                    break
                candidate_lines = wrap_text_to_fit(draw, translated, candidate_font, box_width * 2)
                widths = [draw.textbbox((0, 0), line, font=candidate_font)[2] for line in candidate_lines]
                needed_width_estimate = max(widths) if widths else box_width
                needed_height_estimate = len(candidate_lines) * (candidate_font_size + int(candidate_font_size * 0.2))
                if needed_width_estimate <= box_width * 2 and needed_height_estimate <= box_height * 2:
                    needed_font = candidate_font_size
                    test_width = needed_width_estimate
                    test_height = needed_height_estimate

            expanded = expand_box(x1, y1, x2, y2, img_width, img_height, test_width, test_height)
            expanded_area = (expanded[2] - expanded[0]) * (expanded[3] - expanded[1])

            if expanded_area <= max_area:
                final_x1, final_y1, final_x2, final_y2 = expanded
                final_box_width = final_x2 - final_x1
                final_box_height = final_y2 - final_y1
                font, lines, line_heights, line_spacing = fit_text_in_box(
                    draw, translated, final_box_width, final_box_height, font_path,
                    min(needed_font, ceiling_font_size + 20), min_acceptable_font_size
                )
            else:
                print(f"skipped box at ({x1},{y1})-({x2},{y2}): translated text "
                      f"needs ~{expanded_area}px area, exceeds {max_area_multiplier}x "
                      f"original area cap ({original_area}px). left untranslated.")
                skipped_boxes.append({"bounds": box["bounds"], "text": box["text"], "reason": "too_large"})
                continue

        draw.rectangle([final_x1, final_y1, final_x2, final_y2], fill=bg_color)

        total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0
        current_y = final_y1 + (final_box_height - total_text_height) // 2

        for line, line_height in zip(lines, line_heights):
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = final_x1 + (final_box_width - line_width) // 2
            draw.text((line_x, current_y), line, fill=text_color, font=font)
            current_y += line_height + line_spacing

    img.save(output_path)
    return img, skipped_boxes
