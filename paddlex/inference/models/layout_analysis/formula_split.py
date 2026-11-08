# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conservative, image-only splitting of displayed formula regions.

This finds separated ink rows, not semantic equation boundaries. Shared tall
delimiters block cuts; ambiguous/dense regions are deliberately left intact.
"""

import numpy as np

from ....utils.deps import function_requires_deps, is_dep_available

if is_dep_available("opencv-contrib-python"):
    import cv2


def _formula_ink(gray):
    # Estimate the paper illumination before Otsu: a global threshold alone
    # mistakes the darker side of a shaded scan for foreground. Closing fills
    # thin dark strokes without averaging them into the local background.
    k = max(15, (min(gray.shape) // 8) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    gray = cv2.divide(gray, np.maximum(background, 1), scale=255)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _line_rectangles(gray, min_gap_ratio=0.6):
    """Return crop-local line rectangles, or None when no safe split is found."""
    if min(gray.shape) < 8 or np.ptp(gray) < 20:
        return None
    ink = _formula_ink(gray)
    # Dark backgrounds and densely filled regions are outside this heuristic.
    if np.count_nonzero(ink) > 0.45 * ink.size:
        return None
    _, components, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    stats = stats[1:]
    if len(stats) < 4:
        return None
    widths, heights, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    chars = (heights >= 3) & (widths <= 3 * heights) & (areas >= 4)
    if np.count_nonzero(chars) < 4:
        return None
    char_height = float(np.median(heights[chars]))
    # Only discard isolated speckles when finding gaps. All binarized ink is used
    # below to bound the children, preserving punctuation and small scripts.
    noise_area = max(1, int(char_height * char_height * 0.04))
    retained = np.r_[False, areas > noise_area]
    occupied = retained[components].any(axis=1)
    transitions = np.diff(np.r_[False, ~occupied, False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    cuts = [
        (int(a) + int(b)) // 2
        for a, b in zip(starts, ends)
        if a > 0 and b < len(occupied) and b - a >= max(3, min_gap_ratio * char_height)
    ]
    # A short fraction bar can leave a gap large enough to mimic a line gap.
    bars = stats[
        (widths >= 0.8 * char_height)
        & (heights <= max(3, 0.35 * char_height))
        & (widths >= 2 * heights)
    ]
    # Numerators/denominators need character-height components. Other short
    # horizontal strokes (e.g. equals signs on adjacent dense rows) must not
    # make an equals sign look like a fraction bar.
    fraction_ink = np.r_[False, heights >= max(4, 0.5 * char_height)][components]
    protected = []
    reach = max(2, int(round(1.5 * char_height)))
    for x, y, w, h, area in bars:
        above = fraction_ink[max(0, y - reach) : y, x : x + w]
        below = fraction_ink[y + h : y + h + reach, x : x + w]
        if (
            min(np.count_nonzero(above), np.count_nonzero(below))
            >= 0.15 * char_height**2
        ):
            protected.append((y - 0.75 * char_height, y + h + 0.75 * char_height))
    cuts = [c for c in cuts if not any(a < c < b for a, b in protected)]
    if not cuts:
        return None
    boundaries = [0, *cuts, gray.shape[0]]
    # Attach small detached limits/numbers to their nearest body. A satellite
    # should not veto valid splits elsewhere in a long derivation.
    while len(boundaries) > 2:
        regions = []
        for a, b in zip(boundaries, boundaries[1:]):
            ys, xs = np.nonzero(ink[a:b])
            if not len(xs):
                return None
            regions.append((a + ys.min(), a + ys.max() + 1, len(xs)))
        invalid = [
            i
            for i, (a, b, n) in enumerate(regions)
            if b - a < 1.2 * char_height or n < char_height**2
        ]
        if not invalid:
            break
        i = invalid[0]
        if i == 0:
            remove = 1
        elif i == len(regions) - 1:
            remove = i
        else:
            remove = (
                i
                if regions[i][0] - regions[i - 1][1]
                <= regions[i + 1][0] - regions[i][1]
                else i + 1
            )
        boundaries.pop(remove)
    if len(boundaries) < 3:
        return None
    rects = []
    pad = max(1, int(round(char_height * 0.12)))
    for a, b in zip(boundaries, boundaries[1:]):
        ys, xs = np.nonzero(ink[a:b])
        if not len(xs):
            return None
        # Do not emit detached accents, summation limits, or punctuation rows.
        if ys.max() - ys.min() + 1 < 1.2 * char_height or len(xs) < char_height**2:
            return None
        rects.append(
            [
                max(0, int(xs.min()) - pad),
                max(a, a + int(ys.min()) - pad),
                min(gray.shape[1], int(xs.max()) + 1 + pad),
                min(b, a + int(ys.max()) + 1 + pad),
            ]
        )
    return rects


def _candidate_lines(gray, transform, parent, candidates):
    """Use a complete, stronger detector row set, with image validation."""
    height, width = gray.shape
    selected = []
    for candidate in candidates:
        if candidate["label"] != "display_formula":
            continue
        if candidate["score"] < max(0.5, parent["score"] + 0.05):
            continue
        x1, y1, x2, y2 = candidate["coordinate"]
        shape = candidate.get("quad", candidate.get("polygon_points"))
        quad = np.asarray(
            shape if shape is not None else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            continue
        if not cv2.isContourConvex(quad):
            continue
        local = cv2.perspectiveTransform(quad[None], transform)[0]
        lo, hi = local.min(axis=0), local.max(axis=0)
        # Allow rounding at the parent edges, but no neighboring-column boxes,
        # same-size duplicates, or nested symbols mistaken for entire rows.
        if np.any(lo < -3) or np.any(hi > [width + 3, height + 3]):
            continue
        if hi[1] - lo[1] > 0.65 * height or hi[0] - lo[0] < 0.25 * width:
            continue
        selected.append((float(local[:, 1].mean()), candidate, local))
    if not 2 <= len(selected) <= 16:
        return None
    selected.sort(key=lambda item: item[0])
    # Detector agreement permits narrower gaps, while the same fraction-bar,
    # shared-delimiter and satellite checks still protect mathematical structure.
    rows = _line_rectangles(gray, min_gap_ratio=0.25)
    if rows is None or len(rows) != len(selected):
        return None
    ink = _formula_ink(gray) > 0
    covered = np.zeros(gray.shape, np.uint8)
    children = []
    previous_bottom = None
    for (left, top, right, bottom), (_, candidate, local) in zip(rows, selected):
        lo, hi = local.min(axis=0), local.max(axis=0)
        if not top <= local[:, 1].mean() < bottom:
            return None
        # A slight boundary overlap is normal for detected line boxes; a
        # substantial one indicates competing/nested candidates, not separate rows.
        if previous_bottom is not None and previous_bottom - lo[1] > 0.15 * (
            hi[1] - lo[1]
        ):
            return None
        previous_bottom = hi[1]
        region = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(region, [np.round(local).astype(np.int32)], 1)
        row_ink = ink[top:bottom, left:right]
        if np.count_nonzero(
            row_ink & (region[top:bottom, left:right] > 0)
        ) < 0.98 * np.count_nonzero(row_ink):
            return None
        covered |= region
        child = parent.copy()
        child.update(candidate)
        child["order"] = parent.get("order")
        if "polygon_points" in parent and "quad" in child:
            child["polygon_points"] = child["quad"]
        children.append(child)
    if np.count_nonzero(ink & (covered > 0)) < 0.98 * np.count_nonzero(ink):
        return None
    return children


@function_requires_deps("opencv-contrib-python")
def split_formula_boxes(image, boxes, candidates=None):
    """Replace separable display formulas with line boxes in reading order.

    ``image`` is the original uint8 page (BGR or grayscale). The original list
    and dictionaries are not mutated. Call after overlap/containment filtering
    and before assigning final order indices. Image-derived child scores retain
    the parent detection scores; they are not confidence estimates for the split.
    Optional ``candidates`` are display-formula detections after threshold/NMS
    but before containment filtering; accepted detector children keep their own
    scores. Neither input list is mutated.
    """
    page_h, page_w = image.shape[:2]
    if candidates:
        # Restore only suppressed detections. A caller can disable containment
        # or overlap filtering, in which case row boxes may already be present.
        def geometry_key(box):
            shape = box.get("quad", box.get("polygon_points", box["coordinate"]))
            return box["label"], tuple(np.asarray(shape).ravel())

        present = {geometry_key(box) for box in boxes}
        candidates = [box for box in candidates if geometry_key(box) not in present]
    result = []
    for box in boxes:
        if box["label"] != "display_formula":
            result.append(box)
            continue
        x1, y1, x2, y2 = box["coordinate"]
        # Irregular polygons may hide internal content; leave them intact.
        shape = box.get("quad", box.get("polygon_points"))
        if shape is not None and np.asarray(shape).shape != (4, 2):
            result.append(box)
            continue
        quad = np.asarray(
            shape if shape is not None else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        if not np.isfinite(quad).all():
            result.append(box)
            continue
        # V4 quads already carry ordered TL, TR, BR, BL vertices. Reordering by
        # min(x+y) can rotate a wide, perspective-skewed formula by 90 degrees
        # and cause cuts between symbols instead of between formula rows.
        if "quad" not in box and shape is not None:
            # Arbitrarily ordered segmentation polygons do not encode text
            # orientation. Only normalize axis-aligned rectangles here.
            rect = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            distances = np.linalg.norm(quad[:, None] - rect[None], axis=2)
            if not np.all(distances.min(axis=0) <= 2):
                result.append(box)
                continue
            quad = rect
        if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 64:
            result.append(box)
            continue
        width = int(
            round(
                max(
                    np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
                )
            )
        )
        height = int(
            round(
                max(
                    np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
                )
            )
        )
        if min(width, height) < 8 or width * height > 4 * page_h * page_w:
            result.append(box)
            continue
        dst = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        transform = cv2.getPerspectiveTransform(quad, dst)
        crop = cv2.warpPerspective(
            image, transform, (width, height), borderValue=(255, 255, 255)
        )
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        detected = _candidate_lines(gray, transform, box, candidates or [])
        if detected is not None:
            result.extend(detected)
            continue
        rects = _line_rectangles(gray)
        if rects is None:
            result.append(box)
            continue
        inverse = cv2.getPerspectiveTransform(dst, quad)
        children = []
        for left, top, right, bottom in rects:
            local = np.float32(
                [
                    [
                        [left, top],
                        [right - 1, top],
                        [right - 1, bottom - 1],
                        [left, bottom - 1],
                    ]
                ]
            )
            points = cv2.perspectiveTransform(local, inverse)[0]
            points[:, 0] = np.clip(points[:, 0], 0, page_w)
            points[:, 1] = np.clip(points[:, 1], 0, page_h)
            child = box.copy()
            child["coordinate"] = [
                *np.floor(points.min(axis=0)).astype(int).tolist(),
                *np.ceil(points.max(axis=0)).astype(int).tolist(),
            ]
            if "quad" in child:
                child["quad"] = points.tolist()
            if "polygon_points" in child:
                child["polygon_points"] = points.tolist()
            children.append(child)
        result.extend(children)
    return result
