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

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from paddlex.inference.models.layout_analysis.formula_split import split_formula_boxes
from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess


def _page():
    image = np.full((220, 360, 3), 255, np.uint8)
    for baseline in [45, 105, 165]:
        cv2.putText(
            image,
            "x + y = 123",
            (35, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[20, 15, 300, 185],
        quad=[[20, 15], [300, 15], [300, 185], [20, 185]],
        order=1,
    )
    return image, box


def test_separated_rows_preserve_pixels_metadata_and_input():
    image, box = _page()
    original = copy.deepcopy(box)
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    assert box == original
    covered = np.zeros(image.shape[:2], np.uint8)
    for child in rows:
        assert child["score"] == box["score"]
        assert child["cls_id"] == box["cls_id"]
        x1, y1, x2, y2 = child["coordinate"]
        covered[y1 : y2 + 1, x1 : x2 + 1] = 1
    assert np.all(covered[image[:, :, 0] == 0])
    assert all(a["coordinate"][3] < b["coordinate"][1] for a, b in zip(rows, rows[1:]))


@pytest.mark.parametrize("label", ["text", "inline_formula", "table", "formula_number"])
def test_non_display_labels_are_untouched(label):
    image, box = _page()
    box["label"] = label
    assert split_formula_boxes(image, [box]) == [box]


def test_shared_matrix_bracket_prevents_splitting():
    image, box = _page()
    cv2.line(image, (25, 20), (25, 180), (0, 0, 0), 2)
    cv2.line(image, (25, 20), (33, 20), (0, 0, 0), 2)
    cv2.line(image, (25, 180), (33, 180), (0, 0, 0), 2)
    assert split_formula_boxes(image, [box]) == [box]


def test_fraction_and_detached_limit_are_not_split():
    image = np.full((150, 320, 3), 255, np.uint8)
    cv2.putText(image, "x + 123", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.line(image, (55, 78), (200, 78), (0, 0, 0), 2)
    cv2.putText(
        image, "y + 456", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    box = dict(
        label="display_formula", coordinate=[20, 10, 280, 130], score=0.9, cls_id=0
    )
    assert split_formula_boxes(image, [box]) == [box]
    # A detached small upper limit must not become a separate formula.
    cv2.putText(image, "n", (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    assert split_formula_boxes(image, [box]) == [box]


def test_skewed_quads_are_split_and_mapped_back():
    image, box = _page()
    matrix = np.float32([[1, 0.18, 20], [0.08, 1, 15], [0, 0, 1]])
    image = cv2.warpPerspective(image, matrix, (450, 300), borderValue=(255, 255, 255))
    quad = cv2.perspectiveTransform(np.float32([box["quad"]]), matrix)[0]
    box["quad"] = quad.tolist()
    box["polygon_points"] = quad.tolist()
    box["coordinate"] = [*quad.min(0).tolist(), *quad.max(0).tolist()]
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    for row in rows:
        assert row["quad"] == row["polygon_points"]
        local = cv2.perspectiveTransform(
            np.float32([row["quad"]]), np.linalg.inv(matrix)
        )[0]
        assert np.ptp(local[:2, 1]) < 1
        assert np.ptp(local[2:, 1]) < 1


def test_empty_and_invalid_regions_fall_back():
    image, box = _page()
    assert split_formula_boxes(np.full_like(image, 255), [box]) == [box]
    box["quad"] = [[0, 0]] * 4
    assert split_formula_boxes(image, [box]) == [box]


def test_perspective_keeps_model_vertex_order():
    # For a long, thin formula, bottom-left can have smaller x+y than top-left.
    # Reordering that quad would rotate the crop and split the formula by column.
    image = np.full((70, 520, 3), 255, np.uint8)
    cv2.putText(
        image,
        "x + y = 123 + 456 + 789",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    src = np.float32([[0, 0], [519, 0], [519, 69], [0, 69]])
    dst = np.float32([[150, 20], [550, 36], [500, 89], [30, 73]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (600, 120), borderValue=(255, 255, 255))
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[30, 20, 550, 89],
        quad=dst.tolist(),
    )
    assert np.argmin(dst.sum(axis=1)) == 3
    assert split_formula_boxes(warped, [box]) == [box]


def test_postprocessor_switch_and_reading_order():
    image, box = _page()
    raw = dict(
        boxes=np.array([[0, 0.9, *box["coordinate"]]], np.float32),
        quad=np.float32([box["quad"]]).reshape(1, 8),
    )
    post = LayoutAnalysisProcess(labels=["display_formula"], scale_size=[800, 800])
    settings = dict(
        threshold=0.3,
        layout_nms=True,
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
    )
    # Default mode does not need access to pixels.
    off = post([copy.deepcopy(raw)], [dict(ori_img_size=(360, 220))], **settings)[0]
    on = post(
        [copy.deepcopy(raw)],
        [dict(ori_img_size=(360, 220), ori_img=image)],
        layout_split_formula=True,
        **settings
    )[0]
    assert len(off) == 1
    assert len(on) == 3
    assert [b["order"] for b in on] == [1, 2, 3]


@pytest.mark.parametrize("shadow_axis", [0, 1])
@pytest.mark.parametrize("noise", [False, True])
def test_shaded_noisy_rows_keep_their_ink(shadow_axis, noise):
    image, box = _page()
    original_ink = image[:, :, 0] == 0
    shade = np.linspace(0.25, 1.0, image.shape[shadow_axis])
    shade = shade[:, None, None] if shadow_axis == 0 else shade[None, :, None]
    image = (image.astype(float) * shade).astype(np.uint8)
    if noise:
        speckles = np.random.default_rng(42).random(image.shape[:2]) < 0.001
        image[speckles] = 0
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    covered = np.zeros(image.shape[:2], np.uint8)
    for child in rows:
        cv2.fillPoly(covered, [np.round(child["quad"]).astype(np.int32)], 1)
    assert np.all(covered[original_ink])


@pytest.mark.parametrize("gap", [6, 8, 10])
def test_small_fraction_denominators_are_not_separate_rows(gap):
    # Short bars and a large gap beneath them resemble inter-line whitespace.
    # This models the m/q = p/n regression found in a real historical scan.
    image = np.full((90, 120, 3), 255, np.uint8)
    for x, numerator, denominator in [(15, "m", "q"), (75, "p", "n")]:
        cv2.putText(
            image, numerator, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
        cv2.line(image, (x, 34), (x + 15, 34), (0, 0, 0), 1)
        cv2.putText(
            image,
            denominator,
            (x, 34 + gap + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    cv2.putText(image, "=", (43, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    box = dict(label="display_formula", coordinate=[5, 5, 110, 80])
    assert split_formula_boxes(image, [box]) == [box]


def test_detached_small_number_does_not_cancel_other_splits():
    image, box = _page()
    # A small equation-number-like component near the first line belongs to
    # that line, and must neither become a row nor veto the later line gaps.
    cv2.putText(image, "1", (270, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    x1, y1, x2, y2 = rows[0]["coordinate"]
    assert x1 <= 272 <= x2 and y1 <= 60 <= y2


@pytest.mark.parametrize("angle", [0, 15, -15, 30])
def test_resampled_small_scan_fraction_stays_whole(angle):
    # Crop of old_scans_math/4_pg186.pdf in allenai/olmOCR-bench, revision
    # 54a96a6fb6a2bd3b297e59869491db4d3625b711. Rotation/resampling can make a
    # two-pixel fraction bar three pixels thick, or join it to the numerator.
    image = cv2.imread(str(Path(__file__).parent / "assets/formula_fraction_scan.png"))
    h, w = image.shape[:2]
    quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    matrix[:2, 2] += 40
    warped = cv2.warpPerspective(
        image, matrix, (w + 80, h + 80), borderValue=(255, 255, 255)
    )
    quad = cv2.perspectiveTransform(quad[None], matrix)[0]
    box = dict(
        label="display_formula",
        coordinate=[*quad.min(0).tolist(), *quad.max(0).tolist()],
        quad=quad.tolist(),
    )
    assert split_formula_boxes(warped, [box]) == [box]


def _detected_rows():
    import json

    assets = Path(__file__).parent / "assets"
    image = cv2.imread(str(assets / "formula_detected_rows.png"))
    # Crop/detections from old_scans_math/3_pg204 at the same pinned benchmark
    # revision as formula_fraction_scan.png. The model predicts a parent and
    # four stronger rows, with gaps below the image-only splitting threshold.
    boxes = json.loads((assets / "formula_detected_rows.json").read_text())
    parent = max(boxes, key=lambda b: b["coordinate"][3] - b["coordinate"][1])
    rows = sorted(
        [b for b in boxes if b is not parent], key=lambda b: b["coordinate"][1]
    )
    return image, parent, rows


@pytest.mark.parametrize("warp", ["clean", "rotate", "perspective"])
def test_detector_rows_survive_large_filter_with_switch(warp):
    image, parent, rows = _detected_rows()
    if warp != "clean":
        h, w = image.shape[:2]
        if warp == "rotate":
            matrix = np.eye(3, dtype=np.float32)
            matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
            matrix[:2, 2] += 40
        else:
            matrix = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
                np.float32([[50, 20], [w + 30, 35], [w + 10, h + 30], [20, h + 10]]),
            )
        image = cv2.warpPerspective(
            image, matrix, (w + 100, h + 100), borderValue=(255, 255, 255)
        )
        for box in [parent, *rows]:
            quad = cv2.perspectiveTransform(np.float32([box["quad"]]), matrix)[0]
            box["quad"] = quad.tolist()
            box["coordinate"] = [*quad.min(0).tolist(), *quad.max(0).tolist()]
    # Shuffle rows to make sure output follows the parent and visual row order.
    detections = [rows[2], parent, rows[0], rows[3], rows[1]]
    raw = dict(
        boxes=np.array(
            [[0, b["score"], *b["coordinate"]] for b in detections], np.float32
        ),
        quad=np.float32([b["quad"] for b in detections]).reshape(-1, 8),
    )
    post = LayoutAnalysisProcess(labels=["display_formula"], scale_size=[800, 800])
    settings = dict(
        threshold=0.3,
        layout_nms=True,
        layout_merge_bboxes_mode={0: "large"},
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
    )
    h, w = image.shape[:2]
    off = post([copy.deepcopy(raw)], [dict(ori_img_size=(w, h))], **settings)[0]
    on = post(
        [copy.deepcopy(raw)],
        [dict(ori_img_size=(w, h), ori_img=image)],
        layout_split_formula=True,
        **settings
    )[0]
    assert len(off) == 1
    assert len(on) == 4
    assert [b["order"] for b in on] == [1, 2, 3, 4]
    assert [b["score"] for b in on] == pytest.approx([b["score"] for b in rows])
    assert np.allclose([b["quad"] for b in on], [b["quad"] for b in rows])


@pytest.mark.parametrize(
    "invalid",
    ["missing_row", "weak_scores", "overlapping", "cropped_ink", "shared_bracket"],
)
def test_incomplete_or_unsafe_detector_rows_fall_back(invalid):
    image, parent, rows = _detected_rows()
    if invalid == "missing_row":
        rows.pop(1)
    elif invalid == "weak_scores":
        for row in rows:
            row["score"] = parent["score"]
    elif invalid == "overlapping":
        rows.append(copy.deepcopy(rows[0]))
    elif invalid == "cropped_ink":
        for point in rows[1]["quad"][1:3]:
            point[0] -= 30
    else:
        x, y, xx, yy = parent["coordinate"]
        cv2.line(image, (x + 1, y + 2), (x + 1, yy - 2), (0, 0, 0), 2)
    original = copy.deepcopy((parent, rows))
    assert split_formula_boxes(image, [parent], rows) == [parent]
    assert (parent, rows) == original


def test_detector_fraction_candidates_do_not_override_structure_guard():
    image = cv2.imread(str(Path(__file__).parent / "assets/formula_fraction_scan.png"))
    h, w = image.shape[:2]
    parent = dict(
        label="display_formula", cls_id=0, score=0.4, coordinate=[0, 0, w - 1, h - 1]
    )
    rows = [
        dict(label="display_formula", cls_id=0, score=0.9, coordinate=[0, a, w - 1, b])
        for a, b in [(0, 22), (24, h - 1)]
    ]
    assert split_formula_boxes(image, [parent], rows) == [parent]


def test_detector_candidates_already_present_are_not_restored_twice():
    image, parent, rows = _detected_rows()
    # With filtering disabled the caller may intentionally keep both levels.
    boxes = [parent, *rows]
    assert split_formula_boxes(image, boxes, boxes) == boxes
