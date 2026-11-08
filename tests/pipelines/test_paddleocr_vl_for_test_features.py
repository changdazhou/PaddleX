"""Regression tests for for_test-only PaddleOCR-VL options."""

import inspect

from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline
from paddlex.inference.pipelines.paddleocr_vl.utils import convert_otsl_to_html


def _bare_pipeline():
    pipeline = object.__new__(_PaddleOCRVLPipeline)
    pipeline.use_doc_preprocessor = True
    pipeline.use_layout_detection = True
    pipeline.use_chart_recognition = False
    pipeline.use_seal_recognition = False
    pipeline.use_ocr_for_image_block = False
    pipeline.format_block_content = False
    pipeline.merge_layout_blocks = True
    pipeline.use_layout_deform = False
    pipeline.markdown_ignore_labels = []
    return pipeline


def test_predict_exposes_for_test_and_v4_options():
    parameters = inspect.signature(_PaddleOCRVLPipeline.predict).parameters

    assert "layout_gt_dir" in parameters
    assert "use_xycut" in parameters
    assert "table_post_process" in parameters
    assert "use_layout_deform" in parameters


def test_model_settings_keep_xycut_and_v4_deform():
    pipeline = _bare_pipeline()

    settings = pipeline.get_model_settings(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        use_xycut=True,
        use_layout_deform=True,
    )

    assert settings["use_xycut"] is True
    assert settings["use_layout_deform"] is True
    assert settings["use_layout_detection"] is True


def test_otsl_conversion_keeps_for_test_escape_switch():
    otsl = "<fcel>a<b<fcel>c<nl>"

    escaped = convert_otsl_to_html(otsl, escape=True)
    unescaped = convert_otsl_to_html(otsl, escape=False)

    assert "a&lt;b" in escaped
    assert "a<b" in unescaped
