# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import base64
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests

from ....modules.object_detection.model_list import LAYOUTCLIENT_MODELS as MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ..base import BasePredictor
from .processors import DetPostProcess, ReadImage
from .result import DetResult


class DetClinetPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self,
        *args,
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        """Initializes DetPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            img_size (Optional[Union[int, Tuple[int, int]]], optional): The input image size (w, h). Defaults to None.
            threshold (Optional[float], optional): The threshold for filtering out low-confidence predictions.
                Defaults to None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        if layout_unclip_ratio is not None:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), f"The length of `layout_unclip_ratio` should be 2."
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or Dict, but got {type(layout_unclip_ratio)}."
                )

        if layout_merge_bboxes_mode is not None:
            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'] or a dict, but got {layout_merge_bboxes_mode}"
        self.img_size = None
        self.threshold = threshold
        self.layout_nms = layout_nms
        self.layout_unclip_ratio = layout_unclip_ratio
        self.layout_merge_bboxes_mode = layout_merge_bboxes_mode

        response = requests.get(
            f"{self.server_url}/categories",
        )
        categories_dict = response.json()["categories"]
        custom_map = {
            # 第一组 <-> 第二组
            "title": "paragraph_title",
            "equation": "display_formula",
            "code": "algorithm",
            "list_item": "text",
            "table_caption": "figure_title",
            "image_caption": "figure_title",
            "code_caption": "paragraph_title",
            "table_footnote": "vision_footnote",
            "image_footnote": "vision_footnote",
            "page_number": "number",
            "page_footnote": "footnote",
            "image_block": "image",
            "equation_block": "display_formula",
            "ref_text": "reference_content",
            "phonetic": "aside_text",
            "list": "text",
            "unknown": "aside_text",
            # 第三组 <-> 第四组
            "title": "paragraph_title",
            "plain_text": "text",
            "abandon": "aside_text",
            "figure": "image",
            "figure_caption": "figure_title",
            "table_caption": "figure_title",
            "isolate_formula": "display_formula",
            "formula_caption": "formula_number",
        }
        
        # Convert dict like {'0': 'title', '1': 'plain_text', ...} to list ['title', 'plain_text', ...]
        self.labels = [
            custom_map.get(categories_dict[str(i)], categories_dict[str(i)])
            for i in range(len(categories_dict))
        ]
        self.labels_to_id = {v: k for k, v in enumerate(self.labels)}

        self.post_op = DetPostProcess(labels=self.labels)

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return DetResult

    def _format_output(self, pred: Union[Sequence[Any], dict]) -> List[dict]:
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Union[Sequence[Any], dict]): The input predictions, which can be either:
                - A dict from server response with format:
                  {"success": bool, "count": int, "data": [{"category": str, "bbox": {...}, "score": float, "category_id": int}]}
                - A list of 3 or 4 elements for backward compatibility:
                  - When len(pred) == 4: [boxes, class_ids, scores, masks] (SOLOv2)
                  - When len(pred) == 3: [boxes, box_nums, masks] (Instance Segmentation)
                  - When len(pred) == 2: [boxes, box_nums] (Object Detection)

        Returns:
            List[dict]: A list of dictionaries, each containing either 'class_id' and 'masks' (for SOLOv2),
                or 'boxes' and 'masks' (for Instance Segmentation), or just 'boxes' if no masks are provided.
        """
        # Handle server response format (dict)
        if isinstance(pred, dict):
            if not pred.get("success", False):
                return [{"boxes": np.array([])}]

            data_list = pred.get("data", [])
            if not data_list:
                return [{"boxes": np.array([])}]

            # Convert server response to boxes format: [cls_id, score, xmin, ymin, xmax, ymax]
            boxes_list = []
            for item in data_list:
                cls_id = item.get("class_id", 0)
                if cls_id == -1:
                    cls_id = self.labels_to_id.get(item.get("category", "0"), 0)
                score = item.get("score", 0.0)
                if score == -1:
                    score = 1.0
                bbox = item.get("bbox", {})
                xmin = bbox.get("x1", 0)
                ymin = bbox.get("y1", 0)
                xmax = bbox.get("x2", 0)
                ymax = bbox.get("y2", 0)
                boxes_list.append([cls_id, score, xmin, ymin, xmax, ymax])

            if boxes_list:
                boxes_array = np.array(boxes_list, dtype=np.float32)
            else:
                boxes_array = np.array([])

            return [{"boxes": boxes_array}]

        # Handle original sequence format (backward compatibility)
        box_idx_start = 0
        pred_box = []

        if len(pred) == 4:
            # Adapt to SOLOv2
            pred_class_id = []
            pred_mask = []
            pred_class_id.append([pred[1], pred[2]])
            pred_mask.append(pred[3])
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]

        if len(pred) == 3:
            # Adapt to Instance Segmentation
            pred_mask = []
        for idx in range(len(pred[1])):
            np_boxes_num = pred[1][idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = pred[0][box_idx_start:box_idx_end]
            pred_box.append(np_boxes)
            if len(pred) == 3:
                np_masks = pred[2][box_idx_start:box_idx_end]
                pred_mask.append(np_masks)
            box_idx_start = box_idx_end

        if len(pred) == 3:
            return [
                {"boxes": np.asarray(pred_box[i]), "masks": np.asarray(pred_mask[i])}
                for i in range(len(pred_box))
            ]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: bool = False,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        **kwargs: Any,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            threshold (Optional[float, dict], optional): The threshold for filtering out low-confidence predictions.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to None.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        datas = batch_data.instances

        image_reader = ReadImage(format="RGB")
        datas = image_reader(datas)
        batch_preds = {}
        for data in datas:
            image_base64 = array_to_base64(data["img"])

            response = requests.post(
                f"{self.server_url}/predict",
                json={
                    "image_base64": image_base64,
                    "return_format": "detailed",  # 或 "simple"
                },
            )

            result = response.json()

            if result["success"]:
                batch_preds = result

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)
        # postprocess
        boxes = self.post_op(
            preds_list,
            datas,
            threshold=threshold if threshold is not None else self.threshold,
            layout_nms=layout_nms or self.layout_nms,
            layout_unclip_ratio=layout_unclip_ratio or self.layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
            or self.layout_merge_bboxes_mode,
        )

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }


def array_to_base64(img_array, fmt="PNG"):
    import numpy as np
    import base64
    import io
    from PIL import Image

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
