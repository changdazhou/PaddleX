# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import os
import shutil
import time
import numpy as np
from ...results import *
from ...components import *
from ..ocr import OCRPipeline
from ....utils import logging
from ..ppchatocrv3.utils import *
from ..base import BasePipeline
from ..table_recognition.utils import (
    convert_4point2rect,
    get_ori_coordinate_for_table,
    TableMatch,
)
from ....utils.cache import TEMP_DIR


class LayoutParsingPipeline(BasePipeline):
    """Layout Analysis Pileline"""

    entities = "layout_parsing"

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        layout_batch_size=1,
        text_det_batch_size=1,
        text_rec_batch_size=1,
        recovery=True,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(
            device,
            predictor_kwargs,
        )
        self._build_predictor(
            layout_model=layout_model,
            text_det_model=text_det_model,
            text_rec_model=text_rec_model,
        )
        self.set_predictor(
            layout_batch_size=layout_batch_size,
            text_det_batch_size=text_det_batch_size,
            text_rec_batch_size=text_rec_batch_size,
        )
        self.recovery = recovery

    def _build_predictor(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
    ):
        self.layout_predictor = self._create(model=layout_model)
        self.ocr_pipeline = self._create(
            pipeline=OCRPipeline,
            text_det_model=text_det_model,
            text_rec_model=text_rec_model,
        )
        self._crop_by_boxes = CropByBoxes()
        self._match = TableMatch(filter_ocr_result=False)
        self.img_reader = ReadImage(format="BGR")

    def set_predictor(
        self,
        layout_batch_size=None,
        text_det_batch_size=None,
        text_rec_batch_size=None,
        device=None,
    ):
        if text_det_batch_size and text_det_batch_size > 1:
            logging.warning(
                f"text det model only support batch_size=1 now,the setting of text_det_batch_size={text_det_batch_size} will not using! "
            )
        if layout_batch_size:
            self.layout_predictor.set_predictor(batch_size=layout_batch_size)
        if text_rec_batch_size:
            self.ocr_pipeline.text_rec_model.set_predictor(
                batch_size=text_rec_batch_size
            )

        if device:
            self.layout_predictor.set_predictor(device=device)
            self.ocr_pipeline.set_predictor(device=device)

    def predict(
        self,
        inputs,
        recovery=True,
        **kwargs,
    ):
        self.set_predictor(**kwargs)
        # get oricls and uvdoc results
        img_info_list = list(self.img_reader(inputs))[0]
        img_list = [img_info["img"] for img_info in img_info_list]
        for idx, (img_info, layout_pred) in enumerate(
            zip(img_info_list, self.layout_predictor(img_list))
        ):
            page_id = idx
            single_img_res = {
                "input_path": "",
                "layout_result": DetResult({}),
                "ocr_result": OCRResult({}),
                "layout_parsing_result": {},
            }
            # update layout result
            single_img_res["input_path"] = layout_pred["input_path"]
            single_img_res["layout_result"] = layout_pred
            single_img = img_info["img"]
            structure_res = []
            ocr_res_with_layout = []
            if len(layout_pred["boxes"]) > 0:
                subs_of_img = list(self._crop_by_boxes(layout_pred))
                layout_pred.pop("ori_img")
                # get cropped images
                for sub in subs_of_img:
                    box = sub["box"]
                    xmin, ymin, xmax, ymax = [int(i) for i in box]
                    mask_flag = True
                    if self.recovery and recovery:
                        # TODO: Why use the entire image?
                        wht_im = np.ones(single_img.shape, dtype=single_img.dtype) * 255
                        wht_im[ymin:ymax, xmin:xmax, :] = sub["img"]
                        sub_ocr_res = get_ocr_res(self.ocr_pipeline, wht_im)
                    else:
                        sub_ocr_res = get_ocr_res(self.ocr_pipeline, sub)
                        sub_ocr_res["dt_polys"] = get_ori_coordinate_for_table(
                            xmin, ymin, sub_ocr_res["dt_polys"]
                        )
                    sub_ocr_res.pop("ori_img")
                    layout_label = sub["label"].lower()
                    # Adapt the user label definition to specify behavior.
                    if not sub["label"].lower() in [
                        "image",
                        "figure",
                        "img",
                        "fig",
                    ]:
                        ocr_res_with_layout.append(sub_ocr_res)
                        structure_res.append(
                            {
                                "layout_bbox": box,
                                f"{layout_label}": "\n".join(sub_ocr_res["rec_text"]),
                            }
                        )
                    if mask_flag:
                        single_img[ymin:ymax, xmin:xmax, :] = 255

            use_ocr_without_layout = kwargs.get("use_ocr_without_layout", True)
            ocr_res = {
                "dt_polys": [],
                "rec_text": [],
            }

            if use_ocr_without_layout:
                ocr_res = get_ocr_res(self.ocr_pipeline, single_img)
                ocr_res.pop("ori_img")
                for idx, single_dt_poly in enumerate(ocr_res["dt_polys"]):
                    structure_res.append(
                        {
                            "layout_bbox": convert_4point2rect(single_dt_poly),
                            "text_without_layout": ocr_res["rec_text"][idx],
                        }
                    )
            # update ocr result
            for layout_ocr_res in ocr_res_with_layout:
                ocr_res["dt_polys"].extend(layout_ocr_res["dt_polys"])
                ocr_res["rec_text"].extend(layout_ocr_res["rec_text"])
                ocr_res["rec_score"].extend(layout_ocr_res["rec_score"])

            # sort the layout result by the left top point of the box
            structure_res = sorted_layout_boxes(structure_res, w=single_img.shape[1])
            structure_res = LayoutParsingResult(
                {
                    "parsing_result": structure_res,
                }
            )
            single_img_res["ocr_result"] = ocr_res
            single_img_res["layout_parsing_result"] = structure_res
            single_img_res["layout_parsing_result"]["page_id"] = page_id + 1

            yield VisualResult(single_img_res, page_id, inputs)
