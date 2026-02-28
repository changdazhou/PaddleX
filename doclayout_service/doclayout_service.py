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

import os
import io
import cv2
import base64
import logging
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image, ImageOps
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionBox:
    """检测框数据结构"""

    category: str  # 类别名称
    class_id: int  # 类别ID
    bbox: List[float]  # [x1, y1, x2, y2] 坐标
    score: float  # 置信度分数
    area: float  # 框面积


class LayoutDetectionService(ABC):
    """布局检测服务抽象基类"""

    CATEGORIES: Dict[int, str] = {}

    @abstractmethod
    def predict_base64(self, base64_str: str) -> List[Dict[str, Any]]:
        """
        基于base64编码的图片进行预测

        Args:
            base64_str: base64编码的图片

        Returns:
            检测结果列表
        """
        pass


class MinerU25Service(LayoutDetectionService):
    CATEGORIES = {
        0: "text",  # 文本
        1: "title",  # 段落标题
        2: "table",  # 表格
        3: "equation",  # 公式(独立公式)
        4: "code",  # 代码
        5: "algorithm",  # 算法/伪代码
        6: "aside_text",  # 侧栏文本(装订线等)
        7: "ref_text",  # 参考文献(一条)
        8: "phonetic",  # 注音符号
        9: "list_item",  # 列表项(无序/有序列表)
        10: "table_caption",  # 表格标题
        11: "image_caption",  # 图像标题
        12: "code_caption",  # 代码标题
        13: "table_footnote",  # 表格脚注
        14: "image_footnote",  # 图像脚注
        15: "header",  # 页眉
        16: "footer",  # 页脚
        17: "page_number",  # 页码
        18: "page_footnote",  # 脚注
        19: "image",  # 图像
        20: "chart",
        21: "list",  # 列表块(无序/有序列表)
        22: "image_block",  # 图像块(多图)
        23: "equation_block",  # 公式块(多行公式)
        24: "unknown",  # 未知块
    }
    CATEGORIES = {
        0: "text",  # 文本
        1: "paragraph_title",  # 段落标题
        2: "table",  # 表格
        3: "display_formula",  # 公式(独立公式)
        4: "algorithm",  # 代码
        5: "algorithm",  # 算法/伪代码
        6: "aside_text",  # 侧栏文本(装订线等)
        7: "reference_content",  # 参考文献(一条)
        8: "aside_text",  # 注音符号
        9: "text",  # 列表项(无序/有序列表)
        10: "figure_title",  # 表格标题
        11: "figure_title",  # 图像标题
        12: "paragraph_title",  # 代码标题
        13: "vision_footnote",  # 表格脚注
        14: "vision_footnote",  # 图像脚注
        15: "header",  # 页眉
        16: "footer",  # 页脚
        17: "number",  # 页码
        18: "footnote",  # 脚注
        19: "image",  # 图像
        20: "chart",
        21: "text",  # 列表块(无序/有序列表)
        22: "image",  # 图像块(多图)
        23: "display_formula",  # 公式块(多行公式)
        24: "aside_text",  # 未知块
    }

    def __init__(
        self,
        server_url: Optional[str] = "http://127.0.0.1:30000",
    ):
        from mineru.backend.vlm.vlm_analyze import ModelSingleton

        self.model = ModelSingleton().get_model("http-client", None, server_url)

    def predict_base64(self, base64_str: str) -> List[Dict[str, Any]]:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]  # 去除data:image前缀

        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))

        return self.predict(image)

    def predict(self, image: Image) -> List[Dict[str, Any]]:
        # preprocess
        image = ImageOps.exif_transpose(image) or image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # inference
        results = self.model.layout_detect(image=image)

        # postprocess
        width, height = image.size

        new_results = []
        for block in results:
            x1, y1, x2, y2 = block.bbox
            new_results.append(
                {
                    "category": block.type,
                    "class_id": -1,
                    "bbox": {
                        "x1": round(x1 * width),
                        "y1": round(y1 * height),
                        "x2": round(x2 * width),
                        "y2": round(y2 * height),
                    },
                    "score": -1,
                    "area": -1,
                    "angle": block.angle,
                }
            )
        return new_results


class DocLayoutYOLOService(LayoutDetectionService):
    """
    DocLayout-YOLO 服务封装类
    从MinerU的 DocLayoutYOLOModel 简化提取
    """

    # DocLayout-YOLO类别定义（来自MinerU配置）
    CATEGORIES = {
        0: "title",  # 标题
        1: "plain_text",  # 正文
        2: "abandon",  # 废弃/页眉页脚
        3: "figure",  # 图片
        4: "figure_caption",  # 图片标题
        5: "table",  # 表格
        6: "table_caption",  # 表格标题
        7: "isolate_formula",  # 独立公式
        8: "formula_caption",  # 公式标题
        9: "inline_formula",  # 行内公式
    }
    CATEGORIES = {
        0: "paragraph_title",  # 标题
        1: "text",  # 正文
        2: "aside_text",  # 废弃/页眉页脚
        3: "image",  # 图片
        4: "figure_title",  # 图片标题
        5: "table",  # 表格
        6: "figure_title",  # 表格标题
        7: "display_formula",  # 独立公式
        8: "formula_number",  # 公式标题
        9: "inline_formula",  # 行内公式
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        imgsz: int = 1024,
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.45,
    ):
        """
        初始化DocLayout-YOLO模型

        Args:
            model_path: 模型权重路径，None则自动下载
            device: 运行设备
            imgsz: 输入图像尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 延迟导入，避免启动时加载
        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            raise ImportError("请安装doclayout-yolo: pip install doclayout-yolo")

        # 模型初始化
        if model_path is None:
            # 自动下载模型（使用MinerU默认模型路径逻辑）
            model_path = self._auto_download_model()

        logger.info(f"正在加载DocLayout-YOLO模型: {model_path}")
        self.model = YOLOv10(model_path)
        logger.info(f"模型加载完成，设备: {device}")

    def _auto_download_model(self) -> str:
        """
        自动下载模型（参考MinerU的模型下载逻辑）
        """
        try:
            from modelscope import snapshot_download

            cache_dir = os.environ.get("MODELSCOPE_CACHE", "./models")
            model_dir = snapshot_download(
                "opendatalab/DocLayout-YOLO", cache_dir=cache_dir
            )
            # 查找最佳模型文件
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".pt") or file.endswith(".pth"):
                        return os.path.join(root, file)
            raise FileNotFoundError("未找到模型权重文件")
        except Exception as e:
            logger.error(f"自动下载失败: {e}")
            raise RuntimeError("请手动指定模型路径或安装modelscope")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理（参考MinerU的处理逻辑）

        Args:
            image: BGR格式的numpy数组 (H, W, C)

        Returns:
            预处理后的图像
        """
        # DocLayout-YOLO内部会自动处理resize和归一化
        # 这里仅做基本的格式检查
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("输入必须是3通道BGR图像")
        return image

    def postprocess(self, results) -> List[DetectionBox]:
        """
        后处理：解析YOLO输出为结构化数据

        Args:
            results: YOLOv10预测结果

        Returns:
            检测框列表
        """
        detection_boxes = []

        # 获取第一个图像的结果（batch=1）
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detection_boxes

        # 提取检测信息
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] 坐标
        confs = result.boxes.conf.cpu().numpy()  # [N] 置信度
        clses = result.boxes.cls.cpu().numpy().astype(int)  # [N] 类别ID

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(confs[i])
            cls_id = int(clses[i])

            # 过滤低置信度
            if conf < self.conf_threshold:
                continue

            # 计算面积
            area = float((x2 - x1) * (y2 - y1))

            detection_boxes.append(
                DetectionBox(
                    category=self.CATEGORIES.get(cls_id, f"class_{cls_id}"),
                    class_id=cls_id,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    score=conf,
                    area=area,
                )
            )

        # 按置信度排序
        detection_boxes.sort(key=lambda x: x.score, reverse=True)
        return detection_boxes

    def predict(self, image: np.ndarray) -> List[DetectionBox]:
        """
        执行预测（完整的推理流程）

        Args:
            image: BGR格式的numpy数组

        Returns:
            检测框列表
        """
        # 预处理
        processed_img = self.preprocess(image)

        # 模型推理（参考MinerU的调用方式）
        results = self.model.predict(
            processed_img,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # 后处理
        return self.postprocess(results)

    def predict_base64(self, base64_str: str) -> List[Dict[str, Any]]:
        """
        从base64字符串预测（服务接口用）

        Args:
            base64_str: base64编码的图片字符串

        Returns:
            字典格式的检测结果
        """
        try:
            # 解码base64
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]  # 去除data:image前缀

            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 转换为BGR格式（OpenCV风格，YOLO常用）
            image_np = np.array(image)
            image_bgr = (
                cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                if "cv2" in globals()
                else image_np
            )

            # 如果没有cv2，手动转换RGB到BGR
            if "cv2" not in globals():
                image_bgr = image_np[:, :, ::-1]

            # 执行预测
            detections = self.predict(image_bgr)

            # 转换为字典列表
            return [
                {
                    "category": det.category,
                    "class_id": det.class_id,
                    "bbox": {
                        "x1": det.bbox[0],
                        "y1": det.bbox[1],
                        "x2": det.bbox[2],
                        "y2": det.bbox[3],
                    },
                    "score": round(det.score, 4),
                    "area": round(det.area, 2),
                }
                for det in detections
            ]

        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise


# ==================== FastAPI 服务封装 ====================

app = FastAPI(
    title="DocLayout Service",
    description="文档布局分析服务，支持多种模型后端",
    version="1.0.0",
)

# 全局模型实例和配置
model_service: Optional[LayoutDetectionService] = None


class PredictRequest(BaseModel):
    image_base64: str
    return_format: str = "detailed"  # detailed 或 simple


class PredictResponse(BaseModel):
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    count: int = 0
    message: str = ""


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global model_service
    try:
        SERVICE_TYPE = os.environ.get("SERVICE_TYPE", "doclayout")

        if SERVICE_TYPE == "mineru25":
            server_url = os.environ.get("MINERU_SERVER_URL", "http://127.0.0.1:30000")
            model_service = MinerU25Service(server_url)

        elif SERVICE_TYPE == "doclayout":
            import torch

            model_path = os.environ.get("DOCMODEL_PATH")
            device = os.environ.get("DOCMODEL_DEVICE", "auto")

            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model_service = DocLayoutYOLOService(
                model_path=model_path,
                device=device,
                conf_threshold=float(os.environ.get("DOCMODEL_CONF", "0.2")),
                imgsz=int(os.environ.get("DOCMODEL_IMGSZ", "1024")),
            )

        logger.info(f"✅ {SERVICE_TYPE}服务启动成功")

    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    文档布局分析接口

    - **image_base64**: base64编码的图片（支持data URI格式）
    - **return_format**: 返回格式，detailed包含所有字段，simple只返回关键信息
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        results = model_service.predict_base64(request.image_base64)

        # 简化格式处理
        if request.return_format == "simple":
            results = [
                {
                    "class": r["category"],
                    "box": [
                        r["bbox"]["x1"],
                        r["bbox"]["y1"],
                        r["bbox"]["x2"],
                        r["bbox"]["y2"],
                    ],
                    "score": r["score"],
                }
                for r in results
            ]

        return PredictResponse(
            success=True,
            data=results,
            count=len(results),
            message=f"检测到 {len(results)} 个布局元素",
        )

    except Exception as e:
        logger.error(f"处理请求失败: {e}")
        return PredictResponse(success=False, data=None, count=0, message=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model_service is not None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service",
        type=str,
        default="doclayout",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务绑定地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8787)),
        help="服务绑定端口（默认: 8787）",
    )
    parser.add_argument(
        "--mineru-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="MinerU服务器地址（仅在--service mineru25时有效，默认: http://127.0.0.1:30000）",
    )
    parser.add_argument(
        "--docmodel-path",
        type=str,
        default=os.environ.get("DOCMODEL_PATH"),
        help="DocLayout模型路径（仅在--service doclayout时有效，默认: 环境变量DOCMODEL_PATH）",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    os.environ["SERVICE_TYPE"] = args.service
    os.environ["MINERU_SERVER_URL"] = args.mineru_url

    logger.info(f"✓ 服务地址: http://{args.host}:{args.port}")

    # 启动服务
    uvicorn.run(
        "doclayout_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,  # 模型较大，建议单进程
    )
