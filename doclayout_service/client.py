# client_example.py
import base64
import requests
import cv2
import numpy as np
from PIL import Image
import colorsys


# 定义类别颜色映射
CATEGORY_COLORS = {
    "title": (255, 0, 0),  # 红色
    "text": (0, 255, 0),  # 绿色
    "figure": (0, 0, 255),  # 蓝色
    "table": (255, 255, 0),  # 青色
    "header": (255, 0, 255),  # 品红
    "footer": (0, 255, 255),  # 黄色
    "caption": (128, 0, 255),  # 橙色
    "equation": (255, 128, 0),  # 天蓝
    "list": (128, 255, 0),  # 春绿
    "code": (0, 128, 255),  # 浅蓝
}


def get_color_for_category(category: str) -> tuple:
    """根据类别获取颜色，如果没有预定义则自动生成"""
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    # 自动生成颜色
    hash_val = hash(category) % 256
    hue = hash_val / 256.0
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def visualize_result(
    image_path: str,
    result: dict,
    output_path: str = None,
    show_confidence: bool = True,
    line_thickness: int = 2,
):
    """
    可视化文档布局检测结果

    Args:
        image_path: 原始图片路径
        result: predict_image返回的结果
        output_path: 输出图片路径，如果为None则保存为原文件名_visualized.jpg
        show_confidence: 是否显示置信度
        line_thickness: 边框线宽
    """
    if not result["success"]:
        print(f"无法可视化：{result['message']}")
        return

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 遍历检测结果并绘制
    for item in result["data"]:
        bbox = item["bbox"]
        category = item["category"]
        score = item["score"]

        # 获取边界框坐标
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        # 获取颜色
        color = get_color_for_category(category)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # 准备标签文本
        if show_confidence:
            label = f"{category}: {score:.2f}"
        else:
            label = category

        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # 绘制标签背景
        label_y = max(y1 - 5, text_height + 5)
        cv2.rectangle(
            image,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 5, label_y),
            color,
            -1,
        )

        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1 + 2, label_y - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    # 生成输出路径
    if output_path is None:
        base_name = image_path.rsplit(".", 1)[0]
        output_path = f"{base_name}_visualized.jpg"

    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"✅ 可视化结果已保存到: {output_path}")

    return image


def predict_image(image_path: str, service_url: str = "http://10.21.226.178:8787"):
    """
    调用DocLayout-YOLO服务分析文档布局

    Args:
        image_path: 本地图片路径
        service_url: 服务地址
    """
    # 读取并编码图片
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # 调用服务
    response = requests.post(
        f"{service_url}/predict",
        json={"image_base64": image_base64, "return_format": "detailed"},  # 或 "simple"
    )

    result = response.json()

    if result["success"]:
        print(f"✅ 检测到 {result['count']} 个元素:")
        for item in result["data"]:
            print(
                f"  - {item['category']}: "
                f"box=[{item['bbox']['x1']:.1f}, {item['bbox']['y1']:.1f}, "
                f"{item['bbox']['x2']:.1f}, {item['bbox']['y2']:.1f}], "
                f"score={item['score']:.3f}"
            )
    else:
        print(f"❌ 错误: {result['message']}")

    return result


if __name__ == "__main__":
    # 示例调用
    image_path = (
        "/paddle/project/PaddleX/temp_images/book_zh_HGT51022016_extracted_page_12.png"
    )
    result = predict_image(image_path)

    # 可视化结果
    if result["success"]:
        visualize_result(image_path, result, show_confidence=True)
