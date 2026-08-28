# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Box:
    """矩形框数据结构"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


class NativeXYCut:
    """
    原生 XY-Cut 算法实现
    基于 Nagy et al. (1992) 的经典递归投影分割算法
    """

    def __init__(
        self,
        resolution: int = 1000,  # 投影分辨率
        min_gap_pixels: int = 5,  # 最小间隙像素
        noise_threshold: float = 0.1,
    ):  # 噪声阈值（最大高度的比例）
        self.resolution = resolution
        self.min_gap_pixels = min_gap_pixels
        self.noise_threshold = noise_threshold

    def _calculate_projection(
        self, elements: List[Dict], axis: str = "x"
    ) -> Tuple[np.ndarray, float, float]:
        """
        计算投影直方图（Projection Profile）

        axis: 'x' - 水平投影（对x轴积分，用于垂直分割/水平切割）
              'y' - 垂直投影（对y轴积分，用于水平分割/垂直切割）

        返回: (profile, min_coord, max_coord)
        """
        if not elements:
            return np.array([]), 0, 0

        # 获取坐标范围
        if axis == "x":
            coords = [
                (e["coordinate"][0], e["coordinate"][2]) for e in elements
            ]  # x1, x2
        else:
            coords = [
                (e["coordinate"][1], e["coordinate"][3]) for e in elements
            ]  # y1, y2

        min_coord = min(c[0] for c in coords)
        max_coord = max(c[1] for c in coords)

        if max_coord <= min_coord:
            return np.array([]), min_coord, max_coord

        # 创建投影数组
        scale = self.resolution / (max_coord - min_coord)
        size = int((max_coord - min_coord) * scale) + 1
        profile = np.zeros(size)

        # 填充投影（元素占据的位置标记为1）
        for start, end in coords:
            start_idx = int((start - min_coord) * scale)
            end_idx = int((end - min_coord) * scale)
            start_idx = max(0, min(start_idx, size - 1))
            end_idx = max(0, min(end_idx, size - 1))
            profile[start_idx : end_idx + 1] = 1

        return profile, min_coord, max_coord

    def _find_valleys(self, profile: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        寻找投影直方图中的 valleys（零值间隙）
        返回: [(start_idx, end_idx, length), ...] 按长度降序排列
        """
        if len(profile) == 0:
            return []

        valleys = []
        in_valley = False
        valley_start = 0

        for i, val in enumerate(profile):
            if val == 0 and not in_valley:
                # 进入 valley
                in_valley = True
                valley_start = i
            elif val > 0 and in_valley:
                # 离开 valley
                in_valley = False
                length = i - valley_start
                if length >= self.min_gap_pixels:
                    valleys.append((valley_start, i, length))

        # 处理结尾
        if in_valley:
            length = len(profile) - valley_start
            if length >= self.min_gap_pixels:
                valleys.append((valley_start, len(profile), length))

        # 按长度降序排列
        valleys.sort(key=lambda x: x[2], reverse=True)
        return valleys

    def _find_cut_position(
        self, profile: np.ndarray, min_coord: float, max_coord: float
    ) -> Optional[float]:
        """
        寻找最佳切割位置（最大 valley 的中点）
        """
        valleys = self._find_valleys(profile)

        if not valleys:
            return None

        # 选择最大的 valley
        best_valley = valleys[0]
        valley_center = (best_valley[0] + best_valley[1]) / 2

        # 转换回原始坐标
        scale = (max_coord - min_coord) / self.resolution
        cut_pos = min_coord + valley_center * scale

        return cut_pos

    def _split_elements(
        self, elements: List[Dict], axis: str, cut_pos: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        根据切割位置分割元素列表
        axis: 'x' - 垂直分割（左右），'y' - 水平分割（上下）
        """
        left_or_top = []
        right_or_bottom = []

        for elem in elements:
            coord = elem["coordinate"]
            if axis == "x":
                # 垂直分割：根据中心点 x 坐标
                center = (coord[0] + coord[2]) / 2
                if center < cut_pos:
                    left_or_top.append(elem)
                else:
                    right_or_bottom.append(elem)
            else:
                # 水平分割：根据中心点 y 坐标
                center = (coord[1] + coord[3]) / 2
                if center < cut_pos:
                    left_or_top.append(elem)
                else:
                    right_or_bottom.append(elem)

        return left_or_top, right_or_bottom

    def _recursive_xycut(self, elements: List[Dict], depth: int = 0) -> List[Dict]:
        """
        递归 XY-Cut 核心算法

        策略：
        - 偶数深度（0, 2, 4...）：尝试水平分割（X-Cut，处理多栏）
        - 奇数深度（1, 3, 5...）：尝试垂直分割（Y-Cut，处理段落）
        - 如果当前方向无法分割，尝试另一方向
        """
        if len(elements) <= 1:
            return elements

        # 决定当前切割方向
        # depth % 2 == 0: 水平切割（垂直分割，处理多栏）
        # depth % 2 == 1: 垂直切割（水平分割，处理段落）
        primary_axis = "x" if depth % 2 == 0 else "y"
        secondary_axis = "y" if depth % 2 == 0 else "x"

        # 尝试主要方向
        profile, min_c, max_c = self._calculate_projection(elements, primary_axis)
        cut_pos = self._find_cut_position(profile, min_c, max_c)

        # 如果主要方向无法分割，尝试次要方向
        if cut_pos is None:
            profile, min_c, max_c = self._calculate_projection(elements, secondary_axis)
            cut_pos = self._find_cut_position(profile, min_c, max_c)
            if cut_pos is None:
                # 两个方向都无法分割，按阅读顺序排序并返回
                return sorted(
                    elements, key=lambda e: (e["coordinate"][1], e["coordinate"][0])
                )
            axis = secondary_axis
        else:
            axis = primary_axis

        # 分割元素
        part1, part2 = self._split_elements(elements, axis, cut_pos)

        # 递归处理两个子区域
        sorted_part1 = self._recursive_xycut(part1, depth + 1)
        sorted_part2 = self._recursive_xycut(part2, depth + 1)

        # 合并结果：根据切割方向决定顺序
        # X-Cut（垂直分割）：左 -> 右
        # Y-Cut（水平分割）：上 -> 下
        return sorted_part1 + sorted_part2

    def sort(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        主排序函数

        输入: [{'cls_id': int, 'label': str, 'score': float, 'coordinate': [x1,y1,x2,y2]}, ...]
        输出: 按阅读顺序排序后的相同格式列表
        """
        if not data or len(data) <= 1:
            return data

        # 深拷贝避免修改原始数据
        elements = [dict(item) for item in data]

        # 执行 XY-Cut
        sorted_elements = self._recursive_xycut(elements)

        return sorted_elements


# 便捷函数
def xycut_sort(
    data: List[Dict], resolution: int = 1000, min_gap_pixels: int = 5
) -> List[Dict]:
    """
    原生 XY-Cut 排序

    参数:
        data: 输入的元素列表
        resolution: 投影计算分辨率（越高越精确，但计算量越大）
        min_gap_pixels: 视为有效间隙的最小像素宽度

    返回:
        按阅读顺序排序后的元素列表
    """
    sorter = NativeXYCut(resolution=resolution, min_gap_pixels=min_gap_pixels)
    return sorter.sort(data)


# ==================== 测试 ====================
if __name__ == "__main__":
    # 测试数据
    test_data = [
        {
            "cls_id": 5,
            "label": "table",
            "score": 0.982200026512146,
            "coordinate": [60.871647, 385.56488, 531.97473, 515.497],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.9610000252723694,
            "coordinate": [81.91811, 574.6846, 516.36523, 682.5999],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.9592000246047974,
            "coordinate": [62.34032, 110.59101, 533.2826, 155.41505],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.9453999996185303,
            "coordinate": [60.51577, 157.69757, 532.9518, 186.58481],
        },
        {
            "cls_id": 0,
            "label": "title",
            "score": 0.9190000295639038,
            "coordinate": [60.07066, 295.92142, 121.60342, 309.42285],
        },
        {
            "cls_id": 0,
            "label": "title",
            "score": 0.9139000177383423,
            "coordinate": [60.495594, 324.91953, 163.4991, 338.76758],
        },
        {
            "cls_id": 0,
            "label": "title",
            "score": 0.9110000133514404,
            "coordinate": [61.339836, 691.94763, 165.33566, 705.4755],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.9053999781608582,
            "coordinate": [63.654873, 349.27548, 346.67755, 362.67935],
        },
        {
            "cls_id": 2,
            "label": "abandon",
            "score": 0.8967999815940857,
            "coordinate": [62.350937, 53.90546, 150.0145, 66.69008],
        },
        {
            "cls_id": 0,
            "label": "title",
            "score": 0.8817999958992004,
            "coordinate": [61.210056, 718.15686, 123.36274, 731.5562],
        },
        {
            "cls_id": 6,
            "label": "table_caption",
            "score": 0.8718000054359436,
            "coordinate": [221.91817, 370.39276, 370.3377, 383.2465],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.864300012588501,
            "coordinate": [82.30455, 559.32513, 259.60107, 571.7553],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.8571000099182129,
            "coordinate": [60.233418, 527.1941, 519.71796, 541.0006],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.853600025177002,
            "coordinate": [81.44936, 234.24342, 407.29248, 278.9216],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.8496999740600586,
            "coordinate": [62.26652, 543.4231, 301.662, 556.05664],
        },
        {
            "cls_id": 0,
            "label": "title",
            "score": 0.8464999794960022,
            "coordinate": [61.476288, 86.34719, 123.50807, 99.87304],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.8454999923706055,
            "coordinate": [81.4735, 741.32245, 366.84705, 755.2365],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.8217999935150146,
            "coordinate": [82.83689, 218.53714, 110.286354, 230.69908],
        },
        {
            "cls_id": 2,
            "label": "abandon",
            "score": 0.8194000124931335,
            "coordinate": [288.83542, 758.4836, 305.39725, 767.77374],
        },
        {
            "cls_id": 1,
            "label": "plain_text",
            "score": 0.8104000091552734,
            "coordinate": [435.48456, 197.53156, 532.0936, 209.88019],
        },
        {
            "cls_id": 8,
            "label": "formula_caption",
            "score": 0.7896999716758728,
            "coordinate": [247.51718, 189.88368, 346.1708, 215.29709],
        },
        {
            "cls_id": 2,
            "label": "abandon",
            "score": 0.36980000138282776,
            "coordinate": [68.418045, 759.9731, 76.1235, 768.9891],
        },
    ]

    print("=" * 70)
    print("原生 XY-Cut 算法 - 阅读顺序排序")
    print("=" * 70)

    # 执行排序
    sorted_data = xycut_sort(test_data, resolution=1000, min_gap_pixels=5)

    # 打印结果
    print(
        f"\n{'序号':<5} {'标签':<15} {'类别ID':<8} {'Y坐标':<10} {'X坐标':<10} {'内容预览'}"
    )
    print("-" * 70)

    for i, item in enumerate(sorted_data):
        y = item["coordinate"][1]
        x = item["coordinate"][0]
        label = item["label"]
        cls_id = item["cls_id"]
        score = item["score"]

        # 生成简单的内容预览
        preview = f"score:{score:.2f}"

        print(f"{i+1:<5} {label:<15} {cls_id:<8} {y:<10.1f} {x:<10.1f} {preview}")

    print("\n" + "=" * 70)
    print("排序逻辑说明:")
    print("-" * 70)
    print("1. XY-Cut 交替进行水平和垂直分割")
    print("2. 水平分割(X-Cut): 处理多栏布局，从左到右")
    print("3. 垂直分割(Y-Cut): 处理段落分割，从上到下")
    print("4. 通过投影直方图寻找元素间的空白间隙(valley)")
    print("5. 沿最大间隙递归分割，直到无法分割为止")
    print("=" * 70)
