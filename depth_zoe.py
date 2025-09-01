import torch
import cv2
import numpy as np
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
# 加载 ZoeDepth 模型（基础版）
model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")

def generate_depth_map(image_path, output_path="depth.png"):
    # 读取图像（BGR -> RGB）
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 预处理
    inputs = processor(images=image, return_tensors="pt")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 调整到原图大小
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # 归一化到 0-255
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype("uint8")

    # 保存深度图
    cv2.imwrite(output_path, depth_map)
    print(f"✅ 深度图已保存: {output_path}")


if __name__ == "__main__":
    # 示例：对一张 test.jpg 生成深度图
    generate_depth_map("test.jpg", "depth.png")
