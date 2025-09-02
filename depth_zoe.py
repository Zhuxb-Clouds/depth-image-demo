import torch
import cv2
import numpy as np
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
import os

# 加载 ZoeDepth 模型（基础版）
model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")


def enhance_depth_map(depth_map):
    """
    增强深度图的清晰度和对比度
    """
    # 1. 高斯滤波去噪
    depth_smooth = cv2.GaussianBlur(depth_map, (3, 3), 0.5)

    # 2. 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth_enhanced = clahe.apply((depth_smooth * 255).astype(np.uint8))

    # 3. 边缘保持滤波（双边滤波）
    depth_filtered = cv2.bilateralFilter(depth_enhanced, 9, 75, 75)

    return depth_filtered / 255.0


def apply_gamma_correction(depth_map, gamma=0.7):
    """
    应用伽马校正提升暗部细节
    """
    return np.power(depth_map, gamma)


def normalize_depth_advanced(depth_map, percentile_clip=2):
    """
    高级深度图归一化，使用百分位数裁剪避免极值影响
    """
    # 计算百分位数
    low_percentile = np.percentile(depth_map, percentile_clip)
    high_percentile = np.percentile(depth_map, 100 - percentile_clip)

    # 裁剪极值
    depth_clipped = np.clip(depth_map, low_percentile, high_percentile)

    # 归一化到 0-1
    depth_normalized = (depth_clipped - low_percentile) / (
        high_percentile - low_percentile
    )

    return depth_normalized


def generate_depth_map(
    image_path, output_path="depth.png", enhance_quality=True, save_16bit=False
):
    """
    生成高质量深度图

    Args:
        image_path: 输入图像路径
        output_path: 输出深度图路径
        enhance_quality: 是否启用质量增强
        save_16bit: 是否保存16位深度图（更高精度）
    """
    # 读取图像（BGR -> RGB）
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    original_height, original_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 预处理
    inputs = processor(images=image_rgb, return_tensors="pt")

    # 推理
    print("正在生成深度图...")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 高质量插值到原图大小
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_height, original_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,  # 抗锯齿
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    if enhance_quality:
        print("正在增强深度图质量...")
        # 高级归一化
        depth_map = normalize_depth_advanced(depth_map)

        # 伽马校正
        depth_map = apply_gamma_correction(depth_map)

        # 质量增强
        depth_map = enhance_depth_map(depth_map)
    else:
        # 基础归一化
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # 保存深度图
    if save_16bit:
        # 保存16位深度图以获得更高精度
        depth_16bit = (depth_map * 65535).astype(np.uint16)
        base_name = os.path.splitext(output_path)[0]
        output_16bit = f"{base_name}_16bit.png"
        cv2.imwrite(output_16bit, depth_16bit)
        print(f"✅ 16位深度图已保存: {output_16bit}")

    # 保存8位深度图
    depth_8bit = (depth_map * 255).astype(np.uint8)
    cv2.imwrite(output_path, depth_8bit)
    print(f"✅ 深度图已保存: {output_path}")

    # 生成伪彩色深度图用于可视化
    colormap_path = os.path.splitext(output_path)[0] + "_colormap.png"
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_PLASMA)
    cv2.imwrite(colormap_path, depth_colormap)
    print(f"✅ 彩色深度图已保存: {colormap_path}")

    return depth_map


if __name__ == "__main__":
    # 示例：对一张 test.jpg 生成深度图
    print("=== 深度图生成测试 ===")

    # 生成标准质量深度图
    print("\n1. 生成标准质量深度图...")
    generate_depth_map("test.jpg", "depth_standard.png", enhance_quality=False)

    # 生成高质量深度图
    print("\n2. 生成高质量深度图...")
    generate_depth_map(
        "test.jpg", "depth_enhanced.png", enhance_quality=True, save_16bit=True
    )

    print("\n✅ 所有深度图生成完成！")
    print("📁 输出文件：")
    print("   - depth_standard.png (标准质量)")
    print("   - depth_enhanced.png (增强质量)")
    print("   - depth_enhanced_16bit.png (16位高精度)")
    print("   - depth_enhanced_colormap.png (彩色可视化)")
