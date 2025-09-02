import cv2
import numpy as np
import argparse
import os
import sys


def create_blur_mask(depth_map, focus_distance=0.5, focus_range=0.2):
    """根据深度图创建模糊掩码"""
    # 归一化深度图到 0-1
    normalized_depth = depth_map.astype(np.float32) / 255.0

    # 计算与焦点距离的差异
    focus_diff = np.abs(normalized_depth - focus_distance)

    # 创建模糊强度掩码
    blur_mask = np.zeros_like(focus_diff)

    # 在焦点范围内的区域保持清晰
    in_focus = focus_diff <= focus_range
    blur_mask[in_focus] = 0

    # 焦点范围外的区域根据距离应用不同程度的模糊
    out_of_focus = focus_diff > focus_range
    blur_mask[out_of_focus] = (focus_diff[out_of_focus] - focus_range) / (
        1.0 - focus_range
    )

    # 限制最大模糊强度
    blur_mask = np.clip(blur_mask, 0, 1)

    return blur_mask


def apply_variable_blur(image, blur_mask, max_blur_strength):
    """应用可变模糊效果 - 优化版本"""
    result = image.copy()

    # 根据强度计算核大小范围，使效果更明显
    max_kernel = min(max_blur_strength * 2 + 1, 101)  # 增加最大核大小
    if max_kernel % 2 == 0:
        max_kernel += 1

    # 创建多个模糊级别
    blur_levels = np.linspace(1, max_kernel, 10, dtype=int)
    blur_levels = [k if k % 2 == 1 else k + 1 for k in blur_levels]  # 确保都是奇数

    # 使用向量化操作提高性能
    h, w = blur_mask.shape

    # 对每个模糊级别创建掩码
    for i, kernel_size in enumerate(blur_levels):
        if kernel_size <= 1:
            continue

        # 计算当前级别的掩码
        level_threshold = i / (len(blur_levels) - 1)
        mask = (blur_mask >= level_threshold) & (blur_mask < level_threshold + 0.1)

        if np.any(mask):
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(
                image, (kernel_size, kernel_size), kernel_size / 3
            )

            # 使用掩码混合
            result[mask] = blurred[mask]

    return result


def generate_depth_blur(intensity, input_dir=".", output_path="blurred_result.jpg"):
    """生成深度模糊效果"""

    # 查找原图和深度图
    test_image_path = None
    depth_image_path = None

    # 查找test图片
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        test_path = os.path.join(input_dir, f"test{ext}")
        if os.path.exists(test_path):
            test_image_path = test_path
            break

    # 查找depth图片
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        depth_path = os.path.join(input_dir, f"depth{ext}")
        if os.path.exists(depth_path):
            depth_image_path = depth_path
            break

    if test_image_path is None:
        print("❌ 错误: 未找到test图片文件 (test.jpg, test.jpeg, test.png, test.bmp)")
        return False

    if depth_image_path is None:
        print(
            "❌ 错误: 未找到depth图片文件 (depth.jpg, depth.jpeg, depth.png, depth.bmp)"
        )
        return False

    print(f"📁 找到原图: {test_image_path}")
    print(f"📁 找到深度图: {depth_image_path}")

    try:
        # 读取原图 (BGR -> RGB)
        original_image = cv2.imread(test_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 读取深度图 (灰度)
        depth_map = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

        # 确保深度图尺寸与原图匹配
        if depth_map.shape[:2] != original_image.shape[:2]:
            print("🔄 调整深度图尺寸以匹配原图...")
            depth_map = cv2.resize(
                depth_map, (original_image.shape[1], original_image.shape[0])
            )

        print(f"🖼️  图片尺寸: {original_image.shape[1]}x{original_image.shape[0]}")
        print(f"🎯 模糊强度: {intensity}/100")

        # 将强度从1-100映射到实际的模糊参数
        # 强度越高，模糊效果越强
        max_blur_strength = intensity * 2  # 直接使用强度值
        focus_distance = 1  # 调整焦点距离，聚焦在前景
        focus_range = 0.05  # 减小焦点范围，让模糊效果更明显

        print("🔄 创建模糊掩码...")
        # 创建模糊掩码
        blur_mask = create_blur_mask(depth_map, focus_distance, focus_range)

        print("🔄 应用模糊效果...")
        # 应用可变模糊
        blurred_image = apply_variable_blur(
            original_image, blur_mask, max_blur_strength
        )

        # 转换回BGR格式用于保存
        bgr_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

        # 保存结果
        cv2.imwrite(output_path, bgr_image)
        print(f"✅ 模糊图片已保存: {output_path}")

        return True

    except Exception as e:
        print(f"❌ 处理图片时出错: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="基于深度图的镜头模糊效果生成器")
    parser.add_argument("intensity", type=int, help="模糊强度 (1-100)")
    parser.add_argument(
        "-d", "--dir", default=".", help="输入目录路径 (默认: 当前目录)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="blurred_result.jpg",
        help="输出文件路径 (默认: blurred_result.jpg)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 验证强度参数
    if args.intensity < 1 or args.intensity > 100:
        print("❌ 错误: 强度必须在 1-100 之间")
        sys.exit(1)

    # 验证输入目录
    if not os.path.exists(args.dir):
        print(f"❌ 错误: 目录不存在: {args.dir}")
        sys.exit(1)

    if args.verbose:
        print(f"🚀 开始处理深度模糊效果...")
        print(f"📂 输入目录: {args.dir}")
        print(f"💪 模糊强度: {args.intensity}")
        print(f"💾 输出文件: {args.output}")
        print("-" * 50)

    # 生成深度模糊效果
    success = generate_depth_blur(args.intensity, args.dir, args.output)

    if success:
        print("🎉 处理完成!")
    else:
        print("💥 处理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
