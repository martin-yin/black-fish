#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从pos_color目录的图片中截取指定区域
坐标: x=2566, y=1108, width=52, height=52
"""

import os
import cv2
import numpy as np
from PIL import Image

def crop_images_from_pos_color(x=2566, y=1108, width=52, height=52, 
                               input_dir="pos_color", output_dir="pos_color_cropped"):
    """
    从pos_color目录中的图片截取指定区域
    :param x: 截取区域的x坐标
    :param y: 截取区域的y坐标
    :param width: 截取区域的宽度
    :param height: 截取区域的高度
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"目录 {input_dir} 中没有找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    print(f"截取区域: x={x}, y={y}, width={width}, height={height}")
    
    success_count = 0
    error_count = 0
    
    for filename in png_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = f"cropped_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # 使用OpenCV读取图片
            img = cv2.imread(input_path)
            
            if img is None:
                print(f"警告: 无法读取图片 {filename}")
                error_count += 1
                continue
            
            # 获取图片尺寸
            img_height, img_width = img.shape[:2]
            
            # 检查截取区域是否超出图片边界
            if x + width > img_width or y + height > img_height:
                print(f"警告: 截取区域超出图片边界 {filename} (图片尺寸: {img_width}x{img_height})")
                error_count += 1
                continue
            
            # 截取指定区域
            cropped_img = img[y:y+height, x:x+width]
            
            # 保存截取的图片
            cv2.imwrite(output_path, cropped_img)
            
            print(f"成功截取: {filename} -> {output_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            error_count += 1
    
    print(f"\n=== 截取完成 ===")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")

def crop_single_image(image_path, x=2566, y=1108, width=52, height=52, output_path=None):
    """
    截取单个图片的指定区域
    :param image_path: 输入图片路径
    :param x: 截取区域的x坐标
    :param y: 截取区域的y坐标
    :param width: 截取区域的宽度
    :param height: 截取区域的高度
    :param output_path: 输出路径，如果为None则自动生成
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"错误: 无法读取图片 {image_path}")
            return False
        
        # 获取图片尺寸
        img_height, img_width = img.shape[:2]
        print(f"图片尺寸: {img_width}x{img_height}")
        
        # 检查截取区域是否超出图片边界
        if x + width > img_width or y + height > img_height:
            print(f"错误: 截取区域超出图片边界")
            print(f"截取区域: x={x}, y={y}, width={width}, height={height}")
            print(f"图片尺寸: {img_width}x{img_height}")
            return False
        
        # 截取指定区域
        cropped_img = img[y:y+height, x:x+width]
        
        # 生成输出路径
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"cropped_{base_name}.png"
        
        # 保存截取的图片
        cv2.imwrite(output_path, cropped_img)
        
        print(f"成功截取并保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"截取图片时出错: {e}")
        return False

def preview_crop_area(image_path, x=2566, y=1108, width=52, height=52):
    """
    预览截取区域（在原图上标记截取区域）
    :param image_path: 图片路径
    :param x: 截取区域的x坐标
    :param y: 截取区域的y坐标
    :param width: 截取区域的宽度
    :param height: 截取区域的高度
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"错误: 无法读取图片 {image_path}")
            return
        
        # 在图片上绘制截取区域的矩形框
        img_with_rect = img.copy()
        cv2.rectangle(img_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # 保存预览图片
        preview_path = "crop_preview.png"
        cv2.imwrite(preview_path, img_with_rect)
        
        print(f"预览图片已保存: {preview_path}")
        print(f"绿色矩形框标记了截取区域: x={x}, y={y}, width={width}, height={height}")
        
    except Exception as e:
        print(f"生成预览时出错: {e}")

def main():
    """
    主函数
    """
    print("=== pos_color图片截取工具 ===")
    print("截取坐标: x=2566, y=1108, width=52, height=52")
    print()
    
    while True:
        print("请选择操作:")
        print("1. 批量截取pos_color目录中的所有图片")
        print("2. 截取单个图片")
        print("3. 预览截取区域（在图片上标记）")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n开始批量截取...")
            crop_images_from_pos_color()
            
        elif choice == "2":
            image_path = input("请输入图片路径: ").strip()
            if os.path.exists(image_path):
                crop_single_image(image_path)
            else:
                print(f"错误: 文件不存在 {image_path}")
                
        elif choice == "3":
            image_path = input("请输入图片路径: ").strip()
            if os.path.exists(image_path):
                preview_crop_area(image_path)
            else:
                print(f"错误: 文件不存在 {image_path}")
                
        elif choice == "4":
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")
        
        print()

if __name__ == "__main__":
    main()