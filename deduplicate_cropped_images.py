#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于模板匹配的图片去重脚本
针对pos_color_cropped目录中的小图片进行去重
使用OpenCV模板匹配算法识别相似图片
"""

import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import re

def load_images_from_directory(directory):
    """
    从目录加载所有图片
    :param directory: 图片目录
    :return: 图片信息列表 [{filename, filepath, image, timestamp}, ...]
    """
    images = []
    
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return images
    
    png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    
    for filename in png_files:
        filepath = os.path.join(directory, filename)
        
        # 读取图片
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 提取时间戳
            timestamp = extract_timestamp_from_filename(filename)
            
            images.append({
                'filename': filename,
                'filepath': filepath,
                'image': img,
                'timestamp': timestamp
            })
        else:
            print(f"警告: 无法读取图片 {filename}")
    
    return images

def extract_timestamp_from_filename(filename):
    """
    从文件名中提取时间戳
    :param filename: 文件名
    :return: datetime对象，如果解析失败则返回None
    """
    pattern = r'before_r_key_(\d{8})_(\d{6})_(\d{3})'
    match = re.search(pattern, filename)
    if match:
        date_str, time_str, ms_str = match.groups()
        try:
            dt_str = f"{date_str}_{time_str}_{ms_str}"
            dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S_%f")
            return dt
        except ValueError:
            pass
    return None

def calculate_template_match_score(img1, img2, method=cv2.TM_CCOEFF_NORMED):
    """
    计算两个图片的模板匹配得分
    :param img1: 图片1
    :param img2: 图片2
    :param method: 匹配方法
    :return: 匹配得分 (0-1之间，1表示完全匹配)
    """
    try:
        # 确保两个图片尺寸相同
        if img1.shape != img2.shape:
            return 0.0
        
        # 使用模板匹配
        result = cv2.matchTemplate(img1, img2, method)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max_val
    except Exception as e:
        print(f"模板匹配计算失败: {e}")
        return 0.0

def calculate_histogram_similarity(img1, img2):
    """
    计算两个图片的直方图相似度
    :param img1: 图片1
    :param img2: 图片2
    :return: 相似度得分 (0-1之间，1表示完全相似)
    """
    try:
        # 计算直方图
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        # 计算相关性
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return correlation
    except Exception as e:
        print(f"直方图相似度计算失败: {e}")
        return 0.0

def calculate_ssim_similarity(img1, img2):
    """
    计算结构相似性指数 (SSIM)
    :param img1: 图片1
    :param img2: 图片2
    :return: SSIM得分 (0-1之间，1表示完全相似)
    """
    try:
        # 转换为float32
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # 计算均值
        mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = cv2.GaussianBlur(img1_f * img1_f, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2_f * img2_f, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2
        
        # SSIM常数
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    except Exception as e:
        print(f"SSIM计算失败: {e}")
        return 0.0

def find_similar_images(images, similarity_threshold=0.95, use_multiple_methods=True):
    """
    找到相似的图片组
    :param images: 图片信息列表
    :param similarity_threshold: 相似度阈值
    :param use_multiple_methods: 是否使用多种方法综合判断
    :return: 相似图片组列表
    """
    similar_groups = []
    processed = set()
    
    print(f"开始比较 {len(images)} 个图片...")
    
    for i, img1_info in enumerate(images):
        if i in processed:
            continue
        
        current_group = [img1_info]
        processed.add(i)
        
        for j, img2_info in enumerate(images[i+1:], i+1):
            if j in processed:
                continue
            
            # 计算相似度
            if use_multiple_methods:
                # 使用多种方法综合判断
                template_score = calculate_template_match_score(img1_info['image'], img2_info['image'])
                hist_score = calculate_histogram_similarity(img1_info['image'], img2_info['image'])
                ssim_score = calculate_ssim_similarity(img1_info['image'], img2_info['image'])
                
                # 综合得分 (可以调整权重)
                combined_score = (template_score * 0.4 + hist_score * 0.3 + ssim_score * 0.3)
                
                print(f"比较 {img1_info['filename']} vs {img2_info['filename']}: "
                      f"模板={template_score:.3f}, 直方图={hist_score:.3f}, SSIM={ssim_score:.3f}, 综合={combined_score:.3f}")
                
                is_similar = combined_score >= similarity_threshold
            else:
                # 仅使用模板匹配
                template_score = calculate_template_match_score(img1_info['image'], img2_info['image'])
                print(f"比较 {img1_info['filename']} vs {img2_info['filename']}: 模板匹配={template_score:.3f}")
                is_similar = template_score >= similarity_threshold
            
            if is_similar:
                current_group.append(img2_info)
                processed.add(j)
        
        if len(current_group) > 1:
            similar_groups.append(current_group)
    
    return similar_groups

def deduplicate_by_template_matching(directory="pos_color_cropped", 
                                   similarity_threshold=0.95, 
                                   dry_run=True,
                                   use_multiple_methods=True):
    """
    基于模板匹配进行图片去重
    :param directory: 图片目录
    :param similarity_threshold: 相似度阈值
    :param dry_run: 是否为试运行模式
    :param use_multiple_methods: 是否使用多种相似度计算方法
    """
    print(f"=== 基于模板匹配的图片去重 ===")
    print(f"目录: {directory}")
    print(f"相似度阈值: {similarity_threshold}")
    print(f"使用多种方法: {use_multiple_methods}")
    print()
    
    # 加载所有图片
    images = load_images_from_directory(directory)
    
    if len(images) < 2:
        print("图片数量不足，无需去重")
        return
    
    print(f"加载了 {len(images)} 个图片")
    
    # 找到相似图片组
    similar_groups = find_similar_images(images, similarity_threshold, use_multiple_methods)
    
    if not similar_groups:
        print("没有找到相似的图片")
        return
    
    print(f"\n找到 {len(similar_groups)} 个相似图片组:")
    
    total_files = len(images)
    files_to_keep = 0
    files_to_delete = 0
    
    for group_idx, group in enumerate(similar_groups, 1):
        print(f"\n=== 相似组 {group_idx} ({len(group)} 个图片) ===")
        
        # 按时间戳排序，保留最早的
        group_with_timestamp = [img for img in group if img['timestamp'] is not None]
        group_without_timestamp = [img for img in group if img['timestamp'] is None]
        
        if group_with_timestamp:
            group_with_timestamp.sort(key=lambda x: x['timestamp'])
            sorted_group = group_with_timestamp + group_without_timestamp
        else:
            sorted_group = group
        
        # 保留第一个，删除其余
        keep_img = sorted_group[0]
        delete_imgs = sorted_group[1:]
        
        print(f"保留: {keep_img['filename']} (时间戳: {keep_img['timestamp']})")
        files_to_keep += 1
        
        for img_info in delete_imgs:
            print(f"删除: {img_info['filename']} (时间戳: {img_info['timestamp']})")
            files_to_delete += 1
            
            if not dry_run:
                try:
                    os.remove(img_info['filepath'])
                    print(f"  已删除: {img_info['filename']}")
                except Exception as e:
                    print(f"  删除失败: {img_info['filename']}, 错误: {e}")
    
    # 统计未分组的图片（唯一图片）
    unique_files = total_files - sum(len(group) for group in similar_groups)
    files_to_keep += unique_files
    
    print(f"\n=== 去重统计 ===")
    print(f"总文件数: {total_files}")
    print(f"相似组数: {len(similar_groups)}")
    print(f"唯一文件数: {unique_files}")
    print(f"保留文件数: {files_to_keep}")
    print(f"删除文件数: {files_to_delete}")
    print(f"节省空间: {files_to_delete} 个文件")
    
    if dry_run:
        print("\n注意: 这是试运行模式，没有实际删除文件")
        print("要实际执行删除，请设置 dry_run=False")

def main():
    """
    主函数
    """
    print("=== 基于模板匹配的图片去重工具 ===")
    print("专门用于处理pos_color_cropped目录中的小图片")
    print()
    
    while True:
        print("请选择操作:")
        print("1. 高精度去重 (相似度阈值: 0.98, 使用多种方法)")
        print("2. 中等精度去重 (相似度阈值: 0.95, 使用多种方法)")
        print("3. 低精度去重 (相似度阈值: 0.90, 仅模板匹配)")
        print("4. 自定义参数去重")
        print("5. 退出")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == "1":
            print("\n开始高精度去重 (试运行)...")
            deduplicate_by_template_matching(similarity_threshold=0.98, dry_run=True, use_multiple_methods=True)
            
        elif choice == "2":
            print("\n开始中等精度去重 (试运行)...")
            deduplicate_by_template_matching(similarity_threshold=0.95, dry_run=True, use_multiple_methods=True)
            
        elif choice == "3":
            print("\n开始低精度去重 (试运行)...")
            deduplicate_by_template_matching(similarity_threshold=0.90, dry_run=True, use_multiple_methods=False)
            
        elif choice == "4":
            try:
                threshold = float(input("请输入相似度阈值 (0.0-1.0): "))
                use_multiple = input("是否使用多种方法 (y/n): ").lower().startswith('y')
                
                print(f"\n开始自定义去重 (阈值={threshold}, 多方法={use_multiple}) (试运行)...")
                deduplicate_by_template_matching(similarity_threshold=threshold, dry_run=True, use_multiple_methods=use_multiple)
                
            except ValueError:
                print("输入无效，请输入有效的数字")
                continue
        
        elif choice == "5":
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")
            continue
        
        # 询问是否执行实际删除
        if choice in ["1", "2", "3", "4"]:
            print("\n是否要执行实际删除操作？")
            confirm = input("输入 'yes' 确认删除，其他任意键取消: ").strip().lower()
            
            if confirm == 'yes':
                print("\n开始实际删除重复文件...")
                if choice == "1":
                    deduplicate_by_template_matching(similarity_threshold=0.98, dry_run=False, use_multiple_methods=True)
                elif choice == "2":
                    deduplicate_by_template_matching(similarity_threshold=0.95, dry_run=False, use_multiple_methods=True)
                elif choice == "3":
                    deduplicate_by_template_matching(similarity_threshold=0.90, dry_run=False, use_multiple_methods=False)
                elif choice == "4":
                    deduplicate_by_template_matching(similarity_threshold=threshold, dry_run=False, use_multiple_methods=use_multiple)
                print("去重完成！")
            else:
                print("操作已取消")
        
        print()

if __name__ == "__main__":
    main()