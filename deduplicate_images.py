#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片去重脚本
根据文件名中的RGB颜色值对pos_color目录中的图片进行去重
保留每个颜色值的第一个文件（按时间戳排序）
"""

import os
import re
from collections import defaultdict
from datetime import datetime

def extract_rgb_from_filename(filename):
    """
    从文件名中提取RGB颜色值
    :param filename: 文件名
    :return: RGB颜色值字符串，如 "RGB_57_56_56"，如果未找到则返回None
    """
    pattern = r'_RGB_(\d+)_(\d+)_(\d+)\.png$'
    match = re.search(pattern, filename)
    if match:
        r, g, b = match.groups()
        return f"RGB_{r}_{g}_{b}"
    return None

def extract_timestamp_from_filename(filename):
    """
    从文件名中提取时间戳
    :param filename: 文件名
    :return: datetime对象，如果解析失败则返回None
    """
    pattern = r'before_r_key_(\d{8})_(\d{6})_(\d{3})_RGB'
    match = re.search(pattern, filename)
    if match:
        date_str, time_str, ms_str = match.groups()
        try:
            # 解析日期时间
            dt_str = f"{date_str}_{time_str}_{ms_str}"
            dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S_%f")
            return dt
        except ValueError:
            pass
    return None

def deduplicate_images(directory="pos_color", dry_run=True):
    """
    对图片进行去重
    :param directory: 图片目录
    :param dry_run: 是否为试运行模式（不实际删除文件）
    """
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return
    
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"目录 {directory} 中没有找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 按RGB颜色值分组
    color_groups = defaultdict(list)
    
    for filename in png_files:
        rgb_color = extract_rgb_from_filename(filename)
        if rgb_color:
            timestamp = extract_timestamp_from_filename(filename)
            color_groups[rgb_color].append({
                'filename': filename,
                'timestamp': timestamp,
                'filepath': os.path.join(directory, filename)
            })
        else:
            print(f"警告: 无法从文件名 {filename} 中提取RGB颜色值")
    
    print(f"\n找到 {len(color_groups)} 个不同的颜色组:")
    
    total_files = 0
    files_to_keep = 0
    files_to_delete = 0
    
    for color, files in color_groups.items():
        total_files += len(files)
        print(f"\n颜色 {color}: {len(files)} 个文件")
        
        if len(files) > 1:
            # 按时间戳排序，保留最早的文件
            files_with_timestamp = [f for f in files if f['timestamp'] is not None]
            files_without_timestamp = [f for f in files if f['timestamp'] is None]
            
            # 对有时间戳的文件按时间排序
            files_with_timestamp.sort(key=lambda x: x['timestamp'])
            
            # 合并列表，有时间戳的在前
            sorted_files = files_with_timestamp + files_without_timestamp
            
            # 第一个文件保留，其余删除
            keep_file = sorted_files[0]
            delete_files = sorted_files[1:]
            
            print(f"  保留: {keep_file['filename']} (时间戳: {keep_file['timestamp']})")
            files_to_keep += 1
            
            for file_info in delete_files:
                print(f"  删除: {file_info['filename']} (时间戳: {file_info['timestamp']})")
                files_to_delete += 1
                
                if not dry_run:
                    try:
                        os.remove(file_info['filepath'])
                        print(f"    已删除: {file_info['filename']}")
                    except Exception as e:
                        print(f"    删除失败: {file_info['filename']}, 错误: {e}")
        else:
            print(f"  保留: {files[0]['filename']} (唯一文件)")
            files_to_keep += 1
    
    print(f"\n=== 去重统计 ===")
    print(f"总文件数: {total_files}")
    print(f"保留文件数: {files_to_keep}")
    print(f"删除文件数: {files_to_delete}")
    print(f"节省空间: {files_to_delete} 个文件")
    
    if dry_run:
        print("\n注意: 这是试运行模式，没有实际删除文件")
        print("要实际执行删除，请运行: deduplicate_images(dry_run=False)")

def main():
    """
    主函数
    """
    print("=== 图片去重工具 ===")
    print("根据文件名中的RGB颜色值进行去重")
    print("保留每个颜色的最早时间戳文件\n")
    
    # 先进行试运行
    print("开始试运行（不会实际删除文件）...")
    deduplicate_images(dry_run=True)
    
    # 询问是否执行实际删除
    print("\n是否要执行实际删除操作？")
    choice = input("输入 'yes' 确认删除，其他任意键取消: ").strip().lower()
    
    if choice == 'yes':
        print("\n开始实际删除重复文件...")
        deduplicate_images(dry_run=False)
        print("去重完成！")
    else:
        print("操作已取消")

if __name__ == "__main__":
    main()