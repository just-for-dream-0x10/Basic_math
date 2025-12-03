#!/usr/bin/env python3
"""
视频整理脚本
将分散的视频文件统一移动到一个文件夹中
"""

import os
import shutil
from pathlib import Path

def organize_videos():
    """整理视频文件到统一目录"""
    
    media_dir = Path(__file__).parent / "media"
    unified_videos_dir = media_dir / "unified_videos"
    
    # 创建统一的视频目录
    unified_videos_dir.mkdir(exist_ok=True)
    
    # 创建质量子目录
    for quality in ["480p15", "720p30", "1080p60"]:
        (unified_videos_dir / quality).mkdir(exist_ok=True)
    
    # 查找所有视频文件
    video_files = list(media_dir.glob("videos/**/*.mp4"))
    
    moved_count = 0
    
    for video_file in video_files:
        if "partial_movie_files" in str(video_file):
            continue
            
        # 提取质量信息
        if "480p15" in str(video_file):
            quality = "480p15"
        elif "720p30" in str(video_file):
            quality = "720p30"
        elif "1080p60" in str(video_file):
            quality = "1080p60"
        else:
            quality = "720p30"  # 默认
        
        # 确定模块名
        path_parts = video_file.parts
        module_name = "unknown"
        
        # 从路径中提取模块名
        for part in path_parts:
            if part in ["matrix_transform", "convolution_operation", "loss_function", "optimizer", "svm"]:
                module_name = part
                break
        
        # 提取场景名
        scene_name = video_file.stem
        
        # 创建新的文件名：模块_场景.mp4
        new_filename = f"{module_name}_{scene_name}.mp4"
        new_path = unified_videos_dir / quality / new_filename
        
        # 移动文件
        try:
            shutil.move(str(video_file), str(new_path))
            print(f"✅ 移动: {video_file.name} -> {new_filename}")
            moved_count += 1
        except Exception as e:
            print(f"❌ 移动失败: {video_file.name} - {e}")
    
    print(f"\n🎉 视频整理完成！共移动了 {moved_count} 个文件")
    print(f"📁 统一视频目录: {unified_videos_dir}")
    
    # 显示整理后的文件列表
    print("\n📋 整理后的视频文件:")
    for quality_dir in sorted(unified_videos_dir.iterdir()):
        if quality_dir.is_dir():
            print(f"\n🎬 {quality_dir.name}/")
            for video_file in sorted(quality_dir.glob("*.mp4")):
                print(f"  - {video_file.name}")

if __name__ == "__main__":
    organize_videos()