#!/usr/bin/env python3
"""
视频后处理脚本
在Manim生成视频后，将其移动到统一目录并重命名
"""

import os
import shutil
from pathlib import Path

def move_generated_video(module_name, scene_name, scene_class):
    """移动生成的视频到统一目录"""
    
    media_dir = Path(__file__).parent / "media"
    unified_videos_dir = media_dir / "unified_videos"
    quality_dir = "720p30"  # 默认质量
    
    # 查找刚生成的视频文件
    possible_dirs = [
        media_dir / "videos" / f"{module_name}_{scene_name}" / quality_dir,
        media_dir / "videos" / module_name / quality_dir,
        media_dir / "videos" / f"{module_name}_{scene_name}" / "720p30",
        media_dir / "videos" / module_name / "720p30"
    ]
    
    source_file = None
    for search_dir in possible_dirs:
        potential_file = search_dir / f"{scene_class}.mp4"
        if potential_file.exists():
            source_file = potential_file
            break
    
    if not source_file:
        print(f"❌ 找不到生成的视频文件: {scene_class}.mp4")
        return False
    
    # 目标路径
    target_file = unified_videos_dir / quality_dir / f"{module_name}_{scene_name}.mp4"
    
    # 确保目标目录存在
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 移动文件
        shutil.move(str(source_file), str(target_file))
        print(f"✅ 视频已移动到: {target_file.name}")
        return True
    except Exception as e:
        print(f"❌ 移动视频失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("用法: python post_process_video.py <module_name> <scene_name> <scene_class>")
        sys.exit(1)
    
    module_name = sys.argv[1]
    scene_name = sys.argv[2]
    scene_class = sys.argv[3]
    
    success = move_generated_video(module_name, scene_name, scene_class)
    sys.exit(0 if success else 1)