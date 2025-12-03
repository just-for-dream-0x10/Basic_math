#!/usr/bin/env python3
"""
Manim场景运行脚本
用于生成各个模块的动画视频
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import MODULES, ASSETS_DIR

def run_manim_scene(module_name, scene_name, quality="medium"):
    """运行指定的Manim场景"""
    
    # 场景文件名和类名映射
    scene_mappings = {
        "matrix": {
            "matrix_transform": ("MatrixTransformScene", "matrix_transform"),
            "svd_decomposition": ("SVDScene", "matrix_transform"), 
            "eigenvalues": ("EigenvalueScene", "matrix_transform")
        },
        "convolution": {
            "convolution_operation": ("ConvolutionOperationScene", "convolution_operation"),
            "kernel_types": ("KernelTypesScene", "convolution_operation"),
            "feature_extraction": ("FeatureExtractionScene", "convolution_operation")
        },
        "loss": {
            "least_squares": ("LeastSquaresScene", "loss/least_squares"),
            "cross_entropy": ("CrossEntropyScene", "loss/cross_entropy"),
        },
        "optimizer": {
            "sgd": ("SGDScene", "optimizer"),
            "momentum": ("MomentumScene", "optimizer"),
            "adam": ("AdamScene", "optimizer")
        },
        "svm": {
            "margin": ("MarginScene", "svm"),
            "kernel_trick": ("KernelTrickScene", "svm"),
            "dual_problem": ("DualProblemScene", "svm")
        }
    }
    
    if module_name not in scene_mappings:
        print(f"❌ 模块 '{module_name}' 不存在")
        return False
    
    if scene_name not in scene_mappings[module_name]:
        print(f"❌ 场景 '{scene_name}' 在模块 '{module_name}' 中不存在")
        return False
    
    # 获取场景类名和文件名
    scene_class, file_name = scene_mappings[module_name][scene_name]
    
    # 质量设置
    quality_flags = {
        "low": "-ql",
        "medium": "-qm", 
        "high": "-qh",
        "production": "-qp"
    }
    
    quality_flag = quality_flags.get(quality, "-qm")
    
    # 构建Manim命令
    # 如果file_name已经包含路径，直接使用；否则添加module_name
    if "/" in file_name:
        file_path = f"scenes/{file_name}.py"
    else:
        file_path = f"scenes/{module_name}/{file_name}.py"
    
    command = [
        "manim", "render",
        file_path,
        scene_class,
        quality_flag
    ]
    
    print(f"🎬 正在生成视频：{module_name}_{scene_name}")
    print(f"🔧 执行命令：{' '.join(command)}")
    
    try:
        import subprocess
        result = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 视频生成成功")
            
            # 移动视频到assets目录
            success = move_video_to_assets(module_name, scene_class)
            return success
        else:
            print(f"❌ 视频生成失败")
            print(f"错误信息：{result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Manim执行错误：{e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误：{e}")
        return False

def move_video_to_assets(module_name, scene_class):
    """移动视频到assets目录"""
    
    # 查找生成的视频文件
    media_dir = Path(__file__).parent / "media"
    video_file = None
    
    # 在media/videos下查找
    for search_dir in media_dir.glob("videos/**/"):
        potential_file = search_dir / f"{scene_class}.mp4"
        if potential_file.exists():
            video_file = potential_file
            break
    
    if not video_file:
        print(f"❌ 找不到生成的视频文件: {scene_class}.mp4")
        return False
    
    # 目标路径
    target_dir = ASSETS_DIR / module_name
    target_file = target_dir / f"{scene_class}.mp4"
    
    try:
        # 确保目标目录存在
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动文件
        shutil.move(str(video_file), str(target_file))
        print(f"✅ 视频已保存到: assets/{module_name}/{scene_class}.mp4")
        return True
    except Exception as e:
        print(f"❌ 移动视频失败: {e}")
        return False

def list_available_scenes():
    """列出所有可用的场景"""
    print("📚 可用的模块和场景：")
    print("=" * 50)
    
    for module_name, module_info in MODULES.items():
        print(f"\n📖 {module_info['name']} ({module_name})")
        print(f"   描述：{module_info['description']}")
        print(f"   场景：")
        
        for scene in module_info["scenes"]:
            print(f"     - {scene}")

def generate_all_videos(quality="medium"):
    """生成所有视频"""
    print("🎬 开始生成所有视频...")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    for module_name, module_info in MODULES.items():
        for scene_name in module_info["scenes"]:
            total_count += 1
            if run_manim_scene(module_name, scene_name, quality):
                success_count += 1
            print("-" * 30)
    
    print(f"\n🎉 视频生成完成！")
    print(f"✅ 成功：{success_count}/{total_count}")
    print(f"❌ 失败：{total_count - success_count}/{total_count}")
    
    # 显示assets目录结构
    print(f"\n📁 Assets目录结构：")
    for module_dir in sorted(ASSETS_DIR.iterdir()):
        if module_dir.is_dir():
            print(f"📂 {module_dir.name}/")
            for video_file in sorted(module_dir.glob("*.mp4")):
                print(f"  🎬 {video_file.name}")

def main():
    parser = argparse.ArgumentParser(description="数学可视化Manim场景生成器")
    parser.add_argument("--list", action="store_true", help="列出所有可用场景")
    parser.add_argument("--module", type=str, help="指定模块名称")
    parser.add_argument("--scene", type=str, help="指定场景名称")
    parser.add_argument("--quality", type=str, default="medium", 
                       choices=["low", "medium", "high", "production"],
                       help="视频质量")
    parser.add_argument("--all", action="store_true", help="生成所有视频")
    
    args = parser.parse_args()
    
    # 确保assets目录存在
    ASSETS_DIR.mkdir(exist_ok=True)
    for module in MODULES.keys():
        (ASSETS_DIR / module).mkdir(exist_ok=True)
    
    if args.list:
        list_available_scenes()
    elif args.all:
        generate_all_videos(args.quality)
    elif args.module and args.scene:
        run_manim_scene(args.module, args.scene, args.quality)
    else:
        print("使用 --help 查看使用说明")
        print("常用命令：")
        print("  python run_manim.py --list                    # 列出所有场景")
        print("  python run_manim.py --module matrix --scene matrix_transform  # 生成单个场景")
        print("  python run_manim.py --all --quality high     # 生成所有高质量视频")

if __name__ == "__main__":
    main()