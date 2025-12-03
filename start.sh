#!/bin/bash

# 数学笔记可视化平台统一启动脚本

echo "🧮 数学笔记可视化平台"
echo "============================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖包..."
if ! python3 -c "import manim, streamlit" &> /dev/null; then
    echo "🔧 安装依赖包..."
    pip3 install -r requirements.txt
fi

# 创建必要目录
mkdir -p media output

# 自动创建assets目录结构（基于MODULES配置）
echo "📁 自动创建Assets目录结构..."
mkdir -p assets
python3 -c "
from config import MODULES
import os
for module in MODULES.keys():
    os.makedirs(f'assets/{module}', exist_ok=True)
    print(f'  📂 assets/{module}/')
"

# 显示当前状态
echo ""
echo "📊 当前状态："
if [ -d "assets" ]; then
    echo "📁 Assets目录存在，包含以下模块："
    for dir in assets/*/; do
        if [ -d "$dir" ]; then
            module_name=$(basename "$dir")
            video_count=$(find "$dir" -name "*.mp4" | wc -l)
            echo "  📂 $module_name/ ($video_count 个视频)"
        fi
    done
    
    # 显示所有视频文件
    echo ""
    echo "📋 所有视频文件 (./assets/**/*.mp4):"
    find assets -name "*.mp4" | sort
else
    echo "❌ Assets目录不存在"
fi

echo ""
echo "🎯 请选择操作："
echo "1. 🌐 启动Web界面 (streamlit)"
echo "2. 🎬 生成所有视频 (run_manim.py --all)"
echo "3. 📋 列出所有场景 (run_manim.py --list)"
echo "4. 🎯 生成指定模块视频"
echo "5. 🎬 生成指定场景视频"
echo "6. 📁 查看Assets目录结构"
echo "7. 🧹 清理所有视频文件"
echo "8. 📊 统计信息"
echo "9. 🚪 退出"
echo ""

read -p "请输入选项 (1-9): " choice

case $choice in
    1)
        echo "🌐 启动Streamlit Web界面..."
        echo "📱 浏览器将自动打开 http://localhost:8501"
        streamlit run app.py
        ;;
    2)
        echo "🎬 开始生成所有动画视频..."
        python3 run_manim.py --all --quality medium
        echo ""
        echo "✅ 生成完成！查看Assets目录："
        tree assets 2>/dev/null || ls -la assets/
        ;;
    3)
        echo "📋 所有可用场景："
        python3 run_manim.py --list
        ;;
    4)
        echo "📋 可用模块："
        python3 run_manim.py --list
        echo ""
        read -p "请输入模块名称: " module
        echo "🎬 生成模块 $module 的所有视频..."
        python3 run_manim.py --module $module --quality medium
        ;;
    5)
        echo "📋 可用模块："
        python3 run_manim.py --list
        echo ""
        read -p "请输入模块名称: " module
        read -p "请输入场景名称: " scene
        echo "🎬 生成场景: $module - $scene"
        python3 run_manim.py --module $module --scene $scene --quality medium
        ;;
    6)
        echo "📁 Assets目录结构："
        if [ -d "assets" ]; then
            echo "📂 目录结构："
            tree assets 2>/dev/null || find assets -type d | sort
            echo ""
            echo "📋 视频文件 (./assets/**/*.mp4):"
            find assets -name "*.mp4" | sort
        else
            echo "❌ Assets目录不存在"
        fi
        ;;
    7)
        echo "🧹 清理所有视频文件..."
        read -p "确认清理所有视频？(y/N): " confirm
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            video_count=$(find assets -name "*.mp4" | wc -l)
            find assets -name "*.mp4" -delete 2>/dev/null
            echo "✅ 已清理 $video_count 个视频文件"
        else
            echo "❌ 取消清理"
        fi
        ;;
    8)
        echo "📊 统计信息："
        if [ -d "assets" ]; then
            total_videos=$(find assets -name "*.mp4" | wc -l)
            echo "📁 总视频数量: $total_videos"
            echo ""
            echo "📂 按模块分类："
            for dir in assets/*/; do
                if [ -d "$dir" ]; then
                    module_name=$(basename "$dir")
                    video_count=$(find "$dir" -name "*.mp4" | wc -l)
                    if [ $video_count -gt 0 ]; then
                        echo "  📂 $module_name/: $video_count 个视频"
                        find "$dir" -name "*.mp4" | sed "s|.*/|  - |"
                    fi
                fi
            done
        else
            echo "❌ Assets目录不存在"
        fi
        ;;
    9)
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo "❌ 无效选项，请选择 1-9"
        ;;
    8)
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo "❌ 无效选项，请选择 1-8"
        ;;
esac