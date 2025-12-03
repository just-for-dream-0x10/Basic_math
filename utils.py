import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
from pathlib import Path
from config import COLORS, MATH_SYMBOLS

def parse_latex_formula(text):
    """解析LaTeX公式，转换为Manim可用的MathTex格式"""
    # 简单的LaTeX符号替换
    for symbol, replacement in MATH_SYMBOLS.items():
        text = text.replace(f"\\{symbol}", replacement)
    
    # 处理常见的数学环境
    text = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', text)
    text = re.sub(r'\$(.*?)\$', r'\\(\1\\)', text)
    
    return text

def create_gradient_colors(start_color, end_color, steps):
    """创建渐变色列表"""
    start_rgb = plt.colors.hex2color(start_color)
    end_rgb = plt.colors.hex2color(end_color)
    
    gradient = []
    for i in range(steps):
        ratio = i / (steps - 1) if steps > 1 else 0
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
        gradient.append((r, g, b))
    
    return gradient

def get_module_color(module_name, alpha=1.0):
    """获取模块对应的颜色"""
    from config import MODULES
    if module_name in MODULES:
        color_hex = MODULES[module_name]["color"]
        rgb = plt.colors.hex2color(color_hex)
        return (*rgb, alpha)
    return COLORS["primary"]

def format_math_label(label):
    """格式化数学标签"""
    # 移除多余的空格和特殊字符
    label = label.strip()
    
    # 处理常见的数学符号
    label = label.replace("^2", "²")
    label = label.replace("^3", "³")
    label = label.replace("_", "")
    
    return label

def generate_animation_params(duration=3.0, fps=30):
    """生成动画参数"""
    return {
        "run_time": duration,
        "rate_func": lambda t: t,  # linear
        "frame_rate": fps
    }

def create_matrix_data(rows, cols, value_range=(-2, 2)):
    """创建随机矩阵数据用于可视化"""
    return np.random.uniform(value_range[0], value_range[1], (rows, cols))

def create_vector_data(dim, value_range=(-2, 2)):
    """创建随机向量数据用于可视化"""
    return np.random.uniform(value_range[0], value_range[1], dim)

def smooth_transition(values, window_size=3):
    """平滑数据过渡"""
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    
    return smoothed

def extract_key_concepts(markdown_content, max_concepts=5):
    """从Markdown内容中提取关键概念"""
    # 简单的关键词提取：查找标题和加粗文本
    concepts = []
    
    # 提取标题
    headers = re.findall(r'^#+\s+(.+)$', markdown_content, re.MULTILINE)
    concepts.extend(headers)
    
    # 提取加粗文本
    bold_text = re.findall(r'\*\*(.+?)\*\*', markdown_content)
    concepts.extend(bold_text)
    
    # 去重并限制数量
    unique_concepts = list(dict.fromkeys(concepts))
    return unique_concepts[:max_concepts]

def calculate_animation_steps(total_duration, step_durations):
    """计算动画步骤的时间分配"""
    total_steps = sum(step_durations)
    if total_steps == 0:
        return [1.0] * len(step_durations)
    
    ratios = [d / total_steps for d in step_durations]
    return [ratio * total_duration for ratio in ratios]