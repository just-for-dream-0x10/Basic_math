"""
交互式可视化基础模块
提供通用工具函数和基类
"""

import numpy as np
import streamlit as st
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.font_utils import configure_chinese_font

# 配置中文字体
configure_chinese_font()


def compute_gradient(loss_fn, x, y, h=1e-5):
    """数值计算梯度"""
    grad_x = (loss_fn(x + h, y) - loss_fn(x - h, y)) / (2 * h)
    grad_y = (loss_fn(x, y + h) - loss_fn(x, y - h)) / (2 * h)
    return np.array([grad_x, grad_y])


def get_loss_function(name):
    """
    获取损失函数
    
    返回: (loss_fn, x_range, y_range, title)
    """
    functions = {
        'rosenbrock': (
            lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2,
            (-2.5, 2.5), (-1, 3), "Rosenbrock函数"
        ),
        'sphere': (
            lambda x, y: x**2 + y**2,
            (-3, 3), (-3, 3), "球面函数"
        ),
        'beale': (
            lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
            (-4.5, 4.5), (-4.5, 4.5), "Beale函数"
        ),
        'himmelblau': (
            lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
            (-5, 5), (-5, 5), "Himmelblau函数"
        ),
        'ackley': (
            lambda x, y: (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - 
                         np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20),
            (-5, 5), (-5, 5), "Ackley函数"
        ),
        'rastrigin': (
            lambda x, y: 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y),
            (-5.12, 5.12), (-5.12, 5.12), "Rastrigin函数"
        ),
        'booth': (
            lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2,
            (-10, 10), (-10, 10), "Booth函数"
        ),
        'matyas': (
            lambda x, y: 0.26 * (x**2 + y**2) - 0.48 * x * y,
            (-10, 10), (-10, 10), "Matyas函数"
        ),
        'levi': (
            lambda x, y: (np.sin(3*np.pi*x)**2 + (x-1)**2 * (1 + np.sin(3*np.pi*y)**2) + 
                         (y-1)**2 * (1 + np.sin(2*np.pi*y)**2)),
            (-10, 10), (-10, 10), "Levi N.13函数"
        ),
        'easom': (
            lambda x, y: -np.cos(x) * np.cos(y) * np.exp(-((x-np.pi)**2 + (y-np.pi)**2)),
            (-10, 10), (-10, 10), "Easom函数"
        )
    }
    
    return functions.get(name, functions['sphere'])


# 损失函数显示名称映射
LOSS_FUNCTION_NAMES = {
    "rosenbrock": "🍌 Rosenbrock (香蕉函数)",
    "sphere": "⚪ Sphere (球面函数)",
    "beale": "🎯 Beale函数",
    "himmelblau": "🏔️ Himmelblau函数",
    "ackley": "🌊 Ackley函数 (多峰)",
    "rastrigin": "⛰️ Rastrigin函数 (高度多峰)",
    "booth": "📦 Booth函数",
    "matyas": "💠 Matyas函数 (简单凸)",
    "levi": "🎪 Levi N.13函数",
    "easom": "🎯 Easom函数 (平坦+尖峰)"
}
