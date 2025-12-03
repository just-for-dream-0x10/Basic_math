import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *
from common.base_scene import BaseScene

class LeastSquaresScene(BaseScene):
    """最小二乘法可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "loss"
    
    def construct(self):
        # 标题
        title = self.add_title("最小二乘法", "最小化预测误差的平方和")
        
        # 创建坐标系
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=8,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 24}
        ).shift(DOWN * 0.5)
        
        # 数据点
        data_points = [
            (1, 2.5), (2, 3.8), (3, 4.2), (4, 5.5),
            (5, 6.8), (6, 7.2), (7, 8.5), (8, 8.8)
        ]
        
        # 绘制数据点
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            for x, y in data_points
        ])
        
        # 拟合直线 (简单线性回归)
        x_vals = np.array([p[0] for p in data_points])
        y_vals = np.array([p[1] for p in data_points])
        
        # 计算最小二乘解
        n = len(x_vals)
        x_mean = np.mean(x_vals)
        y_mean = np.mean(y_vals)
        
        slope = np.sum((x_vals - x_mean) * (y_vals - y_mean)) / np.sum((x_vals - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        # 绘制拟合线
        line = axes.plot(
            lambda x: slope * x + intercept,
            x_range=[0.5, 8.5],
            color=RED
        )
        
        # 显示动画
        self.play(Create(axes))
        self.play(FadeIn(dots))
        self.wait(0.5)
        
        # 显示公式
        formula = MathTex(
            r"L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
            font_size=36
        ).to_edge(UP).shift(DOWN * 0.8)
        
        self.play(Write(formula))
        self.wait(0.5)
        
        # 显示拟合线
        self.play(Create(line))
        self.wait(0.5)
        
        # 显示残差线
        residuals = VGroup()
        for x, y in data_points:
            y_pred = slope * x + intercept
            residual_line = DashedLine(
                axes.c2p(x, y),
                axes.c2p(x, y_pred),
                color=YELLOW,
                stroke_width=2
            )
            residuals.add(residual_line)
        
        self.play(Create(residuals))
        self.wait(1)
        
        # 显示解释文字
        explanation = Text(
            "残差：预测值与真实值的差异",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN)
        
        self.play(Write(explanation))
        self.wait(2)
        
        # 淡出
        self.play(
            FadeOut(VGroup(axes, dots, line, residuals, formula, explanation))
        )
