import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *
from common.base_scene import BaseScene
from utils import create_matrix_data, create_vector_data, smooth_transition

class SGDScene(BaseScene):
    """随机梯度下降可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "optimizer"
    
    def construct(self):
        title = self.add_title("随机梯度下降", "SGD的基本原理")
        
        # 创建损失函数曲面
        surface = self.create_loss_surface()
        self.add(surface)
        
        # 演示SGD过程
        self.demonstrate_sgd()
        
        # 显示SGD公式
        self.show_sgd_formula()
        
        self.wait(2)
    
    def create_loss_surface(self):
        """创建损失函数曲面"""
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": self.text_color, "stroke_width": 1}
        )
        
        # 创建等高线
        contours = VGroup()
        for radius in np.linspace(0.5, 2.5, 5):
            circle = Circle(radius=radius, color=self.grid_color, stroke_width=1, stroke_opacity=0.5)
            contours.add(circle)
        
        # 添加最低点标记
        minimum = Dot(point=[0, 0, 0], color=GREEN, radius=0.15)
        min_label = Text("最小值", color=GREEN).scale(0.4)
        min_label.next_to(minimum, DOWN, buff=0.2)
        
        return VGroup(axes, contours, minimum, min_label)
    
    def demonstrate_sgd(self):
        """演示SGD过程"""
        # 起始点
        start_point = Dot(point=[2, 2, 0], color=RED, radius=0.1)
        self.play(FadeIn(start_point))
        
        # SGD路径（随机性更强）
        path_points = [
            [2, 2], [1.3, 1.8], [0.8, 1.2], [0.5, 0.8], [0.2, 0.3], [0, 0]
        ]
        
        # 创建路径线
        path_lines = VGroup()
        current_point = start_point.get_center()
        
        for i, next_pos in enumerate(path_points[1:]):
            next_point = [next_pos[0], next_pos[1], 0]
            path_line = Line(current_point, next_point, color=self.primary_color, stroke_width=2)
            path_lines.add(path_line)
            
            # 移动的点
            moving_dot = Dot(point=current_point, color=RED, radius=0.1)
            
            # 显示随机梯度方向
            gradient_arrow = Arrow(
                start=current_point,
                end=current_point + (np.array(next_point) - np.array(current_point)) * 1.2,
                color=ORANGE,
                stroke_width=2,
                max_stroke_width_to_length_ratio=10
            )
            
            # 步数标签
            step_text = Text(f"Step {i+1}", color=self.text_color).scale(0.4)
            step_text.next_to(moving_dot, UP, buff=0.3)
            
            # 添加随机性说明
            if i == 0:
                noise_text = Text("随机噪声影响", color=self.primary_color).scale(0.3)
                noise_text.next_to(gradient_arrow, RIGHT, buff=0.2)
                self.play(Write(noise_text))
            
            self.play(
                Create(path_line),
                Create(gradient_arrow),
                Write(step_text),
                moving_dot.animate.move_to(next_point),
                run_time=0.8
            )
            
            self.play(FadeOut(gradient_arrow), FadeOut(step_text))
            current_point = next_point
        
        # 最终点
        final_dot = Dot(point=[0, 0, 0], color=GREEN, radius=0.15)
        self.play(FadeOut(moving_dot), FadeIn(final_dot))
        
        # 添加收敛标记
        convergence_text = Text("收敛！", color=GREEN).scale(0.6)
        convergence_text.next_to(final_dot, UP, buff=0.5)
        self.play(Write(convergence_text))
    
    def show_sgd_formula(self):
        """显示SGD公式"""
        # SGD更新公式
        update_formula = MathTex(
            "\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t; x_i)",
            color=self.text_color
        ).scale(0.7)
        
        update_formula.to_corner(UR, buff=1.0)
        
        # 添加符号说明
        explanations = VGroup(
            Text("x_i: 随机样本", color=self.primary_color).scale(0.4),
            Text("η: 学习率", color=self.text_color).scale(0.4),
            Text("∇L: 样本梯度", color=self.text_color).scale(0.4)
        )
        explanations.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        explanations.next_to(update_formula, DOWN, buff=0.5)
        
        self.play(Write(update_formula))
        self.play(Write(explanations))

class MomentumScene(BaseScene):
    """动量优化器可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "optimizer"
    
    def construct(self):
        title = self.add_title("动量优化器", "Momentum的加速效果")
        
        # 创建损失函数曲面
        surface = self.create_loss_surface()
        self.add(surface)
        
        # 对比SGD和Momentum
        self.compare_sgd_momentum()
        
        # 显示Momentum公式
        self.show_momentum_formula()
        
        self.wait(2)
    
    def create_loss_surface(self):
        """创建损失函数曲面"""
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": self.text_color, "stroke_width": 1}
        )
        
        # 创建等高线
        contours = VGroup()
        for radius in np.linspace(0.5, 2.5, 5):
            circle = Circle(radius=radius, color=self.grid_color, stroke_width=1, stroke_opacity=0.5)
            contours.add(circle)
        
        # 添加最低点标记
        minimum = Dot(point=[0, 0, 0], color=GREEN, radius=0.15)
        min_label = Text("最小值", color=GREEN).scale(0.4)
        min_label.next_to(minimum, DOWN, buff=0.2)
        
        return VGroup(axes, contours, minimum, min_label)
    
    def compare_sgd_momentum(self):
        """对比SGD和Momentum"""
        # SGD路径（振荡）
        sgd_path = [
            [2, 2], [1.8, 1.5], [1.2, 1.8], [0.8, 1.2], [0.5, 0.8], [0.2, 0.3], [0, 0]
        ]
        
        # Momentum路径（更平滑）
        momentum_path = [
            [2, 2], [1.5, 1.5], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2], [0, 0]
        ]
        
        # 绘制SGD路径
        sgd_lines = VGroup()
        sgd_points = VGroup()
        
        for i in range(len(sgd_path) - 1):
            start = [sgd_path[i][0], sgd_path[i][1], 0]
            end = [sgd_path[i+1][0], sgd_path[i+1][1], 0]
            
            line = Line(start, end, color=RED, stroke_width=2)
            point = Dot(point=start, color=RED, radius=0.08)
            
            sgd_lines.add(line)
            sgd_points.add(point)
        
        # 绘制Momentum路径
        momentum_lines = VGroup()
        momentum_points = VGroup()
        
        for i in range(len(momentum_path) - 1):
            start = [momentum_path[i][0], momentum_path[i][1], 0]
            end = [momentum_path[i+1][0], momentum_path[i+1][1], 0]
            
            line = Line(start, end, color=BLUE, stroke_width=2)
            point = Dot(point=start, color=BLUE, radius=0.08)
            
            momentum_lines.add(line)
            momentum_points.add(point)
        
        # 添加标签
        sgd_label = Text("SGD", color=RED).scale(0.5)
        sgd_label.next_to(sgd_points[0], UP, buff=0.3)
        
        momentum_label = Text("Momentum", color=BLUE).scale(0.5)
        momentum_label.next_to(momentum_points[0], DOWN, buff=0.3)
        
        # 动画显示
        self.play(
            FadeIn(VGroup(sgd_points, sgd_label)),
            FadeIn(VGroup(momentum_points, momentum_label))
        )
        
        # 逐步显示路径
        max_lines = min(len(sgd_lines), len(momentum_lines))
        for i in range(max_lines):
            animations = []
            if i < len(sgd_lines):
                animations.append(Create(sgd_lines[i]))
            if i < len(momentum_lines):
                animations.append(Create(momentum_lines[i]))
            if animations:
                self.play(*animations, run_time=0.5)
        
        # 添加说明
        explanation = Text(
            "Momentum通过累积动量减少振荡，加速收敛",
            color=self.text_color
        ).scale(0.5)
        explanation.to_edge(DOWN, buff=1.0)
        self.play(Write(explanation))
    
    def show_momentum_formula(self):
        """显示Momentum公式"""
        # Momentum更新公式
        update_formula = MathTex(
            "v_t = \\beta v_{t-1} + \\nabla L(\\theta_t)",
            "\\theta_{t+1} = \\theta_t - \\eta v_t",
            color=self.text_color
        ).scale(0.6)
        
        update_formula.arrange(DOWN, buff=0.5)
        update_formula.to_corner(UR, buff=1.0)
        
        # 添加符号说明
        explanations = VGroup(
            Text("v_t: 动量", color=self.primary_color).scale(0.4),
            Text("β: 动量系数", color=self.text_color).scale(0.4),
            Text("η: 学习率", color=self.text_color).scale(0.4)
        )
        explanations.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        explanations.next_to(update_formula, DOWN, buff=0.5)
        
        self.play(Write(update_formula))
        self.play(Write(explanations))

class AdamScene(BaseScene):
    """Adam优化器可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "optimizer"
    
    def construct(self):
        title = self.add_title("Adam优化器", "自适应学习率的王者")
        
        # 创建损失函数曲面
        surface = self.create_loss_surface()
        self.add(surface)
        
        # 演示Adam的优势
        self.demonstrate_adam_advantages()
        
        # 显示Adam公式
        self.show_adam_formula()
        
        self.wait(2)
    
    def create_loss_surface(self):
        """创建损失函数曲面"""
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": self.text_color, "stroke_width": 1}
        )
        
        # 创建等高线
        contours = VGroup()
        for radius in np.linspace(0.5, 2.5, 5):
            circle = Circle(radius=radius, color=self.grid_color, stroke_width=1, stroke_opacity=0.5)
            contours.add(circle)
        
        # 添加最低点标记
        minimum = Dot(point=[0, 0, 0], color=GREEN, radius=0.15)
        min_label = Text("最小值", color=GREEN).scale(0.4)
        min_label.next_to(minimum, DOWN, buff=0.2)
        
        return VGroup(axes, contours, minimum, min_label)
    
    def demonstrate_adam_advantages(self):
        """演示Adam的优势"""
        # 创建复杂的损失函数地形（鞍点）
        saddle_point = Dot(point=[1, 0, 0], color=ORANGE, radius=0.12)
        saddle_label = Text("鞍点", color=ORANGE).scale(0.4)
        saddle_label.next_to(saddle_point, UP, buff=0.2)
        
        # 不同优化器的路径
        sgd_path = [[2, 1], [1.5, 0.8], [1.2, 0.2], [1.1, -0.1], [1.0, 0.0], [0.8, 0.0], [0.5, 0.0], [0, 0]]
        adam_path = [[2, 1], [1.3, 0.6], [0.8, 0.2], [0.3, 0.0], [0, 0]]
        
        # 绘制路径
        sgd_lines = self.create_path(sgd_path, RED)
        adam_lines = self.create_path(adam_path, self.primary_color)
        
        # 添加标签
        sgd_label = Text("SGD: 困在鞍点", color=RED).scale(0.4)
        sgd_label.to_edge(LEFT, buff=1.0)
        
        adam_label = Text("Adam: 快速逃离", color=self.primary_color).scale(0.4)
        adam_label.to_edge(RIGHT, buff=1.0)
        
        # 动画显示
        self.play(FadeIn(saddle_point), Write(saddle_label))
        self.play(Write(sgd_label), Write(adam_label))
        
        # 逐步显示路径
        for i in range(max(len(sgd_lines), len(adam_lines))):
            if i < len(sgd_lines):
                self.play(Create(sgd_lines[i]), run_time=0.3)
            if i < len(adam_lines):
                self.play(Create(adam_lines[i]), run_time=0.3)
        
        # 添加说明
        advantages = VGroup(
            Text("✓ 自适应学习率", color=self.primary_color).scale(0.5),
            Text("✓ 动量加速", color=self.primary_color).scale(0.5),
            Text("✓ 偏差校正", color=self.primary_color).scale(0.5)
        )
        advantages.arrange(DOWN, buff=0.3)
        advantages.to_edge(DOWN, buff=1.0)
        
        self.play(Write(advantages))
    
    def create_path(self, path_points, color):
        """创建路径"""
        lines = VGroup()
        for i in range(len(path_points) - 1):
            start = [path_points[i][0], path_points[i][1], 0]
            end = [path_points[i+1][0], path_points[i+1][1], 0]
            line = Line(start, end, color=color, stroke_width=2)
            lines.add(line)
        return lines
    
    def show_adam_formula(self):
        """显示Adam公式"""
        # Adam更新公式
        update_formula = VGroup(
            MathTex("m_t = \\beta_1 m_{t-1} + (1-\\beta_1) \\nabla L", color=self.text_color).scale(0.5),
            MathTex("v_t = \\beta_2 v_{t-1} + (1-\\beta_2) \\nabla L^2", color=self.text_color).scale(0.5),
            MathTex("\\hat{m}_t = m_t / (1-\\beta_1^t)", color=self.text_color).scale(0.5),
            MathTex("\\hat{v}_t = v_t / (1-\\beta_2^t)", color=self.text_color).scale(0.5),
            MathTex("\\theta_{t+1} = \\theta_t - \\eta \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)", color=self.text_color).scale(0.5)
        )
        
        update_formula.arrange(DOWN, buff=0.3)
        update_formula.to_corner(UR, buff=0.5)
        
        # 添加符号说明
        explanations = VGroup(
            Text("m_t: 一阶动量", color=self.primary_color).scale(0.35),
            Text("v_t: 二阶动量", color=self.primary_color).scale(0.35),
            Text("β₁, β₂: 衰减率", color=self.text_color).scale(0.35),
            Text("ε: 数值稳定项", color=self.text_color).scale(0.35)
        )
        explanations.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        explanations.next_to(update_formula, DOWN, buff=0.3)
        
        self.play(Write(update_formula))
        self.play(Write(explanations))