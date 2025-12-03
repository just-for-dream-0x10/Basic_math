import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *
from common.base_scene import BaseScene
from utils import create_matrix_data, create_vector_data, smooth_transition

class MarginScene(BaseScene):
    """支持向量机间隔可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "svm"
    
    def construct(self):
        title = self.add_title("支持向量机", "最大间隔分类器")
        
        # 创建数据点
        data_points = self.create_data_points()
        self.add(data_points)
        
        # 演示间隔概念
        self.demonstrate_margin(data_points)
        
        # 显示间隔公式
        self.show_margin_formula()
        
        self.wait(2)
    
    def create_data_points(self):
        """创建二分类数据点"""
        # 正类点
        positive_points = VGroup()
        positive_data = [
            (-2, 2), (-1, 3), (0, 2.5), (1, 3), (2, 2)
        ]
        for x, y in positive_data:
            point = Dot(point=[x, y, 0], color=BLUE, radius=0.1)
            positive_points.add(point)
        
        # 负类点
        negative_points = VGroup()
        negative_data = [
            (-2, -2), (-1, -3), (0, -2.5), (1, -3), (2, -2)
        ]
        for x, y in negative_data:
            point = Dot(point=[x, y, 0], color=RED, radius=0.1)
            negative_points.add(point)
        
        # 添加标签
        pos_label = Text("正类", color=BLUE).scale(0.5)
        pos_label.next_to(positive_points, UP, buff=0.5)
        
        neg_label = Text("负类", color=RED).scale(0.5)
        neg_label.next_to(negative_points, DOWN, buff=0.5)
        
        return VGroup(positive_points, pos_label, negative_points, neg_label)
    
    def demonstrate_margin(self, data_points):
        """演示间隔概念"""
        positive_points = data_points[0]
        negative_points = data_points[2]
        
        # 创建分类超平面
        separator = Line(
            start=[-3, 0, 0],
            end=[3, 0, 0],
            color=GREEN,
            stroke_width=3
        )
        
        # 创建间隔边界
        margin_upper = Line(
            start=[-3, 1, 0],
            end=[3, 1, 0],
            color=YELLOW,
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        margin_lower = Line(
            start=[-3, -1, 0],
            end=[3, -1, 0],
            color=YELLOW,
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        # 添加间隔区域
        margin_region = Rectangle(
            width=6,
            height=2,
            fill_color=YELLOW,
            fill_opacity=0.2,
            stroke_width=0
        )
        margin_region.move_to([0, 0, 0])
        
        # 动画显示
        self.play(Create(separator))
        self.play(FadeIn(margin_region))
        self.play(Create(margin_upper), Create(margin_lower))
        
        # 添加支持向量
        support_vectors = VGroup()
        
        # 找到支持向量（简化版）
        sv_pos = Dot(point=[0, 2.5, 0], color=BLUE, radius=0.15)
        sv_neg = Dot(point=[0, -2.5, 0], color=RED, radius=0.15)
        
        support_vectors.add(sv_pos, sv_neg)
        
        # 高亮支持向量
        for sv in support_vectors:
            highlight = Circle(radius=0.25, color=self.primary_color, stroke_width=2)
            highlight.move_to(sv.get_center())
            self.play(Create(highlight))
        
        # 添加标签
        sv_label = Text("支持向量", color=self.primary_color).scale(0.5)
        sv_label.next_to(support_vectors, RIGHT, buff=1.0)
        self.play(Write(sv_label))
        
        # 添加间隔说明
        margin_text = Text("间隔 = 2", color=YELLOW).scale(0.5)
        margin_text.next_to(margin_upper, UP, buff=0.2)
        self.play(Write(margin_text))
    
    def show_margin_formula(self):
        """显示间隔公式"""
        # 间隔公式
        margin_formula = MathTex(
            "\\text{margin} = \\frac{2}{\\|w\\|}",
            color=self.text_color
        ).scale(0.8)
        
        margin_formula.to_corner(UR, buff=1.0)
        
        # 添加符号说明
        explanations = VGroup(
            Text("w: 法向量", color=self.text_color).scale(0.4),
            Text("最大化间隔 = 最小化||w||", color=self.primary_color).scale(0.4)
        )
        explanations.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        explanations.next_to(margin_formula, DOWN, buff=0.5)
        
        self.play(Write(margin_formula))
        self.play(Write(explanations))

class KernelTrickScene(BaseScene):
    """核技巧可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "svm"
    
    def construct(self):
        title = self.add_title("核技巧", "非线性映射的魔法")
        
        # 演示线性不可分问题
        self.demonstrate_linear_inseparable()
        
        # 演示核映射
        self.demonstrate_kernel_mapping()
        
        # 显示核函数公式
        self.show_kernel_formula()
        
        self.wait(2)
    
    def demonstrate_linear_inseparable(self):
        """演示线性不可分问题"""
        # 创建XOR类型数据
        data_points = VGroup()
        
        # 第一象限和第三象限为正类
        positive_data = [(1, 1), (1.5, 1.5), (-1, -1), (-1.5, -1.5)]
        for x, y in positive_data:
            point = Dot(point=[x, y, 0], color=BLUE, radius=0.1)
            data_points.add(point)
        
        # 第二象限和第四象限为负类
        negative_data = [(-1, 1), (-1.5, 1.5), (1, -1), (1.5, -1.5)]
        for x, y in negative_data:
            point = Dot(point=[x, y, 0], color=RED, radius=0.1)
            data_points.add(point)
        
        self.add(data_points)
        
        # 尝试用直线分割
        line1 = Line(start=[-2, 0, 0], end=[2, 0, 0], color=GREEN, stroke_width=2)
        line2 = Line(start=[0, -2, 0], end=[0, 2, 0], color=GREEN, stroke_width=2)
        
        self.play(Create(line1))
        self.play(Create(line2))
        
        # 添加说明
        inseparable_text = Text("线性不可分！", color=RED).scale(0.6)
        inseparable_text.to_edge(UP, buff=1.0)
        self.play(Write(inseparable_text))
        
        self.wait(1)
        self.play(FadeOut(VGroup(line1, line2, inseparable_text)))
    
    def demonstrate_kernel_mapping(self):
        """演示核映射"""
        # 简化为2D演示
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": self.text_color, "stroke_width": 1}
        )
        
        # 映射后的数据点 (x, x²+y²)
        mapped_points = VGroup()
        original_data = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        colors = [BLUE, RED, RED, BLUE]
        
        for (x, y), color in zip(original_data, colors):
            z = x**2 + y**2
            point = Dot(point=[x, z, 0], color=color, radius=0.1)
            mapped_points.add(point)
        
        # 添加分离线
        separator = Line(
            start=[-2, 2, 0],
            end=[2, 2, 0],
            color=GREEN,
            stroke_width=3
        )
        
        self.play(Create(axes))
        self.play(FadeIn(mapped_points))
        self.play(Create(separator))
        
        # 添加说明
        separable_text = Text("映射后线性可分！", color=GREEN).scale(0.5)
        separable_text.to_edge(UP, buff=1.0)
        self.play(Write(separable_text))
        
        self.wait(1)
    
    def show_kernel_formula(self):
        """显示核函数公式"""
        # 核函数公式
        kernel_formula = VGroup(
            MathTex("K(x_i, x_j) = \\phi(x_i)^T \\phi(x_j)", color=self.text_color).scale(0.7),
            MathTex("K(x_i, x_j) = \\exp(-\\gamma \\|x_i - x_j\\|^2)", color=self.primary_color).scale(0.6)
        )
        
        kernel_formula.arrange(DOWN, buff=0.5)
        kernel_formula.to_corner(UR, buff=1.0)
        
        # 添加标签
        linear_label = Text("线性核", color=self.text_color).scale(0.4)
        linear_label.next_to(kernel_formula[0], LEFT, buff=0.5)
        
        rbf_label = Text("RBF核", color=self.primary_color).scale(0.4)
        rbf_label.next_to(kernel_formula[1], LEFT, buff=0.5)
        
        # 添加说明
        explanations = VGroup(
            Text("无需显式计算φ(x)", color=self.primary_color).scale(0.4),
            Text("直接计算内积", color=self.primary_color).scale(0.4)
        )
        explanations.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        explanations.next_to(kernel_formula, DOWN, buff=0.5)
        
        self.play(Write(kernel_formula), Write(linear_label), Write(rbf_label))
        self.play(Write(explanations))

class DualProblemScene(BaseScene):
    """对偶问题可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "svm"
    
    def construct(self):
        title = self.add_title("对偶问题", "拉格朗日对偶的优雅")
        
        # 展示原问题
        self.show_primal_problem()
        
        # 展示对偶问题
        self.show_dual_problem()
        
        # 展示对偶关系
        self.show_duality_relationship()
        
        self.wait(2)
    
    def show_primal_problem(self):
        """展示原问题"""
        # 原问题公式
        primal_formula = MathTex(
            "\\min_{w,b} \\frac{1}{2}\\|w\\|^2",
            "\\text{s.t. } y_i(w^T x_i + b) \\ge 1, \\forall i",
            color=self.text_color
        ).scale(0.7)
        
        primal_formula.arrange(DOWN, buff=0.5)
        primal_formula.to_corner(UL, buff=1.0)
        
        primal_label = Text("原问题", color=self.text_color).scale(0.6)
        primal_label.next_to(primal_formula, UP, buff=0.3)
        
        self.play(Write(primal_label), Write(primal_formula))
        
        # 添加几何解释
        geometry_text = Text("最小化法向量长度", color=self.primary_color).scale(0.5)
        geometry_text.next_to(primal_formula, DOWN, buff=0.5)
        self.play(Write(geometry_text))
        
        self.wait(1)
        self.play(FadeOut(VGroup(primal_label, primal_formula, geometry_text)))
    
    def show_dual_problem(self):
        """展示对偶问题"""
        # 对偶问题公式
        dual_formula = VGroup(
            MathTex("\\max_\\alpha \\sum_{i=1}^n \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)", color=self.text_color).scale(0.6),
            MathTex("\\text{s.t. } 0 \\le \\alpha_i \\le C", color=self.text_color).scale(0.6),
            MathTex("\\sum_{i=1}^n \\alpha_i y_i = 0", color=self.text_color).scale(0.6)
        )
        
        dual_formula.arrange(DOWN, buff=0.3)
        dual_formula.to_corner(UR, buff=1.0)
        
        dual_label = Text("对偶问题", color=self.text_color).scale(0.6)
        dual_label.next_to(dual_formula, UP, buff=0.3)
        
        self.play(Write(dual_label), Write(dual_formula))
        
        # 添加优势说明
        advantages = VGroup(
            Text("✓ 只涉及内积计算", color=self.primary_color).scale(0.4),
            Text("✓ 自然引入核函数", color=self.primary_color).scale(0.4),
            Text("✓ 支持向量对应α_i > 0", color=self.primary_color).scale(0.4)
        )
        advantages.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        advantages.next_to(dual_formula, DOWN, buff=0.5)
        
        self.play(Write(advantages))
    
    def show_duality_relationship(self):
        """展示对偶关系"""
        # 对偶关系图
        primal_box = Rectangle(
            width=3,
            height=2,
            fill_color=BLUE,
            fill_opacity=0.2,
            stroke_color=BLUE,
            stroke_width=2
        )
        
        dual_box = Rectangle(
            width=3,
            height=2,
            fill_color=GREEN,
            fill_opacity=0.2,
            stroke_color=GREEN,
            stroke_width=2
        )
        
        primal_box.to_corner(DL, buff=1.0)
        dual_box.to_corner(DR, buff=1.0)
        
        primal_text = Text("原问题", color=BLUE).scale(0.5)
        primal_text.move_to(primal_box.get_center())
        
        dual_text = Text("对偶问题", color=GREEN).scale(0.5)
        dual_text.move_to(dual_box.get_center())
        
        # 双向箭头
        arrow_forward = Arrow(
            start=primal_box.get_right(),
            end=dual_box.get_left(),
            color=self.primary_color,
            stroke_width=2
        )
        
        arrow_backward = Arrow(
            start=dual_box.get_left(),
            end=primal_box.get_right(),
            color=self.primary_color,
            stroke_width=2
        )
        
        arrow_backward.shift(UP * 0.5)
        arrow_forward.shift(DOWN * 0.5)
        
        # 标签
        forward_label = Text("拉格朗日对偶", color=self.primary_color).scale(0.3)
        forward_label.next_to(arrow_forward, DOWN, buff=0.1)
        
        backward_label = Text("强对偶性", color=self.primary_color).scale(0.3)
        backward_label.next_to(arrow_backward, UP, buff=0.1)
        
        # 动画显示
        self.play(
            FadeIn(VGroup(primal_box, primal_text)),
            FadeIn(VGroup(dual_box, dual_text))
        )
        
        self.play(Create(arrow_forward), Write(forward_label))
        self.play(Create(arrow_backward), Write(backward_label))
        
        # 添加结论
        conclusion = Text(
            "原问题和对偶问题有相同的最优值",
            color=self.text_color
        ).scale(0.5)
        conclusion.to_edge(DOWN, buff=1.0)
        self.play(Write(conclusion))