import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *
from common.base_scene import BaseScene

class CrossEntropyScene(BaseScene):
    """交叉熵损失可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "loss"
    
    def construct(self):
        # 标题
        title = self.add_title("交叉熵损失", "衡量概率分布的差异")
        
        # 交叉熵公式
        formula = MathTex(
            r"H(p, q) = -\sum_{i} p_i \log(q_i)",
            font_size=40
        ).to_edge(UP).shift(DOWN * 0.8)
        
        self.play(Write(formula))
        self.wait(1)
        
        # 创建两个概率分布的可视化
        # 真实分布 p
        true_probs = [0.7, 0.2, 0.1]
        # 预测分布 q
        pred_probs = [0.5, 0.3, 0.2]
        
        labels = ["类别 A", "类别 B", "类别 C"]
        
        # 创建柱状图区域
        left_group = VGroup()
        right_group = VGroup()
        
        # 真实分布 (左边)
        true_title = Text("真实分布 p", font_size=28, color=BLUE).shift(LEFT * 3.5 + UP * 1.5)
        left_group.add(true_title)
        
        for i, (label, prob) in enumerate(zip(labels, true_probs)):
            bar = Rectangle(
                width=0.6,
                height=prob * 3,
                fill_color=BLUE,
                fill_opacity=0.7,
                stroke_width=2
            ).shift(LEFT * 3.5 + DOWN * (1.5 - prob * 1.5) + RIGHT * i * 0.8)
            
            label_text = Text(label.split()[-1], font_size=20).next_to(bar, DOWN, buff=0.1)
            prob_text = Text(f"{prob:.1f}", font_size=18).next_to(bar, UP, buff=0.1)
            
            left_group.add(bar, label_text, prob_text)
        
        # 预测分布 (右边)
        pred_title = Text("预测分布 q", font_size=28, color=RED).shift(RIGHT * 3.5 + UP * 1.5)
        right_group.add(pred_title)
        
        for i, (label, prob) in enumerate(zip(labels, pred_probs)):
            bar = Rectangle(
                width=0.6,
                height=prob * 3,
                fill_color=RED,
                fill_opacity=0.7,
                stroke_width=2
            ).shift(RIGHT * 3.5 + DOWN * (1.5 - prob * 1.5) + RIGHT * i * 0.8)
            
            label_text = Text(label.split()[-1], font_size=20).next_to(bar, DOWN, buff=0.1)
            prob_text = Text(f"{prob:.1f}", font_size=18).next_to(bar, UP, buff=0.1)
            
            right_group.add(bar, label_text, prob_text)
        
        # 显示分布
        self.play(FadeIn(left_group), FadeIn(right_group))
        self.wait(1)
        
        # 计算交叉熵
        cross_entropy = -sum(p * np.log(q) for p, q in zip(true_probs, pred_probs))
        
        # 显示计算结果
        result = MathTex(
            f"H(p, q) = {cross_entropy:.3f}",
            font_size=36,
            color=YELLOW
        ).to_edge(DOWN).shift(UP * 0.5)
        
        self.play(Write(result))
        self.wait(1)
        
        # 解释
        explanation = Text(
            "交叉熵越小，预测分布越接近真实分布",
            font_size=24,
            color=GREEN
        ).to_edge(DOWN)
        
        self.play(Write(explanation))
        self.wait(2)
        
        # 淡出
        self.play(
            FadeOut(VGroup(formula, left_group, right_group, result, explanation))
        )
