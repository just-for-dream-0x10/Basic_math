import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *
from common.base_scene import BaseScene
from utils import create_matrix_data, create_vector_data, smooth_transition

class MatrixTransformScene(BaseScene):
    """矩阵变换可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "matrix"
    
    def construct(self):
        # 添加标题
        title = self.add_title("矩阵变换：数据的几何变换", "线性变换的几何意义")
        
        # 添加坐标网格
        grid = self.add_grid()
        
        # 创建原始向量
        original_vectors = self.create_basis_vectors()
        self.add(original_vectors)
        
        # 等待一下
        self.wait(1)
        
        # 创建变换矩阵
        transform_matrix = self.create_transform_matrix()
        self.play(Write(transform_matrix))
        self.wait(1)
        
        # 应用变换
        transformed_vectors = self.apply_matrix_transform(original_vectors, transform_matrix)
        
        # 显示变换效果
        self.show_transform_effect(original_vectors, transformed_vectors)
        
        # 演示特征向量
        self.demonstrate_eigenvectors(transform_matrix)
        
        self.wait(2)
    
    def create_basis_vectors(self):
        """创建基向量"""
        # 标准基向量
        e1 = Arrow(ORIGIN, RIGHT * 2, color=RED, stroke_width=3)
        e1_label = MathTex("\\mathbf{e}_1", color=RED).scale(0.7).next_to(e1.get_end(), RIGHT)
        
        e2 = Arrow(ORIGIN, UP * 2, color=GREEN, stroke_width=3)
        e2_label = MathTex("\\mathbf{e}_2", color=GREEN).scale(0.7).next_to(e2.get_end(), UP)
        
        # 额外的向量
        v1 = Arrow(ORIGIN, RIGHT + UP, color=BLUE, stroke_width=2, fill_opacity=0.7)
        v1_label = MathTex("\\mathbf{v}_1", color=BLUE).scale(0.7).next_to(v1.get_end(), RIGHT + UP)
        
        vectors = VGroup(e1, e1_label, e2, e2_label, v1, v1_label)
        return vectors
    
    def create_transform_matrix(self):
        """创建变换矩阵显示"""
        # 示例变换矩阵：旋转+缩放
        matrix_values = [
            [2, -1],
            [1, 1]
        ]
        
        matrix = Matrix(
            matrix_values,
            bracket_h_buff=0.1,
            bracket_v_buff=0.1,
            element_to_mobject_config={"color": self.primary_color}
        ).scale(0.8)
        
        matrix.to_corner(UL, buff=1.5)
        
        # 添加矩阵标签
        matrix_label = Text("变换矩阵 A", color=self.text_color).scale(0.6)
        matrix_label.next_to(matrix, UP, buff=0.3)
        
        self.add(matrix_label)
        return VGroup(matrix, matrix_label)
    
    def apply_matrix_transform(self, vectors, matrix_display):
        """应用矩阵变换"""
        # 实际的变换矩阵
        A = np.array([[2, -1], [1, 1]])
        
        # 变换后的向量
        transformed = VGroup()
        
        # 变换基向量
        e1_end = A @ np.array([2, 0])
        e1_transformed = Arrow(ORIGIN, [e1_end[0], e1_end[1], 0], color=YELLOW, stroke_width=3)
        e1_label = MathTex("A\\mathbf{e}_1", color=YELLOW).scale(0.7).next_to(e1_transformed.get_end(), RIGHT)
        transformed.add(e1_transformed, e1_label)
        
        e2_end = A @ np.array([0, 2])
        e2_transformed = Arrow(ORIGIN, [e2_end[0], e2_end[1], 0], color=ORANGE, stroke_width=3)
        e2_label = MathTex("A\\mathbf{e}_2", color=ORANGE).scale(0.7).next_to(e2_transformed.get_end(), UP)
        transformed.add(e2_transformed, e2_label)
        
        # 变换额外向量
        v1_end = A @ np.array([1, 1])
        v1_transformed = Arrow(ORIGIN, [v1_end[0], v1_end[1], 0], color=PURPLE, stroke_width=2, fill_opacity=0.7)
        v1_label = MathTex("A\\mathbf{v}_1", color=PURPLE).scale(0.7).next_to(v1_transformed.get_end(), RIGHT + UP)
        transformed.add(v1_transformed, v1_label)
        
        # 先淡出原始向量，再显示变换后的向量
        self.play(
            FadeOut(vectors[0]),
            FadeOut(vectors[1]), 
            FadeOut(vectors[2]),
            FadeOut(vectors[3]),
            FadeOut(vectors[4]),
            FadeOut(vectors[5]),
            run_time=1
        )
        
        # 短暂停顿，强调原图像消失
        self.wait(0.5)
        
        # 显示变换后的向量
        self.play(
            GrowArrow(e1_transformed),
            Write(e1_label),
            GrowArrow(e2_transformed),
            Write(e2_label),
            GrowArrow(v1_transformed),
            Write(v1_label),
            run_time=1.5
        )
        
        return transformed
    
    def show_transform_effect(self, original, transformed):
        """显示变换效果"""
        # 添加数学公式说明
        formula = MathTex(r"\mathbf{y} = A\mathbf{x}", color=self.primary_color).scale(0.8)
        formula.to_edge(DOWN, buff=1.0)
        self.play(Write(formula))
        self.wait(2)
        self.play(FadeOut(formula))
    
    def demonstrate_eigenvectors(self, matrix_display):
        """演示特征向量"""
        # 计算特征值和特征向量
        A = np.array([[2, -1], [1, 1]])
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # 创建特征向量可视化
        eigen_group = VGroup()
        
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # 特征向量
            eigen_vec = Arrow(ORIGIN, [vec[0] * 2, vec[1] * 2, 0], color=self.primary_color, stroke_width=4)
            eigen_vec_label = MathTex(f"\\mathbf{{v}}_{i+1}", color=self.primary_color).scale(0.6)
            eigen_vec_label.next_to(eigen_vec.get_end(), RIGHT)
            
            eigen_group.add(eigen_vec, eigen_vec_label)
            
            # 特征值标注
            val_text = MathTex(f"\\lambda_{i+1} = {val:.2f}", color=self.primary_color).scale(0.5)
            val_text.next_to(eigen_vec.get_start(), LEFT)
            eigen_group.add(val_text)
        
        # 添加特征向量说明
        eigen_title = Text("特征向量：变换方向不变的向量", color=self.primary_color).scale(0.6)
        eigen_title.to_edge(DOWN, buff=1.5)
        
        self.play(Write(eigen_title))
        self.play(Create(eigen_group), run_time=2)
        self.wait(2)

class SVDScene(BaseScene):
    """SVD分解可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "matrix"
    
    def construct(self):
        title = self.add_title("奇异值分解 (SVD)", "矩阵的几何分解")
        
        grid = self.add_grid()
        
        # 创建原始形状
        original_shape = self.create_unit_square()
        self.add(original_shape)
        self.wait(1)
        
        # SVD分解步骤
        self.demonstrate_svd_steps(original_shape)
        
        self.wait(2)
    
    def create_unit_square(self):
        """创建单位正方形"""
        square = Square(side_length=2, color=BLUE, fill_opacity=0.3)
        square_label = Text("单位正方形", color=BLUE).scale(0.5)
        square_label.next_to(square, DOWN)
        return VGroup(square, square_label)
    
    def demonstrate_svd_steps(self, original_shape):
        """演示SVD的三个步骤"""
        # SVD公式
        svd_formula = MathTex("A = U \\Sigma V^T", color=self.text_color).scale(0.8)
        svd_formula.to_corner(UL, buff=1.5)
        self.play(Write(svd_formula))
        self.wait(1)
        
        # 步骤1: V^T 旋转
        step1_text = Text("步骤 1: V^T - 旋转坐标系", color=self.primary_color).scale(0.6)
        step1_text.to_edge(DOWN, buff=1.0)
        self.play(Write(step1_text))
        
        # 原图像旋转并消失
        self.play(Rotate(original_shape, PI/4), run_time=2)
        self.wait(1)
        self.play(FadeOut(step1_text))
        
        # 步骤2: Σ 缩放
        step2_text = Text("步骤 2: Σ - 沿坐标轴缩放", color=self.primary_color).scale(0.6)
        step2_text.to_edge(DOWN, buff=1.0)
        self.play(Write(step2_text))
        
        # 对旋转后的形状进行缩放
        self.play(
            original_shape.animate.stretch(2, 0).stretch(0.5, 1),
            run_time=2
        )
        self.wait(1)
        self.play(FadeOut(step2_text))
        
        # 步骤3: U 旋转
        step3_text = Text("步骤 3: U - 旋转到最终位置", color=self.primary_color).scale(0.6)
        step3_text.to_edge(DOWN, buff=1.0)
        self.play(Write(step3_text))
        
        # 最终旋转到目标位置
        self.play(Rotate(original_shape, -PI/6), run_time=2)
        self.wait(1)
        self.play(FadeOut(step3_text))

class EigenvalueScene(BaseScene):
    """特征值可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "matrix"
    
    def construct(self):
        title = self.add_title("特征值与系统稳定性", "特征值谱决定梯度流动")
        
        grid = self.add_grid()
        
        # 演示不同特征值的影响
        self.demonstrate_eigenvalue_effects()
        
        self.wait(2)
    
    def demonstrate_eigenvalue_effects(self):
        """演示特征值对系统的影响"""
        scenarios = [
            {
                "title": "特征值 > 1: 梯度爆炸",
                "matrix": [[1.5, 0], [0, 1.5]],
                "color": RED,
                "iterations": 5
            },
            {
                "title": "特征值 < 1: 梯度消失", 
                "matrix": [[0.5, 0], [0, 0.5]],
                "color": BLUE,
                "iterations": 5
            },
            {
                "title": "特征值 = 1: 稳定",
                "matrix": [[1.0, 0], [0, 1.0]],
                "color": GREEN,
                "iterations": 5
            }
        ]
        
        for scenario in scenarios:
            # 显示场景标题
            title_text = Text(scenario["title"], color=scenario["color"]).scale(0.7)
            title_text.to_edge(UP, buff=2.0)
            self.play(Write(title_text))
            
            # 创建初始向量
            vector = Arrow(ORIGIN, RIGHT + UP, color=scenario["color"], stroke_width=3)
            self.add(vector)
            
            # 重复应用变换
            vectors = [vector]
            for i in range(scenario["iterations"]):
                A = np.array(scenario["matrix"])
                current_end = vectors[-1].get_end()
                new_end = A @ np.array([current_end[0], current_end[1]])
                
                new_vector = Arrow(ORIGIN, [new_end[0], new_end[1], 0], color=scenario["color"], 
                                 stroke_width=3, fill_opacity=0.3 + i * 0.1)
                vectors.append(new_vector)
                
                self.play(Transform(vectors[-1], new_vector), run_time=0.5)
            
            self.wait(1)
            self.play(FadeOut(VGroup(*vectors)), FadeOut(title_text))