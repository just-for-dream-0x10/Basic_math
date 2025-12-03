import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *
from common.base_scene import BaseScene
from utils import create_matrix_data, create_vector_data, smooth_transition

class ConvolutionOperationScene(BaseScene):
    """卷积操作可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "convolution"
    
    def construct(self):
        title = self.add_title("卷积操作", "特征提取的核心机制")
        
        # 创建输入图像 - 调整位置避免遮挡标题
        input_image = self.create_input_image()
        self.add(input_image)
        self.wait(1)
        
        # 创建卷积核
        kernel = self.create_kernel()
        self.play(Write(kernel))
        self.wait(1)
        
        # 演示卷积过程
        self.demonstrate_convolution(input_image, kernel)
        
        # 卷积演示完成后，卷积核消失
        self.play(FadeOut(kernel), run_time=0.8)
        self.wait(0.5)
        
        # 显示特征图
        self.show_feature_map()
        
        self.wait(2)
    
    def create_input_image(self):
        """创建输入图像（5x5网格）"""
        # 创建5x5的像素网格
        pixels = VGroup()
        
        # 示例图像数据（简单的边缘模式）
        image_data = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1], 
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        for i in range(5):
            row = VGroup()
            for j in range(5):
                value = image_data[i][j]
                color = WHITE if value == 1 else BLACK
                pixel = Square(side_length=0.8, fill_color=color, fill_opacity=1.0, 
                             stroke_color=GRAY, stroke_width=1)
                pixel.move_to([j * 0.9 - 1.8, -i * 0.9 + 1.8, 0])
                row.add(pixel)
            pixels.add(row)
        
        pixels.move_to(LEFT * 3 + DOWN * 0.5)
        
        # 添加标签
        label = Text("输入图像", color=self.text_color).scale(0.6)
        label.next_to(pixels, DOWN, buff=0.5)
        
        return VGroup(pixels, label)
    
    def create_kernel(self):
        """创建卷积核（3x3）"""
        # 垂直边缘检测核
        kernel_data = [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]
        
        kernel = VGroup()
        
        for i in range(3):
            row = VGroup()
            for j in range(3):
                value = kernel_data[i][j]
                color = RED if value == -1 else (GREEN if value == 1 else GRAY)
                intensity = abs(value) / 1.0
                
                kernel_cell = Square(side_length=0.6, fill_color=color, 
                                   fill_opacity=intensity * 0.8,
                                   stroke_color=WHITE, stroke_width=1)
                kernel_cell.move_to([j * 0.7 - 0.7, -i * 0.7 + 0.7, 0])
                
                # 添加数值标签
                value_text = Text(str(value), color=WHITE, font_size=20).scale(0.4)
                value_text.move_to(kernel_cell.get_center())
                
                row.add(VGroup(kernel_cell, value_text))
            kernel.add(row)
        
        kernel.move_to(RIGHT * 2 + DOWN * 0.5)
        
        # 添加标签
        label = Text("卷积核", color=self.text_color).scale(0.6)
        label.next_to(kernel, DOWN, buff=0.5)
        
        return VGroup(kernel, label)
    
    def demonstrate_convolution(self, input_image, kernel):
        """演示卷积过程"""
        # 获取输入图像和核的位置
        input_pixels = input_image[0]
        kernel_pixels = kernel[0]
        
        # 创建高亮框表示当前卷积位置
        highlight_rect = Rectangle(width=2.7, height=2.7, color=self.primary_color, 
                                 stroke_width=3, fill_opacity=0.1)
        
        # 添加说明文本 - 调整位置避免遮挡
        process_text = Text("卷积过程：核在图像上滑动", color=self.text_color).scale(0.5)
        process_text.to_edge(DOWN, buff=0.8)
        self.play(Write(process_text))
        
        # 演示几个位置的卷积操作
        positions = [
            (-1.8, 1.8),  # 左上角
            (0, 1.8),     # 上中
            (0, 0),       # 中心
        ]
        
        for pos in positions:
            # 移动高亮框
            highlight_rect.move_to([pos[0] - 3, pos[1], 0])
            self.play(Create(highlight_rect), run_time=0.5)
            
            # 计算卷积结果
            result = self.calculate_convolution_at_position(input_pixels, kernel_pixels, pos)
            
            # 显示计算结果
            result_text = Text(f"结果: {result}", color=self.primary_color).scale(0.5)
            result_text.move_to(RIGHT * 5 + UP * 2)
            self.play(Write(result_text))
            self.wait(0.5)
            self.play(FadeOut(result_text))
        
        self.play(FadeOut(highlight_rect), FadeOut(process_text))
    
    def calculate_convolution_at_position(self, input_pixels, kernel_pixels, center_pos):
        """计算指定位置的卷积结果"""
        # 简化的卷积计算（演示用）
        # 这里返回一个模拟值
        if center_pos[0] == 0 and center_pos[1] == 0:
            return 3  # 中心位置，检测到边缘
        elif center_pos[0] == -1.8:
            return -1  # 左边位置
        else:
            return 1  # 其他位置
    
    def show_feature_map(self):
        """显示输出特征图"""
        # 创建3x3的特征图
        feature_map = VGroup()
        
        # 模拟的特征图数据
        feature_data = [
            [-1, 0, 3],
            [-1, 0, 3],
            [0, 0, 0]
        ]
        
        for i in range(3):
            row = VGroup()
            for j in range(3):
                value = feature_data[i][j]
                # 根据值设置颜色强度
                if value > 0:
                    color = GREEN
                    opacity = min(value / 3.0, 1.0)
                elif value < 0:
                    color = RED
                    opacity = min(abs(value) / 3.0, 1.0)
                else:
                    color = GRAY
                    opacity = 0.3
                
                feature_cell = Square(side_length=0.8, fill_color=color,
                                    fill_opacity=opacity, stroke_color=WHITE, stroke_width=1)
                feature_cell.move_to([j * 0.9 - 0.9, -i * 0.9 + 0.9, 0])
                
                # 添加数值
                value_text = Text(str(value), color=WHITE, font_size=20).scale(0.4)
                value_text.move_to(feature_cell.get_center())
                
                row.add(VGroup(feature_cell, value_text))
            feature_map.add(row)
        
        feature_map.move_to(RIGHT * 3.5 + DOWN * 0.5)
        
        # 添加标签
        label = Text("特征图", color=self.text_color).scale(0.6)
        label.next_to(feature_map, DOWN, buff=0.5)
        
        # 添加箭头表示变换 - 调整箭头位置和长度
        arrow = Arrow(RIGHT * 0.5, RIGHT * 2.5, color=self.primary_color, stroke_width=2)
        arrow_label = Text("卷积操作", color=self.primary_color).scale(0.4)
        arrow_label.next_to(arrow, UP, buff=0.2)
        
        self.play(Create(arrow), Write(arrow_label))
        self.play(FadeIn(VGroup(feature_map, label)), run_time=1)
        
        # 添加解释 - 调整位置
        explanation = Text("特征图显示了图像中的垂直边缘", color=self.text_color).scale(0.5)
        explanation.to_edge(DOWN, buff=0.8)
        self.play(Write(explanation))
        self.wait(1)
        self.play(FadeOut(explanation))

class KernelTypesScene(BaseScene):
    """不同类型卷积核的可视化"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "convolution"
    
    def construct(self):
        title = self.add_title("卷积核类型", "不同核的功能与效果")
        
        # 创建不同类型的卷积核
        kernels = self.create_different_kernels()
        
        # 逐一展示每种核
        for kernel_name, kernel_group in kernels.items():
            self.display_kernel(kernel_name, kernel_group)
            self.wait(1)
        
        self.wait(2)
    
    def create_different_kernels(self):
        """创建不同类型的卷积核"""
        kernels = {}
        
        # 1. 边缘检测核
        edge_kernel_data = [
            [-1, 0, 1],
            [-1, 0, 1], 
            [-1, 0, 1]
        ]
        kernels["边缘检测"] = self.create_kernel_visualization(edge_kernel_data, RED)
        
        # 2. 模糊核
        blur_kernel_data = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]
        kernels["模糊"] = self.create_kernel_visualization(blur_kernel_data, BLUE, normalize=16)
        
        # 3. 锐化核
        sharpen_kernel_data = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
        kernels["锐化"] = self.create_kernel_visualization(sharpen_kernel_data, GREEN)
        
        return kernels
    
    def create_kernel_visualization(self, kernel_data, color, normalize=1):
        """创建卷积核可视化"""
        kernel = VGroup()
        
        for i in range(3):
            row = VGroup()
            for j in range(3):
                value = kernel_data[i][j] / normalize
                intensity = abs(value)
                
                kernel_cell = Square(side_length=0.7, fill_color=color,
                                   fill_opacity=intensity * 0.8,
                                   stroke_color=WHITE, stroke_width=1)
                kernel_cell.move_to([j * 0.8 - 0.8, -i * 0.8 + 0.8, 0])
                
                # 添加数值
                if normalize > 1:
                    value_text = Text(f"{kernel_data[i][j]}/{normalize}", color=WHITE, font_size=16).scale(0.3)
                else:
                    value_text = Text(str(kernel_data[i][j]), color=WHITE, font_size=16).scale(0.3)
                value_text.move_to(kernel_cell.get_center())
                
                row.add(VGroup(kernel_cell, value_text))
            kernel.add(row)
        
        return kernel
    
    def display_kernel(self, kernel_name, kernel_group):
        """展示单个卷积核"""
        # 清除之前的内容
        self.clear()
        
        # 重新添加标题
        title = self.add_title("卷积核类型", kernel_name)
        
        # 显示核
        kernel_group.move_to(ORIGIN)
        self.play(FadeIn(kernel_group), run_time=1)
        
        # 添加功能说明
        descriptions = {
            "边缘检测": "检测图像中的边缘和轮廓",
            "模糊": "平滑图像，去除噪声",
            "锐化": "增强图像细节和边缘"
        }
        
        description = Text(descriptions[kernel_name], color=self.text_color).scale(0.6)
        description.next_to(kernel_group, DOWN, buff=1.0)
        self.play(Write(description))
        
        self.wait(1)

class FeatureExtractionScene(BaseScene):
    """特征提取过程可视化"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "convolution"
    
    def construct(self):
        title = self.add_title("特征提取层次", "从低级到高级的特征")
        
        # 创建横向层次结构
        layers = self.create_feature_layers()
        
        # 展示特征提取过程
        self.demonstrate_hierarchy(layers)
        
        self.wait(2)
    
    def create_feature_layers(self):
        """创建特征层次 - 恢复完整信息并优化布局"""
        layers = VGroup()
        
        # 恢复完整的特征层次信息
        layer_configs = [
            ("Layer 1", 
             ["垂直边缘", "水平边缘", "对角线", "角点", "纹理基元", "颜色梯度"], 
             RED),
            ("Layer 2", 
             ["简单形状", "几何图形", "纹理模式", "边缘组合", "局部结构", "基础轮廓"], 
             ORANGE),
            ("Layer 3", 
             ["物体部件", "面部特征", "肢体部位", "组件结构", "功能单元", "语义部件"], 
             GREEN),
            ("Layer 4", 
             ["完整对象", "场景理解", "语义概念", "抽象关系", "高级语义", "整体识别"], 
             BLUE)
        ]
        
        for i, (layer_name, features, color) in enumerate(layer_configs):
            layer = self.create_horizontal_layer(layer_name, features, color)
            layers.add(layer)
        
        # 横向排列，适当调整间距
        layers.arrange(RIGHT, buff=0.8)
        
        # 整体缩放以适应屏幕
        layers.scale(0.65)
        layers.move_to(ORIGIN)
        
        return layers
    
    def create_horizontal_layer(self, layer_name, features, color):
        """创建横向排列的特征层 - 修复文本重叠问题"""
        layer = VGroup()
        
        # 背景框 - 增加高度避免重叠
        bg = Rectangle(width=3.5, height=4.5, fill_color=color, fill_opacity=0.1, 
                     stroke_color=color, stroke_width=2)
        layer.add(bg)
        
        # 层名称 - 放在顶部
        name_text = Text(layer_name, color=color, font_size=28).scale(0.6)
        name_text.move_to(bg.get_top() + DOWN * 0.8)
        layer.add(name_text)
        
        # 特征示例 - 垂直排列，调整位置避免与名称和类型说明重叠
        features_group = VGroup()
        for feature in features:  # 恢复所有特征，不限制数量
            feature_text = Text(f"• {feature}", color=self.text_color).scale(0.35)  # 稍微缩小字体
            features_group.add(feature_text)
        
        features_group.arrange(DOWN, buff=0.15, aligned_edge=LEFT)  # 减少间距
        features_group.move_to(bg.get_center())  # 居中放置
        layer.add(features_group)
        
        # 添加特征类型说明 - 确保不与特征列表重叠
        feature_types = {
            "Layer 1": "低级特征：边缘、角点、纹理",
            "Layer 2": "中级特征：形状、轮廓、模式", 
            "Layer 3": "高级特征：部件、对象部分",
            "Layer 4": "语义特征：完整对象、概念"
        }
        
        if layer_name in feature_types:
            type_text = Text(feature_types[layer_name], color=color, font_size=20).scale(0.3)  # 缩小字体
            type_text.move_to(bg.get_bottom() + UP * 0.4)  # 调整位置
            layer.add(type_text)
        
        return layer
    
    def demonstrate_hierarchy(self, layers):
        """演示特征层次 - 恢复完整演示过程"""
        # 逐个显示层，强调层次递进
        for i, layer in enumerate(layers):
            self.play(FadeIn(layer), run_time=0.8)
            self.wait(0.3)
        
        self.wait(1)
        
        # 创建连接线表示信息流动
        connections = VGroup()
        for i in range(len(layers) - 1):
            arrow = Arrow(
                start=layers[i].get_right(),
                end=layers[i+1].get_left(),
                color=self.primary_color,
                stroke_width=2,
                buff=0.1
            )
            connections.add(arrow)
        
        # 添加信息流标签
        flow_labels = VGroup()
        for i in range(len(layers) - 1):
            label = Text("特征组合", color=self.primary_color).scale(0.3)
            label.move_to(connections[i].get_center() + UP * 0.3)
            flow_labels.add(label)
        
        self.play(Create(connections), Write(flow_labels), run_time=1.0)
        self.wait(1)
        
        # 逐层高亮并详细说明
        layer_descriptions = [
            "卷积层1：提取基础视觉元素",
            "卷积层2：组合形成简单模式", 
            "卷积层3：识别复杂部件结构",
            "卷积层4：理解完整语义对象"
        ]
        
        for i, (layer, desc) in enumerate(zip(layers, layer_descriptions)):
            # 高亮当前层
            highlight = SurroundingRectangle(
                layer, 
                color=self.primary_color,
                fill_opacity=0.1,
                stroke_width=3
            )
            
            # 显示层说明
            desc_text = Text(desc, color=self.text_color).scale(0.4)
            desc_text.to_edge(DOWN, buff=1.0)
            
            self.play(Create(highlight), Write(desc_text), run_time=0.5)
            self.wait(0.8)
            self.play(FadeOut(highlight), FadeOut(desc_text), run_time=0.3)
        
        # 添加完整的特征提取原理说明
        principle_text = VGroup(
            Text("CNN特征提取原理：", color=self.primary_color).scale(0.5),
            Text("1. 局部感受野：每层只关注前层的局部区域", color=self.text_color).scale(0.4),
            Text("2. 权重共享：同种特征检测器在整个图像上复用", color=self.text_color).scale(0.4),
            Text("3. 层次组合：低级特征逐层组合成高级语义", color=self.text_color).scale(0.4),
            Text("4. 平移不变性：特征检测与位置无关", color=self.text_color).scale(0.4)
        )
        principle_text.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        principle_text.to_edge(DOWN, buff=1.0)
        
        self.play(Write(principle_text), run_time=2.0)
        self.wait(2)
        self.play(FadeOut(principle_text))