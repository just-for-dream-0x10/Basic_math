"""
3D梯度下降可视化场景
功能：
1. 3D损失函数曲面展示
2. 动态梯度下降路径追踪
3. 多角度旋转视图
4. 实时显示参数和损失值
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from manim import *

class Basic3DGradient(ThreeDScene):
    """3D梯度下降场景 - 增强版"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "loss"
    
    def construct(self):
        # 设置3D相机角度
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # 显示标题
        title = Text("3D 梯度下降", font_size=48, color=BLUE).to_edge(UP)
        subtitle = Text("寻找损失函数的最小值", font_size=28, color=GRAY).next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(Write(title), Write(subtitle))
        self.wait(1)
        
        # 淡出标题
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # 创建3D坐标系
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 10, 2],
            x_length=8,
            y_length=8,
            z_length=5,
            axis_config={
                "include_numbers": False,
                "include_tip": True,
                "tip_length": 0.2,
            }
        )
        
        # 添加坐标轴标签
        x_label = MathTex("w_1", font_size=36).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("w_2", font_size=36).next_to(axes.y_axis.get_end(), UP)
        z_label = MathTex("L", font_size=36, color=YELLOW).next_to(axes.z_axis.get_end(), UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))
        self.wait(0.5)
        
        # 定义损失函数 (Rosenbrock函数的修改版)
        def loss_function(x, y):
            """经典的非凸优化测试函数"""
            a = 1
            b = 100
            return (a - x)**2 + b * (y - x**2)**2
        
        # 创建损失函数曲面
        surface = Surface(
            lambda u, v: axes.c2p(u, v, loss_function(u, v)),
            u_range=[-2.5, 2.5],
            v_range=[-1, 3],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.5,
        )
        
        # 应用渐变色
        surface.set_fill_by_value(
            axes=axes,
            colorscale=[(RED, -0.5), (YELLOW, 5), (GREEN, 10)],
            axis=2
        )
        
        self.play(Create(surface), run_time=2)
        self.wait(1)
        
        # 开始相机旋转
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(2)
        
        # 梯度下降参数
        learning_rate = 0.003
        iterations = 100
        start_point = np.array([-2.0, 2.5])
        
        # 存储路径点
        path_points = [start_point.copy()]
        current_point = start_point.copy()
        
        # 计算梯度下降路径
        for i in range(iterations):
            # 数值梯度
            h = 1e-5
            grad_x = (loss_function(current_point[0] + h, current_point[1]) - 
                     loss_function(current_point[0] - h, current_point[1])) / (2 * h)
            grad_y = (loss_function(current_point[0], current_point[1] + h) - 
                     loss_function(current_point[0], current_point[1] - h)) / (2 * h)
            
            # 更新参数
            current_point[0] -= learning_rate * grad_x
            current_point[1] -= learning_rate * grad_y
            
            path_points.append(current_point.copy())
        
        # 创建起始点
        start_z = loss_function(path_points[0][0], path_points[0][1])
        start_dot = Sphere(radius=0.15, color=RED).move_to(
            axes.c2p(path_points[0][0], path_points[0][1], start_z)
        )
        
        start_label = Text("起始点", font_size=24, color=RED)
        self.add_fixed_in_frame_mobjects(start_label)
        start_label.to_corner(UL).shift(DOWN * 0.5)
        
        self.play(GrowFromCenter(start_dot), Write(start_label))
        self.wait(1)
        
        # 创建路径追踪
        path_trace = VGroup()
        moving_dot = Sphere(radius=0.12, color=YELLOW).move_to(start_dot.get_center())
        
        # 创建信息面板
        info_panel = VGroup()
        iteration_text = Text("迭代: 0", font_size=28, color=WHITE)
        loss_text = Text(f"损失: {start_z:.2f}", font_size=28, color=YELLOW)
        param_text = Text(f"w₁={path_points[0][0]:.2f}, w₂={path_points[0][1]:.2f}", 
                         font_size=24, color=GREEN)
        
        info_panel.add(iteration_text, loss_text, param_text)
        info_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        self.add_fixed_in_frame_mobjects(info_panel)
        info_panel.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)
        
        self.play(Write(info_panel))
        self.add(moving_dot)
        
        # 停止相机旋转，设置最佳观察角度
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=1.5)
        
        # 动画展示梯度下降过程
        self.play(FadeOut(start_label))
        
        # 分段显示路径，每10步更新一次信息
        step_size = 5
        for i in range(0, len(path_points) - 1, step_size):
            segment_points = path_points[i:min(i + step_size + 1, len(path_points))]
            
            # 创建路径段
            for j in range(len(segment_points) - 1):
                p1 = segment_points[j]
                p2 = segment_points[j + 1]
                z1 = loss_function(p1[0], p1[1])
                z2 = loss_function(p2[0], p2[1])
                
                line = Line3D(
                    start=axes.c2p(p1[0], p1[1], z1),
                    end=axes.c2p(p2[0], p2[1], z2),
                    color=YELLOW,
                    stroke_width=4
                )
                path_trace.add(line)
            
            # 移动点
            end_point = segment_points[-1]
            end_z = loss_function(end_point[0], end_point[1])
            
            # 更新信息
            new_iteration = Text(f"迭代: {min(i + step_size, len(path_points) - 1)}", 
                               font_size=28, color=WHITE)
            new_loss = Text(f"损失: {end_z:.2f}", font_size=28, color=YELLOW)
            new_param = Text(f"w₁={end_point[0]:.2f}, w₂={end_point[1]:.2f}", 
                           font_size=24, color=GREEN)
            
            new_info = VGroup(new_iteration, new_loss, new_param)
            new_info.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            self.add_fixed_in_frame_mobjects(new_info)
            new_info.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)
            
            # 动画
            animations = [
                Create(path_trace[len(path_trace) - len(segment_points) + 1:]),
                moving_dot.animate.move_to(axes.c2p(end_point[0], end_point[1], end_z)),
                Transform(info_panel, new_info)
            ]
            
            self.play(*animations, run_time=0.3)
        
        # 显示最终结果
        final_point = path_points[-1]
        final_z = loss_function(final_point[0], final_point[1])
        
        # 最小值标记
        min_dot = Sphere(radius=0.15, color=GREEN, resolution=(16, 16)).move_to(
            axes.c2p(final_point[0], final_point[1], final_z)
        )
        
        self.play(
            FadeOut(moving_dot),
            GrowFromCenter(min_dot)
        )
        
        # 最终信息
        final_label = Text("找到最小值!", font_size=36, color=GREEN, weight=BOLD)
        self.add_fixed_in_frame_mobjects(final_label)
        final_label.to_edge(DOWN).shift(UP * 0.5)
        
        self.play(Write(final_label))
        self.wait(1)
        
        # 最后旋转展示
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        
        # 淡出
        self.play(
            FadeOut(VGroup(axes, surface, path_trace, start_dot, min_dot, 
                          x_label, y_label, z_label, info_panel, final_label))
        )
        self.wait(0.5)
