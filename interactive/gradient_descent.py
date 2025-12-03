"""
交互式梯度下降可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from .base import compute_gradient, get_loss_function, LOSS_FUNCTION_NAMES


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveGradientDescent:
    """交互式梯度下降可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎮 交互式梯度下降")
        st.markdown("""
        **梯度下降算法**: 迭代优化方法，沿着梯度的负方向更新参数
        
        $$\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L(\\theta_t)$$
        
        其中:
        - $\\theta$: 模型参数
        - $\\eta$: 学习率 (步长)
        - $\\nabla_\\theta L$: 损失函数的梯度
        
        **收敛条件**: 
        - 凸函数: $\\eta < \\frac{2}{L}$ (L为Lipschitz常数)
        - 非凸函数: 收敛到局部最小值或鞍点
        """)
        
        # 侧边栏参数控制
        with st.sidebar:
            st.markdown("### 📊 参数设置")
            
            loss_function = st.selectbox(
                "损失函数",
                list(LOSS_FUNCTION_NAMES.keys()),
                format_func=lambda x: LOSS_FUNCTION_NAMES[x]
            )
            
            learning_rate = st.slider("学习率 (Learning Rate)", 0.001, 0.1, 0.003, 0.001, 
                                     help="控制每次参数更新的步长")
            
            iterations = st.slider("迭代次数", 10, 300, 100, 10)
            
            col1, col2 = st.columns(2)
            with col1:
                start_x = st.number_input("起始点 x", -3.0, 3.0, -2.0, 0.1)
            with col2:
                start_y = st.number_input("起始点 y", -3.0, 3.0, 2.0, 0.1)
            
            show_contour = st.checkbox("显示等高线图", value=True)
            show_3d = st.checkbox("显示3D曲面", value=True)
        
        # 设置损失函数
        loss_fn, x_range, y_range, title = get_loss_function(loss_function)
        
        # 执行梯度下降
        path, loss_history = InteractiveGradientDescent._gradient_descent(
            loss_fn, start_x, start_y, learning_rate, iterations
        )
        
        # 布局
        if show_3d:
            st.markdown("### 📈 3D损失曲面与梯度下降路径")
            fig_3d = InteractiveGradientDescent._create_3d_surface(
                loss_fn, path, loss_history, x_range, y_range, title
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        if show_contour:
            st.markdown("### 📉 等高线图与收敛曲线")
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig_contour = InteractiveGradientDescent._create_contour(
                    loss_fn, path, x_range, y_range, title
                )
                st.pyplot(fig_contour)
            
            with col2:
                fig_loss = InteractiveGradientDescent._create_loss_curve(loss_history)
                st.pyplot(fig_loss)
        
        # 显示统计信息
        st.markdown("### 📊 统计信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("起始损失", f"{loss_history[0]:.4f}")
        with col2:
            st.metric("最终损失", f"{loss_history[-1]:.4f}")
        with col3:
            st.metric("损失降低", f"{(1 - loss_history[-1]/loss_history[0])*100:.2f}%")
        with col4:
            st.metric("最终位置", f"({path[-1, 0]:.2f}, {path[-1, 1]:.2f})")
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("gradient_descent")
        quizzes = QuizTemplates.get_gradient_descent_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _gradient_descent(loss_fn, start_x, start_y, lr, iterations):
        """执行梯度下降"""
        path = [np.array([start_x, start_y])]
        loss_history = [loss_fn(start_x, start_y)]
        current = np.array([start_x, start_y])
        
        for _ in range(iterations):
            grad = compute_gradient(loss_fn, current[0], current[1])
            current = current - lr * grad
            path.append(current.copy())
            loss_history.append(loss_fn(current[0], current[1]))
        
        return np.array(path), np.array(loss_history)
    
    @staticmethod
    def _create_3d_surface(loss_fn, path, loss_history, x_range, y_range, title):
        """创建3D曲面图"""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = loss_fn(X, Y)
        Z = np.minimum(Z, np.percentile(Z, 95))
        
        fig = go.Figure()
        
        # 曲面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.8,
            name='损失函数'
        ))
        
        # 路径
        path_z = [loss_fn(p[0], p[1]) for p in path]
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path_z,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=3, color='red'),
            name='梯度下降路径'
        ))
        
        # 起点和终点
        fig.add_trace(go.Scatter3d(
            x=[path[0, 0]], y=[path[0, 1]], z=[path_z[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='diamond'),
            name='起始点'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[path[-1, 0]], y=[path[-1, 1]], z=[path_z[-1]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='diamond'),
            name='终点'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='w₁',
                yaxis_title='w₂',
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=600
        )
        
        return fig
    
    @staticmethod
    def _create_contour(loss_fn, path, x_range, y_range, title):
        """创建等高线图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.linspace(x_range[0], x_range[1], 200)
        y = np.linspace(y_range[0], y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = loss_fn(X, Y)
        
        levels = np.logspace(np.log10(Z.min() + 1e-8), np.log10(np.percentile(Z, 95)), 20)
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='路径')
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='起点')
        ax.plot(path[-1, 0], path[-1, 1], 'b*', markersize=15, label='终点')
        
        ax.set_xlabel('w₁', fontsize=12)
        ax.set_ylabel('w₂', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def _create_loss_curve(loss_history):
        """创建损失曲线"""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.plot(loss_history, linewidth=2, color='orange')
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('损失值', fontsize=12)
        ax.set_title('收敛曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        return fig
