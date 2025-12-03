"""
交互式优化器对比可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from .base import compute_gradient, get_loss_function
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render, safe_compute
from common.quiz_system import QuizSystem, QuizTemplates
from common.performance import cache_data, PerformanceMonitor


class InteractiveOptimizer:
    """交互式优化器对比"""
    
    @staticmethod
    @safe_render
    def render():
        st.subheader("🚀 交互式优化器对比")
        st.markdown("""
        **常用优化器对比**:
        
        1. **SGD**: $\\theta_{t+1} = \\theta_t - \\eta g_t$
        
        2. **Momentum**: 
        $$v_{t+1} = \\beta v_t + g_t$$
        $$\\theta_{t+1} = \\theta_t - \\eta v_{t+1}$$
        
        3. **Adam** (自适应学习率):
        $$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$$
        $$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$$
        $$\\theta_{t+1} = \\theta_t - \\eta \\frac{m_t}{\\sqrt{v_t} + \\epsilon}$$
        
        其中 $g_t = \\nabla_\\theta L(\\theta_t)$ 为梯度
        """)
        
        with st.sidebar:
            st.markdown("### 🎯 损失函数")
            loss_fn_name = st.selectbox("选择函数", 
                ["rosenbrock", "rastrigin", "ackley", "beale"])
            
            st.markdown("### 🛠️ 优化器设置")
            optimizers = st.multiselect(
                "选择优化器（可多选）",
                ["SGD", "Momentum", "NAG", "AdaGrad", "RMSprop", "Adam"],
                default=["SGD", "Momentum", "Adam"]
            )
            
            st.markdown("### 📊 参数设置")
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            iterations = st.slider("迭代次数", 50, 500, 200, 50)
            
            col1, col2 = st.columns(2)
            with col1:
                start_x = st.number_input("起始 X", -3.0, 3.0, -2.0, 0.1)
            with col2:
                start_y = st.number_input("起始 Y", -3.0, 3.0, 2.0, 0.1)
        
        if not optimizers:
            st.warning("请至少选择一个优化器")
            return
        
        # 获取损失函数
        loss_fn, x_range, y_range, title = get_loss_function(loss_fn_name)
        
        # 运行所有优化器
        results = {}
        for opt_name in optimizers:
            path, loss_hist = InteractiveOptimizer._run_optimizer(
                opt_name, loss_fn, start_x, start_y, learning_rate, iterations
            )
            results[opt_name] = (path, loss_hist)
        
        # 可视化对比
        st.markdown("### 📈 优化路径对比")
        
        # 3D可视化
        fig_3d = InteractiveOptimizer._create_3d_comparison(
            loss_fn, results, x_range, y_range, title
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 收敛曲线对比
        st.markdown("### 📉 收敛速度对比")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_conv = InteractiveOptimizer._create_convergence_plot(results)
            st.plotly_chart(fig_conv, use_container_width=True)
        
        with col2:
            st.markdown("#### 最终结果")
            for opt_name, (path, loss_hist) in results.items():
                final_loss = loss_hist[-1]
                initial_loss = loss_hist[0]
                improvement = (1 - final_loss/initial_loss) * 100
                st.metric(
                    opt_name,
                    f"{final_loss:.4f}",
                    f"{improvement:.1f}% ↓",
                    delta_color="inverse"
                )
    

        # 添加交互式测验
        quiz_system = QuizSystem("optimizer")
        quizzes = QuizTemplates.get_optimizer_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _run_optimizer(opt_name, loss_fn, start_x, start_y, lr, iterations):
        """运行指定的优化器"""
        path = [np.array([start_x, start_y])]
        loss_hist = [loss_fn(start_x, start_y)]
        current = np.array([start_x, start_y])
        
        # 初始化优化器特定的状态
        velocity = np.zeros(2)  # for Momentum, NAG
        cache = np.zeros(2)  # for AdaGrad, RMSprop
        m = np.zeros(2)  # for Adam
        v = np.zeros(2)  # for Adam
        
        beta1, beta2 = 0.9, 0.999  # Adam参数
        epsilon = 1e-8
        
        for t in range(1, iterations + 1):
            grad = compute_gradient(loss_fn, current[0], current[1])
            
            if opt_name == "SGD":
                current = current - lr * grad
            
            elif opt_name == "Momentum":
                velocity = 0.9 * velocity - lr * grad
                current = current + velocity
            
            elif opt_name == "NAG":
                # Nesterov Accelerated Gradient
                look_ahead = current + 0.9 * velocity
                grad_ahead = compute_gradient(loss_fn, look_ahead[0], look_ahead[1])
                velocity = 0.9 * velocity - lr * grad_ahead
                current = current + velocity
            
            elif opt_name == "AdaGrad":
                cache += grad ** 2
                current = current - lr * grad / (np.sqrt(cache) + epsilon)
            
            elif opt_name == "RMSprop":
                cache = 0.9 * cache + 0.1 * grad ** 2
                current = current - lr * grad / (np.sqrt(cache) + epsilon)
            
            elif opt_name == "Adam":
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                current = current - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            path.append(current.copy())
            loss_hist.append(loss_fn(current[0], current[1]))
        
        return np.array(path), np.array(loss_hist)
    
    @staticmethod
    def _create_3d_comparison(loss_fn, results, x_range, y_range, title):
        """创建3D对比图"""
        x = np.linspace(x_range[0], x_range[1], 80)
        y = np.linspace(y_range[0], y_range[1], 80)
        X, Y = np.meshgrid(x, y)
        Z = loss_fn(X, Y)
        Z = np.minimum(Z, np.percentile(Z, 95))
        
        fig = go.Figure()
        
        # 曲面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.7,
            showscale=False,
            name='损失函数'
        ))
        
        # 各优化器路径
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        for i, (opt_name, (path, _)) in enumerate(results.items()):
            path_z = [loss_fn(p[0], p[1]) for p in path]
            fig.add_trace(go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path_z,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=4),
                marker=dict(size=2),
                name=opt_name
            ))
        
        fig.update_layout(
            title=f"{title} - 优化器对比",
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
    def _create_convergence_plot(results):
        """创建收敛曲线对比图"""
        fig = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        for i, (opt_name, (_, loss_hist)) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(loss_hist))),
                y=loss_hist,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=opt_name
            ))
        
        fig.update_layout(
            title="损失函数收敛曲线",
            xaxis_title="迭代次数",
            yaxis_title="损失值",
            yaxis_type="log",
            height=400
        )
        
        # 添加交互式测验
        
        return fig
