"""
交互式扩散模型可视化
严格按照 15.DiffusionModel.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import io
import base64


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveDiffusionModel:
    """交互式扩散模型可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🌊 扩散模型与随机微分方程 (SDE)")
        st.markdown("""
        **核心思想**: 从热力学熵增到生成智能，学习让时间倒流的力场
        
        关键概念：
        - **前向SDE**: 数据逐渐变成噪声（熵增过程）
        - **逆向SDE**: 噪声逐渐变成数据（时间倒流）
        - **得分函数**: $\nabla_{x_t} \log p_t(x_t)$，指向数据密度增加最快的方向
        - **朗之万动力学**: 在能量地形图上的采样过程
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["扩散过程演示", "得分函数可视化", "SDE求解器对比", "朗之万动力学"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "扩散过程演示":
            InteractiveDiffusionModel._render_diffusion_process()
        elif viz_type == "得分函数可视化":
            InteractiveDiffusionModel._render_score_function()
        elif viz_type == "SDE求解器对比":
            InteractiveDiffusionModel._render_sde_solvers()
        elif viz_type == "朗之万动力学":
            InteractiveDiffusionModel._render_langevin_dynamics()
    

        # 添加交互式测验
        quiz_system = QuizSystem("diffusion_model")
        quizzes = QuizTemplates.get_diffusion_model_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_diffusion_process():
        """扩散过程演示"""
        st.markdown("### 🌊 扩散过程：从有序到无序")
        
        st.latex(r"""
        dX_t = f(x_t, t)dt + g(t)dW_t
        """)
        
        with st.sidebar:
            num_steps = st.slider("扩散步数", 10, 100, 50, 5)
            noise_strength = st.slider("噪声强度", 0.1, 2.0, 0.5, 0.1)
            show_forward = st.checkbox("显示前向过程", value=True)
            show_reverse = st.checkbox("显示逆向过程", value=True)
        
        # 创建简单的2D数据分布（两个高斯簇）
        np.random.seed(42)
        
        # 生成初始数据
        cluster1 = np.random.multivariate_normal([2, 2], [[0.3, 0], [0, 0.3]], 100)
        cluster2 = np.random.multivariate_normal([-2, -2], [[0.3, 0], [0, 0.3]], 100)
        data = np.vstack([cluster1, cluster2])
        
        # 前向扩散过程
        forward_steps = []
        current_data = data.copy()
        
        for step in range(num_steps):
            t = step / num_steps
            beta_t = noise_strength * t
            alpha_t = 1 - beta_t
            
            # 添加噪声
            noise = np.random.randn(*current_data.shape) * np.sqrt(beta_t)
            current_data = np.sqrt(alpha_t) * current_data + noise
            forward_steps.append(current_data.copy())
        
        # 逆向扩散过程（简化模拟）
        reverse_steps = []
        current_data = forward_steps[-1].copy()
        
        for step in range(num_steps):
            t = 1 - (step / num_steps)
            beta_t = noise_strength * t
            alpha_t = 1 - beta_t
            
            # 简化的去噪（模拟得分函数）
            noise = np.random.randn(*current_data.shape) * np.sqrt(beta_t) * 0.1
            current_data = (current_data - noise * 0.5) / np.sqrt(alpha_t)
            reverse_steps.append(current_data.copy())
        
        # 创建可视化
        fig = make_subplots(
            rows=2, cols=5,
            subplot_titles=[f"t={i/10:.1f}" for i in range(0, 10, 2)],
            specs=[[{"type": "scatter"}]*5]*2
        )
        
        # 显示前向过程
        if show_forward:
            for i in range(5):
                step_idx = i * (num_steps // 5)
                if step_idx < len(forward_steps):
                    data_step = forward_steps[step_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=data_step[:100, 0], y=data_step[:100, 1],
                            mode='markers', marker=dict(color='blue', size=4),
                            name='簇1', showlegend=False
                        ),
                        row=1, col=i+1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data_step[100:, 0], y=data_step[100:, 1],
                            mode='markers', marker=dict(color='red', size=4),
                            name='簇2', showlegend=False
                        ),
                        row=1, col=i+1
                    )
        
        # 显示逆向过程
        if show_reverse:
            for i in range(5):
                step_idx = i * (num_steps // 5)
                if step_idx < len(reverse_steps):
                    data_step = reverse_steps[step_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=data_step[:100, 0], y=data_step[:100, 1],
                            mode='markers', marker=dict(color='blue', size=4),
                            name='簇1', showlegend=False
                        ),
                        row=2, col=i+1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data_step[100:, 0], y=data_step[100:, 1],
                            mode='markers', marker=dict(color='red', size=4),
                            name='簇2', showlegend=False
                        ),
                        row=2, col=i+1
                    )
        
        fig.update_layout(
            title="扩散过程：前向（熵增）vs 逆向（时间倒流）",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="x", row=2, col=3)
        fig.update_yaxes(title_text="y", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示熵的变化
        st.markdown("### 📊 熵的变化分析")
        
        entropies = []
        for step_data in forward_steps:
            # 计算协方差矩阵的行列式作为熵的代理
            cov = np.cov(step_data.T)
            entropy = 0.5 * np.log(np.linalg.det(cov) + 1e-8)
            entropies.append(entropy)
        
        fig_entropy = go.Figure()
        fig_entropy.add_trace(go.Scatter(
            x=np.linspace(0, 1, len(entropies)),
            y=entropies,
            mode='lines+markers',
            name='熵'
        ))
        
        fig_entropy.update_layout(
            title="前向扩散过程中的熵增",
            xaxis_title="时间步 t",
            yaxis_title="熵（代理指标）",
            height=400
        )
        
        st.plotly_chart(fig_entropy, use_container_width=True)
        
        st.info("""
        **物理直觉**：
        - 前向过程：墨水滴入清水，逐渐扩散，熵增加
        - 逆向过程：AI学习"力场"，让时间倒流，熵减少
        - 得分函数：指向数据密度最高的方向（山谷）
        """)
    
    @staticmethod
    def _render_score_function():
        """得分函数可视化"""
        st.markdown("### 🎯 得分函数：概率空间的力场")
        
        st.latex(r"""
        s_\theta(x,t) \approx \nabla_{x_t} \log p_t(x_t) = -\frac{\epsilon}{\sigma_t}
        """)
        
        with st.sidebar:
            grid_size = st.slider("网格大小", 20, 50, 30, 5)
            time_step = st.slider("时间步", 0.0, 1.0, 0.5, 0.1)
            show_contour = st.checkbox("显示等高线", value=True)
            show_streamlines = st.checkbox("显示流线", value=True)
        
        # 创建2D网格
        x = np.linspace(-4, 4, grid_size)
        y = np.linspace(-4, 4, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 模拟两个高斯分布的得分函数
        def score_function(x, y, t):
            # 两个高斯中心
            center1 = np.array([2, 2])
            center2 = np.array([-2, -2])
            
            # 计算到中心的距离
            dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
            dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)
            
            # 得分函数指向最近的高斯中心
            sigma_t = 0.1 + 0.9 * t  # 时间相关的方差
            
            # 计算得分（指向中心的力）
            score1 = -(np.array([x - center1[0], y - center1[1]]) / (sigma_t**2)) * np.exp(-dist1**2 / (2 * sigma_t**2))
            score2 = -(np.array([x - center2[0], y - center2[1]]) / (sigma_t**2)) * np.exp(-dist2**2 / (2 * sigma_t**2))
            
            return score1 + score2
        
        # 计算得分函数
        score_x = np.zeros_like(X)
        score_y = np.zeros_like(Y)
        
        for i in range(grid_size):
            for j in range(grid_size):
                score = score_function(X[i, j], Y[i, j], time_step)
                score_x[i, j] = score[0]
                score_y[i, j] = score[1]
        
        # 计算得分大小
        score_magnitude = np.sqrt(score_x**2 + score_y**2)
        
        # 创建可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["得分函数向量场", "得分大小分布"],
            specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 向量场
        skip = max(1, grid_size // 15)  # 减少箭头数量
        fig.add_trace(
            go.Scatter(
                x=X[::skip, ::skip].flatten(),
                y=Y[::skip, ::skip].flatten(),
                mode='markers',
                marker=dict(size=3, color='lightblue'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 添加向量箭头（用线段表示）
        for i in range(0, grid_size, skip):
            for j in range(0, grid_size, skip):
                scale = 0.3
                fig.add_trace(
                    go.Scatter(
                        x=[X[i, j], X[i, j] + scale * score_x[i, j]],
                        y=[Y[i, j], Y[i, j] + scale * score_y[i, j]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 等高线
        if show_contour:
            # 模拟概率分布
            prob = np.exp(-((X-2)**2 + (Y-2)**2) / (2 * (0.5 + time_step)**2)) + \
                   np.exp(-((X+2)**2 + (Y+2)**2) / (2 * (0.5 + time_step)**2))
            
            fig.add_trace(
                go.Contour(
                    x=x, y=y, z=prob,
                    colorscale='Viridis',
                    showscale=False,
                    contours=dict(showlabels=True),
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # 得分大小热力图
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=score_magnitude,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="得分大小")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"得分函数可视化 (t={time_step:.1f})",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="y", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示关键洞察
        st.markdown("### 🔍 得分函数的物理意义")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **向量场方向**：
            - 红色箭头指向数据密度高的区域
            - 箭头长度表示"力"的大小
            - 这是神经网络学习的"力场"
            """)
        
        with col2:
            st.markdown("""
            **时间演化**：
            - t=0: 得分函数尖锐，指向明确
            - t=1: 得分函数平缓，方向模糊
            - 对应噪声逐渐增加的过程
            """)
        
        st.info("""
        **核心洞察**：
        - 得分函数 = 概率密度的梯度 = 物理力场
        - 生成过程 = 沿着力场"下山"
        - 神经网络学习的是这个力场
        """)
    
    @staticmethod
    def _render_sde_solvers():
        """SDE求解器对比"""
        st.markdown("### ⚙️ SDE求解器对比")
        
        st.latex(r"""
        x_{t-1} = x_t - [f(x_t, t) - g(t)^2 s_\theta(x_t, t)]\Delta t + g(t)\sqrt{|\Delta t|} z
        """)
        
        with st.sidebar:
            solver_type = st.selectbox("求解器类型", 
                ["Euler-Maruyama", "DDIM", "DPM-Solver"])
            num_steps = st.slider("采样步数", 10, 100, 20, 5)
            noise_scale = st.slider("噪声尺度", 0.5, 2.0, 1.0, 0.1)
        
        # 模拟不同的求解器
        np.random.seed(42)
        
        # 初始噪声
        x0 = np.random.randn(2) * 3
        
        # 模拟得分函数（简单实现）
        def simple_score(x, t):
            # 指向原点的得分
            return -x / (1 + t)
        
        # 不同求解器的实现
        def euler_maruyama(x0, num_steps, noise_scale):
            trajectory = [x0.copy()]
            x = x0.copy()
            
            for i in range(num_steps):
                t = i / num_steps
                dt = 1.0 / num_steps
                
                # 确定性项
                drift = -x * dt
                score = simple_score(x, t) * dt
                deterministic = drift + score
                
                # 随机项
                noise = np.random.randn(2) * noise_scale * np.sqrt(abs(dt))
                
                x = x + deterministic + noise
                trajectory.append(x.copy())
            
            return np.array(trajectory)
        
        def ddim_solver(x0, num_steps, noise_scale):
            trajectory = [x0.copy()]
            x = x0.copy()
            
            for i in range(num_steps):
                t = i / num_steps
                dt = 1.0 / num_steps
                
                # DDIM：确定性ODE求解
                drift = -x * dt
                score = simple_score(x, t) * dt * 2  # 调整系数
                x = x + drift + score
                
                trajectory.append(x.copy())
            
            return np.array(trajectory)
        
        def dpm_solver(x0, num_steps, noise_scale):
            trajectory = [x0.copy()]
            x = x0.copy()
            
            for i in range(num_steps):
                t = i / num_steps
                dt = 1.0 / num_steps
                
                # DPM-Solver：多步预测
                if i == 0:
                    drift = -x * dt
                    score = simple_score(x, t) * dt
                else:
                    # 使用历史信息改进预测
                    drift = -x * dt * 1.2
                    score = simple_score(x, t) * dt * 0.8
                
                noise = np.random.randn(2) * noise_scale * np.sqrt(abs(dt)) * 0.5
                
                x = x + drift + score + noise
                trajectory.append(x.copy())
            
            return np.array(trajectory)
        
        # 运行选定的求解器
        if solver_type == "Euler-Maruyama":
            trajectory = euler_maruyama(x0, num_steps, noise_scale)
        elif solver_type == "DDIM":
            trajectory = ddim_solver(x0, num_steps, noise_scale)
        else:
            trajectory = dpm_solver(x0, num_steps, noise_scale)
        
        # 可视化轨迹
        fig = go.Figure()
        
        # 绘制轨迹
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            name='生成轨迹'
        ))
        
        # 标记起点和终点
        fig.add_trace(go.Scatter(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name='起点（噪声）'
        ))
        
        fig.add_trace(go.Scatter(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='star'),
            name='终点（生成）'
        ))
        
        # 添加目标区域（模拟数据分布）
        theta = np.linspace(0, 2*np.pi, 100)
        target_x = 2 * np.cos(theta)
        target_y = 2 * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=target_x,
            y=target_y,
            mode='lines',
            line=dict(color='lightgray', width=2, dash='dash'),
            name='目标分布'
        ))
        
        fig.update_layout(
            title=f"{solver_type} 求解器轨迹 ({num_steps} 步)",
            xaxis_title="x",
            yaxis_title="y",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 求解器对比
        st.markdown("### 📊 求解器特性对比")
        
        # 运行所有求解器进行对比
        euler_traj = euler_maruyama(x0, num_steps, noise_scale)
        ddim_traj = ddim_solver(x0, num_steps, noise_scale)
        dpm_traj = dpm_solver(x0, num_steps, noise_scale)
        
        fig_compare = go.Figure()
        
        fig_compare.add_trace(go.Scatter(
            x=euler_traj[:, 0], y=euler_traj[:, 1],
            mode='lines', name='Euler-Maruyama', line=dict(color='blue')
        ))
        
        fig_compare.add_trace(go.Scatter(
            x=ddim_traj[:, 0], y=ddim_traj[:, 1],
            mode='lines', name='DDIM', line=dict(color='red')
        ))
        
        fig_compare.add_trace(go.Scatter(
            x=dpm_traj[:, 0], y=dpm_traj[:, 1],
            mode='lines', name='DPM-Solver', line=dict(color='green')
        ))
        
        fig_compare.update_layout(
            title="不同求解器轨迹对比",
            xaxis_title="x",
            yaxis_title="y",
            height=400
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # 显示求解器特性
        solver_info = {
            "Euler-Maruyama": {
                "类型": "随机微分方程求解器",
                "特点": "每步都加噪声，生成质量高",
                "速度": "慢（需要1000步）",
                "适用": "高质量生成"
            },
            "DDIM": {
                "类型": "确定性ODE求解器",
                "特点": "去除随机项，可加速",
                "速度": "快（20-50步）",
                "适用": "快速生成"
            },
            "DPM-Solver": {
                "类型": "高级求解器",
                "特点": "多步预测，自适应",
                "速度": "中等（20-100步）",
                "适用": "平衡质量与速度"
            }
        }
        
        df = pd.DataFrame(solver_info).T
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def _render_langevin_dynamics():
        """朗之万动力学可视化"""
        st.markdown("### ⚡ 朗之万动力学：能量地形图上的采样")
        
        st.latex(r"""
        x_{\text{new}} = x + \eta \nabla_{x} \log p(x) + \sqrt{2\eta} z
        """)
        
        with st.sidebar:
            landscape_type = st.selectbox("地形类型", 
                ["双井势", "墨西哥帽势", "随机势"])
            step_size = st.slider("步长 η", 0.01, 0.5, 0.1, 0.01)
            temperature = st.slider("温度", 0.1, 2.0, 1.0, 0.1)
            num_steps = st.slider("采样步数", 100, 1000, 500, 50)
        
        # 定义不同的势能函数
        def double_well_potential(x, y):
            """双井势"""
            return (x**2 - 1)**2 + y**2
        
        def mexican_hat_potential(x, y):
            """墨西哥帽势"""
            r2 = x**2 + y**2
            return r2**2 - 2 * r2
        
        def random_potential(x, y):
            """随机势"""
            return (x**2 + y**2) + 0.5 * np.sin(3*x) * np.cos(3*y)
        
        # 选择势能函数
        if landscape_type == "双井势":
            potential = double_well_potential
        elif landscape_type == "墨西哥帽势":
            potential = mexican_hat_potential
        else:
            potential = random_potential
        
        # 计算得分函数（负梯度）
        def score_function(x, y):
            eps = 1e-6
            dx = (potential(x + eps, y) - potential(x - eps, y)) / (2 * eps)
            dy = (potential(x, y + eps) - potential(x, y - eps)) / (2 * eps)
            return np.array([-dx, -dy])  # 负梯度
        
        # 朗之万动力学采样
        def langevin_sampling(start_pos, num_steps, step_size, temperature):
            trajectory = [start_pos.copy()]
            pos = start_pos.copy()
            
            for i in range(num_steps):
                # 计算得分（梯度）
                score = score_function(pos[0], pos[1])
                
                # 确定性项（向低能量移动）
                deterministic = step_size * score
                
                # 随机项（热运动）
                noise = np.sqrt(2 * step_size * temperature) * np.random.randn(2)
                
                # 更新位置
                pos = pos + deterministic + noise
                trajectory.append(pos.copy())
            
            return np.array(trajectory)
        
        # 创建势能地形图
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = potential(X[j, i], Y[j, i])
        
        # 多个采样轨迹
        np.random.seed(42)
        trajectories = []
        
        for i in range(5):
            # 随机初始位置
            start_pos = np.random.randn(2) * 2
            traj = langevin_sampling(start_pos, num_steps, step_size, temperature)
            trajectories.append(traj)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["势能地形图", "采样轨迹"],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # 势能地形图
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="势能")
            ),
            row=1, col=1
        )
        
        # 采样轨迹
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, traj in enumerate(trajectories):
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0], y=traj[:, 1],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=3),
                    name=f'轨迹 {i+1}',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # 标记起点和终点
            fig.add_trace(
                go.Scatter(
                    x=[traj[0, 0]], y=[traj[0, 1]],
                    mode='markers',
                    marker=dict(size=8, color=colors[i], symbol='circle'),
                    name=f'起点 {i+1}',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[traj[-1, 0]], y=[traj[-1, 1]],
                    mode='markers',
                    marker=dict(size=8, color=colors[i], symbol='star'),
                    name=f'终点 {i+1}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="朗之万动力学采样过程",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="y", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 能量变化分析
        st.markdown("### 📈 能量变化分析")
        
        fig_energy = go.Figure()
        
        for i, traj in enumerate(trajectories):
            energies = [potential(pos[0], pos[1]) for pos in traj]
            fig_energy.add_trace(
                go.Scatter(
                    x=np.arange(len(energies)),
                    y=energies,
                    mode='lines',
                    name=f'轨迹 {i+1}',
                    line=dict(color=colors[i])
                )
            )
        
        fig_energy.update_layout(
            title="采样过程中的能量变化",
            xaxis_title="步数",
            yaxis_title="势能",
            height=400
        )
        
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # 关键参数影响
        st.markdown("### 🎛️ 参数影响分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **步长影响**：
            - 大步长：快速收敛但可能不稳定
            - 小步长：稳定但收敛慢
            - 需要平衡效率与稳定性
            """)
        
        with col2:
            st.markdown("""
            **温度影响**：
            - 高温度：更多随机性，避免局部最优
            - 低温度：确定性，收敛到最近极小值
            - 模拟热运动的"晃动"效应
            """)
        
        st.info("""
        **物理意义**：
        - 势能地形图 = Loss Landscape
        - 低能量区域 = 数据分布（山谷）
        - 朗之万动力学 = 带噪声的梯度下降
        - 温度 = 随机性强度
        """)

        # 添加交互式测验
