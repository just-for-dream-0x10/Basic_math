"""
交互式信息几何与自然梯度可视化
严格按照 19.Information_Geometry.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import norm, multivariate_normal
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation

warnings.filterwarnings('ignore')


class InteractiveInformationGeometry:
    """交互式信息几何与自然梯度可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("📐 信息几何与自然梯度：黎曼流形上的优化")
        st.markdown(r"""
        **核心思想**: 参数空间的欧氏距离不等于概率分布的距离，需要在黎曼流形上进行优化
        
        **关键概念**：
        """)
        
        st.markdown("**1. 费雪信息矩阵**:")
        st.latex(r"""
        \mathbf{F} = \mathbb{E}[\nabla_\theta \log p(x|\theta) \nabla_\theta \log p(x|\theta)^T]
        """)
        
        st.markdown("**2. KL散度近似**:")
        st.latex(r"""
        D_{KL}(p_\theta \| p_{\theta+\delta}) \approx \frac{1}{2} \delta^T \mathbf{F}(\theta) \delta
        """)
        
        st.markdown("**3. 自然梯度**:")
        st.latex(r"""
        \tilde{\nabla} L = \mathbf{F}^{-1} \nabla L
        """)
        
        st.markdown("**4. Adam近似**:")
        st.latex(r"""
        \Delta \theta \propto -\frac{1}{\sqrt{v_t}} \nabla L \approx -\frac{1}{\sqrt{\text{diag}(\mathbf{F})}} \nabla L
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["参数空间 vs 概率空间", "费雪信息矩阵", "自然梯度 vs SGD", "Adam的几何解释"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "参数空间 vs 概率空间":
            InteractiveInformationGeometry._render_parameter_vs_probability()
        elif viz_type == "费雪信息矩阵":
            InteractiveInformationGeometry._render_fisher_information()
        elif viz_type == "自然梯度 vs SGD":
            InteractiveInformationGeometry._render_natural_gradient()
        elif viz_type == "Adam的几何解释":
            InteractiveInformationGeometry._render_adam_geometry()
    

        # 添加交互式测验
        quiz_system = QuizSystem("information_geometry")
        quizzes = QuizTemplates.get_information_geometry_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_parameter_vs_probability():
        """参数空间vs概率空间演示"""
        st.markdown("### 🌍 参数空间 ≠ 概率空间")
        
        st.markdown("""
        **核心洞察**：参数的数值距离不代表模型的行为距离
        
        - **情况A**：σ = 10，μ从0变到1，分布几乎没变
        - **情况B**：σ = 0.01，μ从0变到1，分布完全分离
        
        在参数空间中位移都是1，但在概率流形上距离完全不同！
        """)
        
        with st.sidebar:
            sigma_a = st.slider("情况A的σ", 5.0, 20.0, 10.0, 0.5)
            sigma_b = st.slider("情况B的σ", 0.005, 0.05, 0.01, 0.001)
            mu_range = st.slider("μ变化范围", 0.5, 3.0, 1.0, 0.1)
            show_overlap = st.checkbox("显示重叠区域", value=True)
        
        # 创建x轴范围
        x = np.linspace(-5, 5, 1000)
        
        # 情况A：大方差
        dist_a1 = norm.pdf(x, loc=0, scale=sigma_a)
        dist_a2 = norm.pdf(x, loc=mu_range, scale=sigma_a)
        
        # 情况B：小方差
        dist_b1 = norm.pdf(x, loc=0, scale=sigma_b)
        dist_b2 = norm.pdf(x, loc=mu_range, scale=sigma_b)
        
        # 计算KL散度
        kl_a = np.sum(dist_a1 * np.log((dist_a1 + 1e-10) / (dist_a2 + 1e-10))) * (x[1] - x[0])
        kl_b = np.sum(dist_b1 * np.log((dist_b1 + 1e-10) / (dist_b2 + 1e-10))) * (x[1] - x[0])
        
        # 计算重叠面积
        if show_overlap:
            overlap_a = np.minimum(dist_a1, dist_a2)
            overlap_b = np.minimum(dist_b1, dist_b2)
            overlap_area_a = np.sum(overlap_a) * (x[1] - x[0])
            overlap_area_b = np.sum(overlap_b) * (x[1] - x[0])
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "情况A：大方差分布", "情况B：小方差分布",
                "参数空间距离", "概率空间距离"
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 情况A分布
        fig.add_trace(
            go.Scatter(
                x=x, y=dist_a1,
                mode='lines',
                name='μ=0',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=dist_a2,
                mode='lines',
                name=f'μ={mu_range}',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        if show_overlap:
            fig.add_trace(
                go.Scatter(
                    x=x, y=overlap_a,
                    mode='lines',
                    name='重叠',
                    line=dict(color='green', width=3),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # 情况B分布
        fig.add_trace(
            go.Scatter(
                x=x, y=dist_b1,
                mode='lines',
                name='μ=0',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=dist_b2,
                mode='lines',
                name=f'μ={mu_range}',
                line=dict(color='red', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        if show_overlap:
            fig.add_trace(
                go.Scatter(
                    x=x, y=overlap_b,
                    mode='lines',
                    name='重叠',
                    line=dict(color='green', width=3),
                    fill='tonexty',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 参数空间距离（都是1）
        fig.add_trace(
            go.Bar(
                x=['情况A', '情况B'],
                y=[1, 1],
                name='参数距离',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # 概率空间距离（KL散度）
        fig.add_trace(
            go.Bar(
                x=['情况A', '情况B'],
                y=[abs(kl_a), abs(kl_b)],
                name='KL散度',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="参数空间 vs 概率空间对比",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 数值分析
        st.markdown("### 📊 数值分析")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("情况A KL散度", f"{abs(kl_a):.6f}")
        with col2:
            st.metric("情况B KL散度", f"{abs(kl_b):.6f}")
        with col3:
            if show_overlap:
                st.metric("情况A重叠面积", f"{overlap_area_a:.4f}")
        with col4:
            if show_overlap:
                st.metric("情况B重叠面积", f"{overlap_area_b:.4f}")
        
        st.warning("""
        **关键结论**：
        - 参数空间相同距离 ≠ 概率空间相同距离
        - 小方差区域对参数变化更敏感
        - 优化应该基于概率分布的变化，而非参数数值的变化
        """)
    
    @staticmethod
    def _render_fisher_information():
        """费雪信息矩阵演示"""
        st.markdown("### 🧮 费雪信息矩阵：概率流形的度量张量")
        
        st.latex(r"""
        \mathbf{F} = \mathbb{E}_{x \sim p(x|\theta)} \left[ \nabla_\theta \log p(x|\theta) \nabla_\theta \log p(x|\theta)^T \right]
        """)
        
        with st.sidebar:
            dist_type = st.selectbox("分布类型", ["高斯分布", "伯努利分布", "多项分布"])
            param_ranges = st.slider("参数范围", 5, 50, 20, 5)
            show_eigendecomposition = st.checkbox("显示特征分解", value=True)
        
        if dist_type == "高斯分布":
            # 高斯分布的费雪信息矩阵
            mu_range = np.linspace(-2, 2, param_ranges)
            sigma_range = np.linspace(0.5, 3, param_ranges)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "FIM(μ,μ)", "FIM(σ,σ)", 
                    "条件数分布", "特征值分析"
                ]
            )
            
            # 计算费雪信息矩阵
            F_mu_mu = np.zeros((len(mu_range), len(sigma_range)))
            F_sigma_sigma = np.zeros((len(mu_range), len(sigma_range)))
            condition_numbers = np.zeros((len(mu_range), len(sigma_range)))
            
            for i, mu in enumerate(mu_range):
                for j, sigma in enumerate(sigma_range):
                    # 高斯分布的FIM解析解
                    F_mu_mu[i, j] = 1 / (sigma ** 2)
                    F_sigma_sigma[i, j] = 2 / (sigma ** 2)
                    condition_numbers[i, j] = F_sigma_sigma[i, j] / F_mu_mu[i, j]
            
            # FIM(μ,μ)
            fig.add_trace(
                go.Heatmap(
                    z=F_mu_mu,
                    x=mu_range,
                    y=sigma_range,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=1, col=1
            )
            
            # FIM(σ,σ)
            fig.add_trace(
                go.Heatmap(
                    z=F_sigma_sigma,
                    x=mu_range,
                    y=sigma_range,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=1, col=2
            )
            
            # 条件数
            fig.add_trace(
                go.Heatmap(
                    z=condition_numbers,
                    x=mu_range,
                    y=sigma_range,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="条件数")
                ),
                row=2, col=1
            )
            
            # 特征值分析（选择几个点）
            if show_eigendecomposition:
                sample_points = [(0, 0.5), (0, 1.5), (0, 2.5)]
                colors = ['red', 'green', 'blue']
                
                for idx, (mu, sigma) in enumerate(sample_points):
                    F = np.array([[1/(sigma**2), 0], [0, 2/(sigma**2)]])
                    eigenvals = np.linalg.eigvals(F)
                    
                    fig.add_trace(
                        go.Bar(
                            x=[f'点{idx+1}_λ1', f'点{idx+1}_λ2'],
                            y=eigenvals,
                            name=f'σ={sigma}',
                            marker_color=colors[idx]
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title="高斯分布的费雪信息矩阵",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 几何解释
            st.markdown("### 📐 几何解释")
            
            st.markdown("""
            **高斯分布FIM的几何意义**：
            - **F(μ,μ) = 1/σ²**：σ越小，μ方向越敏感（曲率越大）
            - **F(σ,σ) = 2/σ²**：σ方向的敏感度是μ方向的2倍
            - **条件数 = 2**：固定比例，各向异性程度恒定
            """)
        
        elif dist_type == "伯努利分布":
            # 伯努利分布的费雪信息
            p_range = np.linspace(0.01, 0.99, param_ranges)
            F_values = 1 / (p_range * (1 - p_range))
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=p_range,
                    y=F_values,
                    mode='lines',
                    name='F(p)',
                    line=dict(width=3)
                )
            )
            
            fig.update_layout(
                title="伯努利分布的费雪信息",
                xaxis_title="参数 p",
                yaxis_title="费雪信息 F(p)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **伯努利分布特点**：
            - F(p) = 1/(p(1-p))，在p=0.5处最小
            - p接近0或1时，F(p)→∞（高曲率区域）
            - 这解释了为什么分类问题在边界处敏感
            """)
        
        st.success("""
        **费雪信息矩阵的核心作用**：
        - **局部度量**：定义概率流形上的局部距离
        - **曲率信息**：反映分布对参数变化的敏感度
        - **优化指导**：告诉我们在哪个方向应该走多远
        """)
    
    @staticmethod
    def _render_natural_gradient():
        """自然梯度vs SGD演示"""
        st.markdown("### 🧭 自然梯度 vs 普通梯度")
        
        st.latex(r"""
        \theta_{t+1} = \theta_t - \eta \mathbf{F}^{-1} \nabla L(\theta_t)
        """)
        
        with st.sidebar:
            true_mu = st.slider("真实μ", 1.0, 5.0, 4.0, 0.1)
            true_sigma = st.slider("真实σ", 1.0, 5.0, 3.0, 0.1)
            init_mu = st.slider("初始μ", -5.0, 0.0, -2.0, 0.5)
            init_sigma = st.slider("初始σ", 0.1, 2.0, 0.5, 0.1)
            learning_rate_sgd = st.slider("SGD学习率", 0.001, 0.1, 0.01, 0.001)
            learning_rate_nat = st.slider("自然梯度学习率", 0.01, 1.0, 0.2, 0.01)
            num_steps = st.slider("优化步数", 20, 100, 50, 5)
        
        # 定义损失函数（KL散度）
        def loss_function(mu, sigma):
            return (np.log(sigma) + 
                   (true_sigma**2 + (true_mu - mu)**2) / (2 * sigma**2))
        
        def get_gradients(mu, sigma):
            # 计算梯度
            grad_mu = (mu - true_mu) / (sigma**2)
            grad_sigma = (1.0/sigma) - ((true_sigma**2 + (true_mu - mu)**2) / (sigma**3))
            return np.array([grad_mu, grad_sigma])
        
        def get_fisher_inverse(sigma):
            # 高斯分布的FIM逆矩阵（对角线）
            return np.array([sigma**2, 0.5 * sigma**2])
        
        # 优化过程
        def optimize(method, lr, steps):
            mu, sigma = init_mu, init_sigma
            path = [[mu, sigma]]
            
            for _ in range(steps):
                grad = get_gradients(mu, sigma)
                
                if method == 'SGD':
                    update = -lr * grad
                elif method == 'Natural':
                    F_inv = get_fisher_inverse(sigma)
                    update = -lr * (grad * F_inv)  # 元素乘法（对角矩阵）
                
                mu += update[0]
                sigma += update[1]
                sigma = max(sigma, 0.05)  # 保持数值稳定性
                path.append([mu, sigma])
            
            return np.array(path)
        
        # 运行优化
        sgd_path = optimize('SGD', learning_rate_sgd, num_steps)
        nat_path = optimize('Natural', learning_rate_nat, num_steps)
        
        # 创建损失等高线
        mu_range = np.linspace(-3, 5, 50)
        sigma_range = np.linspace(0.1, 4, 50)
        M, S = np.meshgrid(mu_range, sigma_range)
        
        # 计算损失函数值
        Loss = np.log(S) + (true_sigma**2 + (true_mu - M)**2) / (2 * S**2)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["优化轨迹", "收敛过程"]
        )
        
        # 优化轨迹
        fig.add_trace(
            go.Contour(
                x=mu_range, y=sigma_range, z=Loss,
                colorscale='Gray',
                showscale=False,
                contours=dict(showlabels=False),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[true_mu], y=[true_sigma],
                mode='markers',
                marker=dict(color='black', size=15, symbol='star'),
                name='最优点'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sgd_path[:, 0], y=sgd_path[:, 1],
                mode='lines+markers',
                name='SGD',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=nat_path[:, 0], y=nat_path[:, 1],
                mode='lines+markers',
                name='自然梯度',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # 收敛过程
        sgd_losses = [loss_function(p[0], p[1]) for p in sgd_path]
        nat_losses = [loss_function(p[0], p[1]) for p in nat_path]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sgd_losses))),
                y=sgd_losses,
                mode='lines',
                name='SGD损失',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(nat_losses))),
                y=nat_losses,
                mode='lines',
                name='自然梯度损失',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="自然梯度 vs SGD 优化对比",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能分析
        st.markdown("### 📊 性能分析")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sgd_final_loss = sgd_losses[-1]
            st.metric("SGD最终损失", f"{sgd_final_loss:.4f}")
        with col2:
            nat_final_loss = nat_losses[-1]
            st.metric("自然梯度最终损失", f"{nat_final_loss:.4f}")
        with col3:
            sgd_distance = np.sqrt((sgd_path[-1, 0] - true_mu)**2 + (sgd_path[-1, 1] - true_sigma)**2)
            st.metric("SGD最终距离", f"{sgd_distance:.3f}")
        with col4:
            nat_distance = np.sqrt((nat_path[-1, 0] - true_mu)**2 + (nat_path[-1, 1] - true_sigma)**2)
            st.metric("自然梯度最终距离", f"{nat_distance:.3f}")
        
        st.success("""
        **自然梯度的优势**：
        - **几何感知**：考虑概率流形的曲率
        - **自适应步长**：在高曲率区域自动减小步长
        - **直接路径**：沿着测地线走向最优点
        - **数值稳定**：避免梯度爆炸问题
        """)
    
    @staticmethod
    def _render_adam_geometry():
        """Adam的几何解释演示"""
        st.markdown("### 🤖 Adam的几何解释：FIM的对角近似")
        
        st.latex(r"""
        \Delta \theta \propto -\frac{1}{\sqrt{v_t}} \nabla L \approx -\frac{1}{\sqrt{\text{diag}(\mathbf{F})}} \nabla L
        """)
        
        with st.sidebar:
            dimension = st.slider("参数维度", 2, 10, 5, 1)
            correlation = st.slider("参数间相关性", 0.0, 0.9, 0.7, 0.1)
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            beta1 = st.slider("β1", 0.8, 0.99, 0.9, 0.01)
            beta2 = st.slider("β2", 0.9, 0.999, 0.999, 0.001)
            num_steps = st.slider("优化步数", 50, 200, 100, 10)
        
        # 创建相关的高斯分布作为目标
        np.random.seed(42)
        
        # 构建相关矩阵
        true_cov = np.ones((dimension, dimension)) * correlation
        np.fill_diagonal(true_cov, 1.0)
        true_mean = np.zeros(dimension)
        
        # 初始化参数
        theta = np.random.randn(dimension) * 2
        m = np.zeros(dimension)  # 一阶矩
        v = np.zeros(dimension)  # 二阶矩
        
        # 记录优化过程
        theta_history = [theta.copy()]
        loss_history = []
        
        # 模拟优化过程
        for step in range(num_steps):
            # 计算损失和梯度（简化的二次损失）
            loss = 0.5 * theta.T @ np.linalg.inv(true_cov) @ theta
            grad = np.linalg.inv(true_cov) @ theta
            
            # Adam更新
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))
            
            # 更新参数
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            theta_history.append(theta.copy())
            loss_history.append(loss)
        
        theta_history = np.array(theta_history)
        
        # 计算真实的FIM
        true_fim = np.linalg.inv(true_cov)
        fim_diagonal = np.diag(true_fim)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "参数轨迹（前2维）", "损失收敛",
                "自适应步长", "FIM vs Adam估计"
            ]
        )
        
        # 参数轨迹（前2维）
        fig.add_trace(
            go.Scatter(
                x=theta_history[:, 0],
                y=theta_history[:, 1],
                mode='lines+markers',
                name='Adam轨迹',
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name='最优点'
            ),
            row=1, col=1
        )
        
        # 损失收敛
        fig.add_trace(
            go.Scatter(
                x=list(range(len(loss_history))),
                y=loss_history,
                mode='lines',
                name='损失',
                line=dict(width=2)
            ),
            row=1, col=2
        )
        
        # 自适应步长
        step_sizes = learning_rate / (np.sqrt(v_hat) + 1e-8)
        
        # 记录每个维度的步长历史（需要重新计算）
        step_size_history = []
        for step in range(num_steps):
            # 重新计算该步的步长
            temp_v = np.zeros(dimension)
            temp_m = np.zeros(dimension)
            
            # 模拟到该步的更新
            for s in range(step + 1):
                grad = np.linalg.inv(true_cov) @ theta_history[s]
                temp_m = beta1 * temp_m + (1 - beta1) * grad
                temp_v = beta2 * temp_v + (1 - beta2) * (grad ** 2)
            
            v_hat_current = temp_v / (1 - beta2 ** (step + 1))
            step_sizes_current = learning_rate / (np.sqrt(v_hat_current) + 1e-8)
            step_size_history.append(step_sizes_current.copy())
        
        step_size_history = np.array(step_size_history)
        
        # 绘制每个维度的步长变化
        for i in range(min(dimension, 3)):  # 只显示前3个维度
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(step_size_history))),
                    y=step_size_history[:, i],
                    mode='lines',
                    name=f'维度{i+1}步长',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # FIM vs Adam估计
        dimensions_show = min(dimension, 5)
        fig.add_trace(
            go.Bar(
                x=[f'FIM_{i+1}' for i in range(dimensions_show)],
                y=fim_diagonal[:dimensions_show],
                name='真实FIM对角线',
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=[f'Adam_{i+1}' for i in range(dimensions_show)],
                y=v_hat[:dimensions_show],
                name='Adam二阶矩估计',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Adam算法的几何分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 理论分析
        st.markdown("### 📐 理论分析")
        
        st.markdown("""
        **Adam作为自然梯度的近似**：
        
        1. **完整自然梯度**：使用完整的FIM矩阵 $\mathbf{F}^{-1}$
        2. **Adam近似**：只使用对角线元素 $1/\sqrt{\text{diag}(\mathbf{F})}$
        3. **假设**：参数间相互独立，忽略相关性
        4. **效果**：在高维情况下计算可行，但丢失了相关信息
        """)
        
        # 性能指标
        st.markdown("### 📊 性能指标")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_loss = loss_history[-1]
            st.metric("最终损失", f"{final_loss:.6f}")
        with col2:
            convergence_step = next((i for i, l in enumerate(loss_history) 
                                   if l < loss_history[0] * 0.1), len(loss_history))
            st.metric("收敛步数", f"{convergence_step}")
        with col3:
            final_norm = np.linalg.norm(theta)
            st.metric("最终参数范数", f"{final_norm:.4f}")
        with col4:
            fim_condition = np.linalg.cond(true_fim)
            st.metric("FIM条件数", f"{fim_condition:.2f}")
        
        st.info("""
        **Adam的几何意义**：
        - **自适应缩放**：每个参数根据其二阶矩自适应缩放
        - **对角近似**：计算高效，但忽略了参数间相关性
        - **实践效果**：在深度学习中表现优异，是自然梯度的实用版本
        - **理论保证**：在凸优化条件下有收敛保证
        """)

        # 添加交互式测验
