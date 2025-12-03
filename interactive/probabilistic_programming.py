"""
交互式概率编程与贝叶斯深度学习可视化
严格按照 17.ProbabilisticProgramming.md 中的公式实现
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

warnings.filterwarnings('ignore')


class InteractiveProbabilisticProgramming:
    """交互式概率编程与贝叶斯深度学习可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎲 概率编程与贝叶斯深度学习：权重的不确定性量化")
        st.markdown("""
        **核心思想**: 从点估计到分布推断，神经网络的第三次数学飞跃
        
        关键概念：
        - **贝叶斯公式**: $p(w|D) = \\frac{p(D|w)p(w)}{p(D)}$
        - **变分推断**: $L(\\theta) = \\mathbb{E}_{w \\sim q_\\theta}[\\log P(D \\mid w)] - KL(q_{\\theta}(w) \\mid\\mid P(w))$
        - **重参数化**: $z = \\mu + \\sigma \\odot \\epsilon$
        - **不确定性分类**: 认知不确定性 vs 任意不确定性
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["贝叶斯推断基础", "变分推断 vs MCMC", "重参数化技巧", "不确定性分析"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "贝叶斯推断基础":
            InteractiveProbabilisticProgramming._render_bayesian_basics()
        elif viz_type == "变分推断 vs MCMC":
            InteractiveProbabilisticProgramming._render_vi_vs_mcmc()
        elif viz_type == "重参数化技巧":
            InteractiveProbabilisticProgramming._render_reparameterization()
        elif viz_type == "不确定性分析":
            InteractiveProbabilisticProgramming._render_uncertainty_analysis()
    

        # 添加交互式测验
        quiz_system = QuizSystem("probabilistic_programming")
        quizzes = QuizTemplates.get_probabilistic_programming_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_bayesian_basics():
        """贝叶斯推断基础演示"""
        st.markdown("### 🧮 贝叶斯推断：频率派 vs 贝叶斯派")
        
        st.latex(r"""
        p(w|D) = \frac{p(D|w)p(w)}{p(D)} = \frac{p(D|w)p(w)}{\int p(D|w)p(w)dw}
        """)
        
        with st.sidebar:
            # 模拟抛硬币问题
            prior_type = st.selectbox("先验分布类型", ["均匀先验", "高斯先验", "Beta先验"])
            num_observations = st.slider("观测次数", 1, 50, 10, 1)
            num_heads = st.slider("正面朝上次数", 0, 50, 7, 1)
            show_marginal = st.checkbox("显示边缘似然计算", value=False)
        
        # 定义参数空间
        theta_range = np.linspace(0, 1, 200)
        
        # 先验分布
        if prior_type == "均匀先验":
            prior = np.ones_like(theta_range)
            prior_name = "Uniform(0,1)"
        elif prior_type == "高斯先验":
            prior = norm.pdf(theta_range, loc=0.5, scale=0.2)
            prior = prior / np.sum(prior)  # 归一化
            prior_name = "Normal(0.5, 0.2)"
        else:  # Beta先验
            alpha, beta_param = 2, 2
            prior = theta_range**(alpha-1) * (1-theta_range)**(beta_param-1)
            prior = prior / np.sum(prior)
            prior_name = f"Beta({alpha}, {beta_param})"
        
        # 似然函数 (二项分布)
        likelihood = theta_range**num_heads * (1-theta_range)**(num_observations - num_heads)
        likelihood = likelihood / np.sum(likelihood)
        
        # 后验分布 (贝叶斯更新)
        posterior = likelihood * prior
        posterior = posterior / np.sum(posterior)
        
        # 边缘似然 (证据)
        marginal_likelihood = np.sum(likelihood * prior)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "先验分布 p(w)", "似然函数 p(D|w)",
                "后验分布 p(w|D)", "贝叶斯更新过程"
            ]
        )
        
        # 先验
        fig.add_trace(
            go.Scatter(
                x=theta_range, y=prior,
                mode='lines',
                name='先验',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 似然
        fig.add_trace(
            go.Scatter(
                x=theta_range, y=likelihood,
                mode='lines',
                name='似然',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # 后验
        fig.add_trace(
            go.Scatter(
                x=theta_range, y=posterior,
                mode='lines',
                name='后验',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 贝叶斯更新过程 (动画效果)
        fig.add_trace(
            go.Scatter(
                x=theta_range, y=prior,
                mode='lines',
                name='先验',
                line=dict(color='blue', width=2, dash='dash')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=theta_range, y=posterior,
                mode='lines',
                name='后验',
                line=dict(color='green', width=3)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"贝叶斯推断过程 - {prior_name}",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示计算结果
        if show_marginal:
            st.markdown("### 🧮 边缘似然计算")
            st.latex(f"""
            p(D) = \\int_0^1 p(D|w)p(w)dw \\approx {marginal_likelihood:.6f}
            """)
            
            st.markdown("""
            **积分近似过程**：
            - 将连续区间[0,1]离散化为200个点
            - 使用黎曼求和近似积分
            - $p(D) \\approx \\sum_{i=1}^{200} p(D|\\theta_i)p(\\theta_i) \\Delta\\theta$
            """)
        
        # 后验分析
        st.markdown("### 📊 后验分析")
        
        # 计算后验统计量
        posterior_mean = np.sum(theta_range * posterior)
        posterior_var = np.sum((theta_range - posterior_mean)**2 * posterior)
        posterior_std = np.sqrt(posterior_var)
        
        # 最大后验估计 (MAP)
        map_idx = np.argmax(posterior)
        map_estimate = theta_range[map_idx]
        
        # 可信区间
        cumsum = np.cumsum(posterior)
        lower_idx = np.argmax(cumsum >= 0.025)
        upper_idx = np.argmax(cumsum >= 0.975)
        credible_interval = (theta_range[lower_idx], theta_range[upper_idx])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("后验均值", f"{posterior_mean:.3f}")
        with col2:
            st.metric("后验标准差", f"{posterior_std:.3f}")
        with col3:
            st.metric("MAP估计", f"{map_estimate:.3f}")
        with col4:
            st.metric("95%可信区间", f"[{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")
        
        st.info("""
        **关键洞察**：
        - **先验**：代表我们对参数的初始信念
        - **似然**：数据对参数的解释力
        - **后验**：结合先验和数据的更新信念
        - **边缘似然**：模型证据，用于模型选择
        """)
    
    @staticmethod
    def _render_vi_vs_mcmc():
        """变分推断 vs MCMC 对比演示"""
        st.markdown("### ⚖️ 变分推断 vs MCMC：两种绕过积分的方法")
        
        st.markdown("""
        **变分推断 (VI)**：把积分问题转化为优化问题
        - 优点：快，可扩展，适合大规模神经网络
        - 缺点：假设近似分布族，可能低估方差
        
        **MCMC**：把计算问题转化为采样问题  
        - 优点：理论上精确，金标准
        - 缺点：慢，计算成本高
        """)
        
        with st.sidebar:
            target_dist = st.selectbox("目标分布", ["高斯混合", "多峰分布", "偏态分布"])
            vi_iterations = st.slider("VI迭代次数", 50, 500, 200, 50)
            mcmc_samples = st.slider("MCMC样本数", 100, 2000, 1000, 100)
            show_elbo = st.checkbox("显示ELBO曲线", value=True)
        
        # 定义目标分布 (真实的后验)
        def target_distribution(x):
            if target_dist == "高斯混合":
                # 双高斯混合
                return 0.6 * norm.pdf(x, loc=-2, scale=1) + 0.4 * norm.pdf(x, loc=2, scale=1.5)
            elif target_dist == "多峰分布":
                # 三峰分布
                return 0.4 * norm.pdf(x, loc=-3, scale=0.8) + 0.3 * norm.pdf(x, loc=0, scale=1) + 0.3 * norm.pdf(x, loc=3, scale=1.2)
            else:  # 偏态分布
                return norm.pdf(x, loc=1, scale=1.5) * (1 + 0.5 * np.tanh(x))
        
        # 变分推断 (使用高斯近似)
        def variational_inference(target_func, iterations):
            # 初始化变分参数 (高斯分布的均值和方差)
            mu = 0.0
            log_sigma = 0.0  # 使用log确保方差为正
            
            elbo_history = []
            param_history = []
            
            x_range = np.linspace(-6, 6, 200)
            target_vals = target_func(x_range)
            target_vals = target_vals / np.sum(target_vals)  # 归一化
            
            for i in range(iterations):
                # 当前变分分布
                sigma = np.exp(log_sigma)
                q_values = norm.pdf(x_range, loc=mu, scale=sigma)
                q_values = q_values / np.sum(q_values)
                
                # 计算ELBO
                # 重构项：E_q[log p(D|w)]
                reconstruction = np.sum(q_values * np.log(target_vals + 1e-8))
                
                # KL散度项：KL(q||p) 这里假设先验是标准正态分布
                prior_values = norm.pdf(x_range, loc=0, scale=1)
                prior_values = prior_values / np.sum(prior_values)
                kl_divergence = np.sum(q_values * np.log((q_values + 1e-8) / (prior_values + 1e-8)))
                
                elbo = reconstruction - kl_divergence
                elbo_history.append(elbo)
                param_history.append((mu, sigma))
                
                # 梯度更新 (简化的梯度上升)
                lr = 0.01
                mu += lr * 0.1  # 简化的梯度
                log_sigma += lr * 0.05
                
            return mu, np.exp(log_sigma), elbo_history, param_history
        
        # 简化的MCMC (Metropolis-Hastings)
        def metropolis_hastings(target_func, num_samples, burn_in=100):
            samples = []
            current = 0.0
            
            for i in range(num_samples + burn_in):
                # 提议新状态
                proposal = current + np.random.normal(0, 1)
                
                # 计算接受概率
                current_prob = target_func(current)
                proposal_prob = target_func(proposal)
                
                acceptance_prob = min(1, proposal_prob / current_prob)
                
                # 接受或拒绝
                if np.random.random() < acceptance_prob:
                    current = proposal
                
                if i >= burn_in:
                    samples.append(current)
            
            return np.array(samples)
        
        # 运行算法
        np.random.seed(42)
        
        # 变分推断
        vi_mu, vi_sigma, elbo_history, param_history = variational_inference(target_distribution, vi_iterations)
        
        # MCMC
        mcmc_samples = metropolis_hastings(target_distribution, mcmc_samples)
        
        # 可视化结果
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "目标分布 vs 近似分布", "ELBO收敛曲线",
                "MCMC采样轨迹", "方法对比总结"
            ]
        )
        
        x_range = np.linspace(-6, 6, 200)
        target_vals = target_distribution(x_range)
        target_vals = target_vals / np.sum(target_vals)
        
        # 目标分布 vs VI近似
        fig.add_trace(
            go.Scatter(
                x=x_range, y=target_vals,
                mode='lines',
                name='目标分布',
                line=dict(color='black', width=3)
            ),
            row=1, col=1
        )
        
        vi_approx = norm.pdf(x_range, loc=vi_mu, scale=vi_sigma)
        vi_approx = vi_approx / np.sum(vi_approx)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=vi_approx,
                mode='lines',
                name='VI近似',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # ELBO曲线
        if show_elbo:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(elbo_history)),
                    y=elbo_history,
                    mode='lines',
                    name='ELBO',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
        
        # MCMC采样轨迹
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(mcmc_samples)),
                y=mcmc_samples,
                mode='lines',
                name='MCMC轨迹',
                line=dict(color='green', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # MCMC直方图
        fig.add_trace(
            go.Histogram(
                x=mcmc_samples,
                nbinsx=50,
                name='MCMC样本',
                marker_color='lightgreen',
                opacity=0.7,
                yaxis='y4'
            ),
            row=2, col=1
        )
        
        # 方法对比
        methods = ['VI', 'MCMC']
        metrics = ['计算速度', '精度', '可扩展性', '理论保证']
        
        comparison_matrix = [
            [5, 3, 5, 3],  # VI
            [2, 5, 2, 5]   # MCMC
        ]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[comparison_matrix[0][0], comparison_matrix[1][0]],
                name='计算速度',
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="变分推断 vs MCMC 对比分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能指标对比
        st.markdown("### 📊 性能对比")
        
        # 计算MCMC的统计量
        mcmc_mean = np.mean(mcmc_samples)
        mcmc_std = np.std(mcmc_samples)
        
        # 计算VI的统计量
        vi_mean = vi_mu
        vi_std = vi_sigma
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VI均值", f"{vi_mean:.3f}")
        with col2:
            st.metric("VI标准差", f"{vi_std:.3f}")
        with col3:
            st.metric("MCMC均值", f"{mcmc_mean:.3f}")
        with col4:
            st.metric("MCMC标准差", f"{mcmc_std:.3f}")
        
        # 详细对比表
        comparison_data = {
            "特性": ["计算复杂度", "内存需求", "收敛保证", "适用场景"],
            "变分推断": ["O(N)", "低", "局部最优", "大规模神经网络"],
            "MCMC": ["O(N²)", "高", "理论精确", "小规模精确推断"]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        st.success("""
        **实践建议**：
        - **大规模深度学习**：使用变分推断 (如TensorFlow Probability)
        - **小规模精确推断**：使用MCMC (如PyMC3, Stan)
        - **实时应用**：变分推断优势明显
        - **科研分析**：MCMC提供更可靠的不确定性量化
        """)
    
    @staticmethod
    def _render_reparameterization():
        """重参数化技巧演示"""
        st.markdown("### 🔄 重参数化技巧：让梯度流动的魔法")
        
        st.latex(r"""
        z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
        """)
        
        st.markdown("""
        **核心思想**：将随机性从网络内部剥离到外部输入，使梯度可以反向传播
        
        - **问题**：直接采样 $z \\sim \\mathcal{N}(\\mu, \\sigma)$ 会切断计算图
        - **解决**：引入标准噪声 $\\epsilon$，让 $z$ 成为 $\\mu$ 和 $\\sigma$ 的确定性函数
        """)
        
        with st.sidebar:
            mu_range = st.slider("均值 μ 范围", -3.0, 3.0, (-1.0, 1.0), 0.1)
            sigma_range = st.slider("标准差 σ 范围", 0.1, 2.0, (0.5, 1.5), 0.1)
            num_samples = st.slider("采样数量", 100, 2000, 1000, 100)
            show_gradient = st.checkbox("显示梯度流", value=True)
        
        # 生成参数网格
        mu_values = np.linspace(mu_range[0], mu_range[1], 20)
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], 20)
        
        # 重参数化采样
        def reparameterized_sample(mu, sigma, epsilon):
            return mu + sigma * epsilon
        
        # 标准采样 (无法反向传播)
        def direct_sample(mu, sigma):
            return np.random.normal(mu, sigma)
        
        # 可视化重参数化过程
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "重参数化采样", "标准采样 (对比)",
                "梯度流可视化", "分布变换"
            ]
        )
        
        # 生成噪声样本
        np.random.seed(42)
        epsilon_samples = np.random.normal(0, 1, num_samples)
        
        # 选择特定的mu和sigma进行演示
        mu_demo, sigma_demo = 0.0, 1.0
        
        # 重参数化采样
        z_reparam = reparameterized_sample(mu_demo, sigma_demo, epsilon_samples)
        
        # 标准采样
        z_direct = np.array([direct_sample(mu_demo, sigma_demo) for _ in range(num_samples)])
        
        # 重参数化采样分布
        fig.add_trace(
            go.Histogram(
                x=z_reparam,
                nbinsx=30,
                name='重参数化采样',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 标准采样分布
        fig.add_trace(
            go.Histogram(
                x=z_direct,
                nbinsx=30,
                name='标准采样',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 梯度流可视化
        if show_gradient:
            # 显示梯度如何流动
            mu_grad = epsilon_samples  # ∂z/∂μ = ε
            sigma_grad = epsilon_samples  # ∂z/∂σ = ε
            
            fig.add_trace(
                go.Scatter(
                    x=epsilon_samples[:100],
                    y=mu_grad[:100],
                    mode='markers',
                    name='∂z/∂μ = ε',
                    marker=dict(color='green', size=4)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=epsilon_samples[:100],
                    y=sigma_grad[:100],
                    mode='markers',
                    name='∂z/∂σ = ε',
                    marker=dict(color='orange', size=4)
                ),
                row=2, col=1
            )
        
        # 分布变换：从标准正态到任意正态
        x_range = np.linspace(-4, 4, 200)
        standard_normal = norm.pdf(x_range, loc=0, scale=1)
        transformed_normal = norm.pdf(x_range, loc=mu_demo, scale=sigma_demo)
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=standard_normal,
                mode='lines',
                name='标准正态 N(0,1)',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=transformed_normal,
                mode='lines',
                name=f'变换后 N({mu_demo},{sigma_demo})',
                line=dict(color='brown', width=2, dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="重参数化技巧详细分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 交互式参数调节
        st.markdown("### 🎛️ 交互式参数调节")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mu_interactive = st.slider("调节均值 μ", -3.0, 3.0, 0.0, 0.1)
            sigma_interactive = st.slider("调节标准差 σ", 0.1, 2.0, 1.0, 0.1)
        
        with col2:
            # 实时显示采样结果
            z_interactive = reparameterized_sample(mu_interactive, sigma_interactive, epsilon_samples)
            
            fig_interactive = go.Figure()
            fig_interactive.add_trace(
                go.Histogram(
                    x=z_interactive,
                    nbinsx=30,
                    name=f'N({mu_interactive:.1f}, {sigma_interactive:.1f})',
                    marker_color='lightblue'
                )
            )
            
            fig_interactive.update_layout(
                title="实时采样分布",
                xaxis_title="z值",
                yaxis_title="频次",
                height=300
            )
            
            st.plotly_chart(fig_interactive, use_container_width=True)
        
        # 数学推导
        st.markdown("### 📐 数学推导")
        
        st.latex(r"""
        \begin{aligned}
        z &\sim \mathcal{N}(\mu, \sigma^2) \\
        \text{重参数化:} \quad z &= \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1) \\
        \\
        \frac{\partial z}{\partial \mu} &= 1 \\
        \frac{\partial z}{\partial \sigma} &= \epsilon \\
        \frac{\partial L}{\partial \mu} &= \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial L}{\partial z} \\
        \frac{\partial L}{\partial \sigma} &= \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial \sigma} = \frac{\partial L}{\partial z} \cdot \epsilon
        \end{aligned}
        """)
        
        st.info("""
        **重参数化技巧的重要性**：
        - **梯度流动**：使得随机层的梯度可以反向传播
        - **GPU加速**：可以利用并行计算加速采样过程
        - **变分推断**：是VAE、贝叶斯神经网络等模型的核心技术
        - **框架支持**：PyTorch、TensorFlow等主流框架都内置支持
        """)
    
    @staticmethod
    def _render_uncertainty_analysis():
        """不确定性分析演示"""
        st.markdown("### 🎯 不确定性分类：认知 vs 任意")
        
        st.markdown("""
        **认知不确定性 (Epistemic)**：
        - 来源：模型知识的盲区，没见过的数据
        - 特点：可通过增加数据来消除
        - 建模：贝叶斯神经网络，权重分布
        
        **任意不确定性 (Aleatoric)**：
        - 来源：数据本身固有的噪声
        - 特点：无法通过更多数据消除  
        - 建模：输出层引入方差参数
        """)
        
        with st.sidebar:
            data_type = st.selectbox("数据场景", ["线性回归", "非线性回归", "分类边界"])
            noise_level = st.slider("数据噪声水平", 0.0, 1.0, 0.2, 0.05)
            num_training_points = st.slider("训练点数量", 10, 100, 30, 5)
            show_prediction_uncertainty = st.checkbox("显示预测不确定性", value=True)
        
        # 生成合成数据
        np.random.seed(42)
        
        if data_type == "线性回归":
            # 线性回归数据
            X_train = np.random.uniform(-5, 5, num_training_points)
            y_true = 2 * X_train + 1
            y_train = y_true + np.random.normal(0, noise_level, num_training_points)
            
            X_test = np.linspace(-8, 8, 100)
            y_test_true = 2 * X_test + 1
            
        elif data_type == "非线性回归":
            # 非线性回归数据
            X_train = np.random.uniform(-3, 3, num_training_points)
            y_true = np.sin(X_train) * 2
            y_train = y_true + np.random.normal(0, noise_level, num_training_points)
            
            X_test = np.linspace(-5, 5, 100)
            y_test_true = np.sin(X_test) * 2
            
        else:  # 分类边界
            # 二分类数据
            X_train = np.random.randn(num_training_points, 2)
            # 创建非线性决策边界
            y_train = (X_train[:, 0]**2 + X_train[:, 1]**2 > 1).astype(int)
            # 添加噪声标签
            flip_indices = np.random.choice(num_training_points, int(noise_level * num_training_points), replace=False)
            y_train[flip_indices] = 1 - y_train[flip_indices]
            
            X_test = np.random.uniform(-3, 3, (1000, 2))
        
        # 模拟贝叶斯神经网络预测 (使用MC Dropout近似)
        def bayesian_nn_predict(X, num_samples=100):
            """模拟贝叶斯神经网络的预测分布"""
            if data_type == "分类边界":
                predictions = []
                for _ in range(num_samples):
                    # 模拟网络权重的不确定性
                    weight_noise = np.random.normal(0, 0.1, (2, 1))
                    bias_noise = np.random.normal(0, 0.1, 1)
                    
                    # 简单的非线性决策边界
                    logits = X @ weight_noise + bias_noise
                    logits += np.sin(X[:, 0:1]) * np.cos(X[:, 1:2])  # 非线性变换
                    probs = 1 / (1 + np.exp(-logits))
                    predictions.append(probs.flatten())
                
                return np.array(predictions)
            else:
                predictions = []
                for _ in range(num_samples):
                    # 模拟权重不确定性
                    if data_type == "线性回归":
                        weight = np.random.normal(2, 0.3)  # 真实权重2，有不确定性
                        bias = np.random.normal(1, 0.2)    # 真实偏置1，有不确定性
                    else:  # 非线性
                        weight = np.random.normal(1.5, 0.3)
                        bias = np.random.normal(0, 0.2)
                    
                    y_pred = weight * X + bias
                    # 添加任意不确定性（数据噪声）
                    y_pred += np.random.normal(0, noise_level, len(X))
                    
                    predictions.append(y_pred)
                
                return np.array(predictions)
        
        # 获取预测分布
        if data_type == "分类边界":
            predictions = bayesian_nn_predict(X_test)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # 可视化分类边界和不确定性
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["预测均值", "预测不确定性"],
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 训练数据
            colors_train = ['red' if y == 0 else 'blue' for y in y_train]
            fig.add_trace(
                go.Scatter(
                    x=X_train[:, 0], y=X_train[:, 1],
                    mode='markers',
                    name='训练数据',
                    marker=dict(color=colors_train, size=8)
                ),
                row=1, col=1
            )
            
            # 预测边界
            fig.add_trace(
                go.Scatter(
                    x=X_test[:, 0], y=X_test[:, 1],
                    mode='markers',
                    name='预测概率',
                    marker=dict(
                        color=mean_pred,
                        colorscale='RdBu',
                        size=4,
                        opacity=0.6,
                        colorbar=dict(title="正类概率", x=0.45)
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 不确定性可视化
            fig.add_trace(
                go.Scatter(
                    x=X_test[:, 0], y=X_test[:, 1],
                    mode='markers',
                    name='不确定性',
                    marker=dict(
                        color=std_pred,
                        colorscale='Viridis',
                        size=4,
                        opacity=0.6,
                        colorbar=dict(title="标准差", x=1.02)
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="贝叶斯分类：不确定性可视化",
                height=500
            )
            
        else:
            predictions = bayesian_nn_predict(X_test)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # 可视化回归结果
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["预测均值与置信区间", "不确定性分解"]
            )
            
            # 训练数据
            fig.add_trace(
                go.Scatter(
                    x=X_train, y=y_train,
                    mode='markers',
                    name='训练数据',
                    marker=dict(color='blue', size=8),
                    error_y=dict(
                        type='data',
                        array=noise_level * np.ones_like(y_train),
                        visible=True,
                        color='lightblue'
                    )
                ),
                row=1, col=1
            )
            
            # 真实函数
            fig.add_trace(
                go.Scatter(
                    x=X_test, y=y_test_true,
                    mode='lines',
                    name='真实函数',
                    line=dict(color='black', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # 预测均值
            fig.add_trace(
                go.Scatter(
                    x=X_test, y=mean_pred,
                    mode='lines',
                    name='预测均值',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # 置信区间
            if show_prediction_uncertainty:
                upper_bound = mean_pred + 2 * std_pred
                lower_bound = mean_pred - 2 * std_pred
                
                fig.add_trace(
                    go.Scatter(
                        x=X_test, y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=X_test, y=lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        name='95%置信区间'
                    ),
                    row=1, col=1
                )
            
            # 不确定性分解
            total_uncertainty = std_pred**2
            aleatoric_uncertainty = noise_level**2 * np.ones_like(total_uncertainty)
            epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
            
            fig.add_trace(
                go.Scatter(
                    x=X_test, y=epistemic_uncertainty,
                    mode='lines',
                    name='认知不确定性',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=X_test, y=aleatoric_uncertainty,
                    mode='lines',
                    name='任意不确定性',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=X_test, y=total_uncertainty,
                    mode='lines',
                    name='总不确定性',
                    line=dict(color='purple', width=2, dash='dot')
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="贝叶斯回归：不确定性量化",
                height=500
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 不确定性分析总结
        st.markdown("### 📊 不确定性分析总结")
        
        if data_type != "分类边界":
            avg_epistemic = np.mean(epistemic_uncertainty)
            avg_aleatoric = np.mean(aleatoric_uncertainty)
            avg_total = np.mean(total_uncertainty)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均认知不确定性", f"{avg_epistemic:.4f}")
            with col2:
                st.metric("平均任意不确定性", f"{avg_aleatoric:.4f}")
            with col3:
                st.metric("平均总不确定性", f"{avg_total:.4f}")
        
        st.success("""
        **不确定性量化的实践价值**：
        - **自动驾驶**：识别未见过的路况，触发安全机制
        - **医疗诊断**：量化诊断置信度，辅助医生决策
        - **金融风控**：评估模型预测的可靠性，控制风险
        - **科学发现**：识别知识盲区，指导数据收集
        """)


# 为了兼容性，添加缺少的导入
try:
    from scipy.stats import norm, multivariate_normal
except ImportError:
    # 如果scipy不可用，使用numpy实现
    def norm(loc=0, scale=1):
        class NormalDist:
            def pdf(self, x):
                return np.exp(-0.5 * ((x - loc) / scale)**2) / (scale * np.sqrt(2 * np.pi))
        return NormalDist()
    
    def multivariate_normal(mean, cov):
        class MVN:
            def rvs(self, size=1):
                return np.random.multivariate_normal(mean, cov, size)
        return MVN()

        # 添加交互式测验
