"""
交互式神经进化与进化策略可视化
严格按照 14.Neuroevolution.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import multivariate_normal
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation

warnings.filterwarnings('ignore')


class InteractiveNeuroevolution:
    """交互式神经进化与进化策略可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🧬 神经进化与进化策略：零阶优化超越梯度")
        st.markdown("""
        **核心思想**: 进化策略通过群体搜索和随机采样，在不可微、非凸、稀疏奖励的场景中超越梯度下降
        
        关键概念：
        - **零阶优化**: $\\nabla_\\theta J(\\theta) \\approx \\frac{1}{\\sigma N} \\sum_{i=1}^N F(\\theta + \\epsilon_i) \\cdot \\epsilon_i$
        - **高斯平滑**: $J(\\theta) = \\mathbb{E}_{\\epsilon \\sim \\mathcal{N}(\\theta, \\sigma^2 I)} [F(\\theta + \\epsilon)]$
        - **镜像采样**: $R_i^+ = F(\\theta_t + \\epsilon_i \\sigma)$ 和 $R_i^- = F(\\theta_t - \\epsilon_i \\sigma)$
        - **协方差适应**: CMA-ES 学习参数相关性，椭球探索
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["ES vs 梯度下降", "OpenAI ES算法", "PBT种群训练", "CMA-ES协方差适应"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "ES vs 梯度下降":
            InteractiveNeuroevolution._render_es_vs_gd()
        elif viz_type == "OpenAI ES算法":
            InteractiveNeuroevolution._render_openai_es()
        elif viz_type == "PBT种群训练":
            InteractiveNeuroevolution._render_pbt()
        elif viz_type == "CMA-ES协方差适应":
            InteractiveNeuroevolution._render_cma_es()
    

        # 添加交互式测验
        quiz_system = QuizSystem("neuroevolution")
        quizzes = QuizTemplates.get_neuroevolution_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_es_vs_gd():
        """ES vs 梯度下降对比演示"""
        st.markdown("### 🥾 登山者 vs 空降兵：ES vs 梯度下降")
        
        with st.sidebar:
            test_function = st.selectbox("测试函数", 
                ["Rastrigin (多峰)", "Rosenbrock (峡谷)", "Ackley (平台)", "Sphere (简单)"])
            num_iterations = st.slider("迭代次数", 50, 200, 100, 10)
            population_size = st.slider("种群大小", 10, 100, 50, 10)
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            noise_std = st.slider("噪声标准差", 0.01, 0.5, 0.1, 0.01)
        
        # 定义测试函数
        def rastrigin(x):
            A = 10
            return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
        
        def rosenbrock(x):
            return sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)])
        
        def ackley(x):
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = sum([xi**2 for xi in x])
            sum2 = sum([np.cos(c * xi) for xi in x])
            return -a * np.exp(-b * np.sqrt(sum1 / len(x))) - np.exp(sum2 / len(x)) + a + np.e
        
        def sphere(x):
            return sum([xi**2 for xi in x])
        
        # 选择测试函数
        if test_function == "Rastrigin (多峰)":
            func = rastrigin
            bounds = (-5.12, 5.12)
        elif test_function == "Rosenbrock (峡谷)":
            func = rosenbrock
            bounds = (-2, 2)
        elif test_function == "Ackley (平台)":
            func = ackley
            bounds = (-5, 5)
        else:  # Sphere
            func = sphere
            bounds = (-2, 2)
        
        # 梯度下降 (需要数值梯度)
        def numerical_gradient(f, x, eps=1e-6):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
            return grad
        
        # 进化策略
        def evolution_strategy(f, x0, num_iter, pop_size, sigma, lr):
            x = x0.copy()
            history = [x.copy()]
            fitness_history = [f(x)]
            
            for t in range(num_iter):
                # 生成噪声样本
                epsilon = np.random.randn(pop_size, len(x))
                fitness = np.array([f(x + sigma * e) for e in epsilon])
                
                # 标准化适应度
                fitness = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)
                
                # ES更新
                gradient_estimate = np.dot(epsilon.T, fitness) / (pop_size * sigma)
                x = x + lr * gradient_estimate
                
                history.append(x.copy())
                fitness_history.append(f(x))
            
            return np.array(history), np.array(fitness_history)
        
        # 梯度下降
        def gradient_descent(f, x0, num_iter, lr):
            x = x0.copy()
            history = [x.copy()]
            fitness_history = [f(x)]
            
            for t in range(num_iter):
                grad = numerical_gradient(f, x)
                x = x - lr * grad
                
                history.append(x.copy())
                fitness_history.append(f(x))
            
            return np.array(history), np.array(fitness_history)
        
        # 初始化
        np.random.seed(42)
        dim = 2
        x0 = np.random.uniform(bounds[0], bounds[1], dim)
        
        # 运行算法
        es_history, es_fitness = evolution_strategy(func, x0, num_iterations, population_size, noise_std, learning_rate)
        gd_history, gd_fitness = gradient_descent(func, x0, num_iterations, learning_rate)
        
        # 可视化优化轨迹
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["优化轨迹 (2D投影)", "收敛曲线"],
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 轨迹图
        fig.add_trace(
            go.Scatter(
                x=es_history[:, 0], y=es_history[:, 1],
                mode='lines+markers',
                name='ES策略',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=gd_history[:, 0], y=gd_history[:, 1],
                mode='lines+markers',
                name='梯度下降',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # 收敛曲线
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(es_fitness)), y=es_fitness,
                mode='lines',
                name='ES适应度',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(gd_fitness)), y=gd_fitness,
                mode='lines',
                name='GD损失',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"ES vs 梯度下降 - {test_function}",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能指标
        st.markdown("### 📊 性能对比")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ES最终值", f"{es_fitness[-1]:.4f}")
        with col2:
            st.metric("GD最终值", f"{gd_fitness[-1]:.4f}")
        with col3:
            improvement = (gd_fitness[-1] - es_fitness[-1]) / abs(gd_fitness[-1]) * 100
            st.metric("ES改进", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
        
        # 算法特性对比
        st.markdown("### 🔄 算法特性对比")
        
        comparison_data = {
            "特性": ["导数需求", "并行性", "局部最优逃避", "样本效率", "高维适应"],
            "ES策略": ["不需要", "高", "强", "低", "差"],
            "梯度下降": ["必需", "低", "弱", "高", "好"]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        st.info("""
        **关键洞察**：
        - ES在多峰函数上表现更好，因为群体搜索能跳出局部最优
        - 梯度下降在光滑函数上更高效，但容易陷入峡谷或平台
        - ES的并行性使其在大规模计算中具有优势
        - 样本效率是ES的主要瓶颈，特别在数据获取昂贵的场景
        """)
    
    @staticmethod
    def _render_openai_es():
        """OpenAI ES算法演示"""
        st.markdown("### 🚀 OpenAI ES：镜像采样与方差优化")
        
        st.latex(r"""
        \theta_{t+1} = \theta_t + \alpha \cdot \frac{1}{n\sigma} \sum_{i=1}^n (R_i^+ - R_i^-) \epsilon_i
        """)
        
        with st.sidebar:
            population_size = st.slider("种群大小", 10, 100, 32, 2)
            sigma = st.slider("噪声强度", 0.01, 0.5, 0.1, 0.01)
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.02, 0.001)
            use_mirror_sampling = st.checkbox("使用镜像采样", value=True)
            num_iterations = st.slider("迭代次数", 50, 200, 100, 10)
        
        # 目标函数 (简单的二次函数)
        def target_function(theta):
            """简单的二次目标函数，最小化 ||theta - target||^2"""
            target = np.array([1.0, 2.0])
            return -np.sum((theta - target)**2)  # 负号因为我们要最大化
        
        # OpenAI ES实现
        def openai_es(f, theta0, num_iter, pop_size, sigma, lr, mirror=True):
            theta = theta0.copy()
            history = [theta.copy()]
            fitness_history = [f(theta)]
            
            for t in range(num_iter):
                if mirror:
                    # 镜像采样
                    epsilon = np.random.randn(pop_size // 2, len(theta))
                    epsilon_full = np.concatenate([epsilon, -epsilon])
                    
                    # 评估正负扰动
                    rewards_plus = np.array([f(theta + sigma * e) for e in epsilon])
                    rewards_minus = np.array([f(theta - sigma * e) for e in epsilon])
                    
                    # 组合奖励
                    rewards_diff = rewards_plus - rewards_minus
                    rewards_diff = np.concatenate([rewards_diff, -rewards_diff])
                else:
                    # 标准采样
                    epsilon_full = np.random.randn(pop_size, len(theta))
                    rewards = np.array([f(theta + sigma * e) for e in epsilon_full])
                    rewards_diff = rewards
                
                # 标准化奖励
                rewards_diff = (rewards_diff - np.mean(rewards_diff)) / (np.std(rewards_diff) + 1e-8)
                
                # 更新参数
                gradient_estimate = np.dot(epsilon_full.T, rewards_diff) / (pop_size * sigma)
                theta = theta + lr * gradient_estimate
                
                history.append(theta.copy())
                fitness_history.append(f(theta))
            
            return np.array(history), np.array(fitness_history)
        
        # 运行算法
        np.random.seed(42)
        theta0 = np.array([0.0, 0.0])
        
        # 对比镜像采样 vs 标准采样
        history_mirror, fitness_mirror = openai_es(
            target_function, theta0, num_iterations, population_size, sigma, learning_rate, mirror=True
        )
        history_standard, fitness_standard = openai_es(
            target_function, theta0, num_iterations, population_size, sigma, learning_rate, mirror=False
        )
        
        # 可视化结果
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "参数轨迹 (镜像采样)", "参数轨迹 (标准采样)",
                "适应度曲线对比", "方差分析"
            ]
        )
        
        # 镜像采样轨迹
        fig.add_trace(
            go.Scatter(
                x=history_mirror[:, 0], y=history_mirror[:, 1],
                mode='lines+markers',
                name='镜像采样',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 标准采样轨迹
        fig.add_trace(
            go.Scatter(
                x=history_standard[:, 0], y=history_standard[:, 1],
                mode='lines+markers',
                name='标准采样',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # 适应度对比
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(fitness_mirror)), y=fitness_mirror,
                mode='lines',
                name='镜像采样',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(fitness_standard)), y=fitness_standard,
                mode='lines',
                name='标准采样',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # 方差分析
        mirror_var = np.var(fitness_mirror[-20:])  # 最后20步的方差
        standard_var = np.var(fitness_standard[-20:])
        
        fig.add_trace(
            go.Bar(
                x=['镜像采样', '标准采样'],
                y=[mirror_var, standard_var],
                marker_color=['lightblue', 'lightcoral']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="OpenAI ES 算法分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("镜像最终值", f"{fitness_mirror[-1]:.4f}")
        with col2:
            st.metric("标准最终值", f"{fitness_standard[-1]:.4f}")
        with col3:
            st.metric("镜像方差", f"{mirror_var:.6f}")
        with col4:
            st.metric("标准方差", f"{standard_var:.6f}")
        
        # 算法步骤可视化
        st.markdown("### 🔄 单步算法演示")
        
        # 展示单步的采样和评估过程
        step_demo = st.slider("选择演示步骤", 0, min(num_iterations-1, 10), 0)
        
        if step_demo > 0:
            theta_current = history_mirror[step_demo]
            
            # 生成采样点
            epsilon_demo = np.random.randn(population_size // 2, len(theta_current))
            epsilon_full_demo = np.concatenate([epsilon_demo, -epsilon_demo])
            
            # 计算奖励
            rewards_plus_demo = np.array([target_function(theta_current + sigma * e) for e in epsilon_demo])
            rewards_minus_demo = np.array([target_function(theta_current - sigma * e) for e in epsilon_demo])
            
            # 可视化采样分布
            fig_demo = go.Figure()
            
            # 正扰动点
            pos_points = theta_current + sigma * epsilon_demo
            fig_demo.add_trace(go.Scatter(
                x=pos_points[:, 0], y=pos_points[:, 1],
                mode='markers',
                name='正扰动',
                marker=dict(color='green', size=8, opacity=0.7),
                text=[f"R+: {r:.3f}" for r in rewards_plus_demo]
            ))
            
            # 负扰动点
            neg_points = theta_current - sigma * epsilon_demo
            fig_demo.add_trace(go.Scatter(
                x=neg_points[:, 0], y=neg_points[:, 1],
                mode='markers',
                name='负扰动',
                marker=dict(color='red', size=8, opacity=0.7),
                text=[f"R-: {r:.3f}" for r in rewards_minus_demo]
            ))
            
            # 当前参数点
            fig_demo.add_trace(go.Scatter(
                x=[theta_current[0]], y=[theta_current[1]],
                mode='markers',
                name='当前参数',
                marker=dict(color='blue', size=15, symbol='star')
            ))
            
            # 目标点
            target_point = np.array([1.0, 2.0])
            fig_demo.add_trace(go.Scatter(
                x=[target_point[0]], y=[target_point[1]],
                mode='markers',
                name='目标',
                marker=dict(color='gold', size=15, symbol='diamond')
            ))
            
            fig_demo.update_layout(
                title=f"步骤 {step_demo} 采样分布",
                xaxis_title="参数 1",
                yaxis_title="参数 2",
                height=500
            )
            
            st.plotly_chart(fig_demo, use_container_width=True)
        
        st.success("""
        **OpenAI ES 的优势**：
        - 镜像采样显著降低方差，提高收敛稳定性
        - 无需反向传播，计算图简单
        - 天然适合分布式训练
        - 对梯度消失/爆炸不敏感
        """)
    
    @staticmethod
    def _render_pbt():
        """PBT种群训练演示"""
        st.markdown("### 🧬 PBT：基于种群的训练")
        
        st.markdown("""
        **核心思想**：
        - **Exploit (利用)**：表现差的模型复制表现好的模型参数
        - **Explore (探索)**：对继承的超参数进行随机扰动
        - **双层优化**：内层SGD优化权重，外层进化优化超参数
        """)
        
        with st.sidebar:
            population_size = st.slider("种群大小", 4, 16, 8, 2)
            num_generations = st.slider("代数", 10, 50, 20, 5)
            exploit_interval = st.slider("利用间隔", 2, 10, 5, 1)
            mutation_strength = st.slider("变异强度", 0.1, 0.5, 0.2, 0.05)
            initial_lr_range = st.slider("初始学习率范围", 0.001, 0.1, (0.01, 0.05))
        
        # PBT算法实现
        class PBTAgent:
            def __init__(self, agent_id, lr, momentum):
                self.id = agent_id
                self.lr = lr
                self.momentum = momentum
                self.weights = np.random.randn(2) * 0.1
                self.fitness_history = []
                self.age = 0
            
            def train_step(self, target_function, steps=5):
                """模拟几步训练"""
                for _ in range(steps):
                    # 简单的梯度下降步骤
                    grad = self._compute_gradient(target_function)
                    self.weights = self.weights - self.lr * grad
                self.age += 1
            
            def _compute_gradient(self, target_function, eps=1e-6):
                """数值梯度"""
                grad = np.zeros_like(self.weights)
                for i in range(len(self.weights)):
                    w_plus = self.weights.copy()
                    w_minus = self.weights.copy()
                    w_plus[i] += eps
                    w_minus[i] -= eps
                    grad[i] = (target_function(w_plus) - target_function(w_minus)) / (2 * eps)
                return grad
            
            def evaluate(self, target_function):
                """评估当前性能"""
                fitness = target_function(self.weights)
                self.fitness_history.append(fitness)
                return fitness
            
            def copy_from(self, other):
                """复制另一个智能体的参数"""
                self.weights = other.weights.copy()
                self.fitness_history = other.fitness_history.copy()
            
            def mutate_hyperparams(self):
                """变异超参数"""
                self.lr *= np.random.uniform(1 - mutation_strength, 1 + mutation_strength)
                self.lr = np.clip(self.lr, 0.001, 0.1)
                self.momentum *= np.random.uniform(1 - mutation_strength, 1 + mutation_strength)
                self.momentum = np.clip(self.momentum, 0.0, 0.99)
        
        # 目标函数
        def target_function(weights):
            target = np.array([1.0, 2.0])
            return -np.sum((weights - target)**2)
        
        # 初始化种群
        np.random.seed(42)
        population = []
        for i in range(population_size):
            lr = np.random.uniform(initial_lr_range[0], initial_lr_range[1])
            momentum = np.random.uniform(0.5, 0.95)
            population.append(PBTAgent(i, lr, momentum))
        
        # PBT训练循环
        history = {
            'weights': [],
            'lr': [],
            'fitness': [],
            'exploit_events': []
        }
        
        for generation in range(num_generations):
            # 训练每个智能体
            for agent in population:
                agent.train_step(target_function)
            
            # 评估种群
            fitnesses = [agent.evaluate(target_function) for agent in population]
            
            # 记录历史
            history['weights'].append([agent.weights.copy() for agent in population])
            history['lr'].append([agent.lr for agent in population])
            history['fitness'].append(fitnesses)
            
            # PBT利用和探索
            if generation % exploit_interval == 0 and generation > 0:
                # 找到表现最好和最差的智能体
                best_idx = np.argmax(fitnesses)
                worst_idx = np.argmin(fitnesses)
                
                # 利用：最差的复制最好的
                population[worst_idx].copy_from(population[best_idx])
                
                # 探索：变异超参数
                population[worst_idx].mutate_hyperparams()
                
                history['exploit_events'].append({
                    'generation': generation,
                    'best': best_idx,
                    'worst': worst_idx
                })
        
        # 可视化PBT过程
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "种群适应度演化", "学习率演化",
                "参数空间轨迹", "利用/探索事件"
            ]
        )
        
        # 适应度演化
        for i in range(population_size):
            fitness_traj = [gen[i] for gen in history['fitness']]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(fitness_traj)),
                    y=fitness_traj,
                    mode='lines',
                    name=f'智能体 {i}',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 学习率演化
        for i in range(population_size):
            lr_traj = [gen[i] for gen in history['lr']]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(lr_traj)),
                    y=lr_traj,
                    mode='lines',
                    name=f'智能体 {i} LR',
                    line=dict(width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 参数空间轨迹 (只显示前3个智能体避免混乱)
        colors = ['blue', 'red', 'green']
        for i in range(min(3, population_size)):
            weights_traj = np.array([gen[i] for gen in history['weights']])
            fig.add_trace(
                go.Scatter(
                    x=weights_traj[:, 0],
                    y=weights_traj[:, 1],
                    mode='lines+markers',
                    name=f'智能体 {i} 轨迹',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
        
        # 利用/探索事件
        if history['exploit_events']:
            event_gens = [event['generation'] for event in history['exploit_events']]
            event_fitness = [max(history['fitness'][gen]) for gen in event_gens]
            
            fig.add_trace(
                go.Scatter(
                    x=event_gens,
                    y=event_fitness,
                    mode='markers',
                    name='利用/探索事件',
                    marker=dict(color='gold', size=10, symbol='star')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="PBT 种群训练过程",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 最终种群状态
        st.markdown("### 🏆 最终种群状态")
        
        final_fitnesses = history['fitness'][-1]
        final_lrs = history['lr'][-1]
        
        results_df = pd.DataFrame({
            '智能体': [f'Agent {i}' for i in range(population_size)],
            '最终适应度': final_fitnesses,
            '最终学习率': final_lrs,
            '年龄': [agent.age for agent in population]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # PBT优势分析
        st.markdown("### 📈 PBT 优势分析")
        
        # 计算最佳智能体的性能提升
        best_fitness_per_gen = [max(gen) for gen in history['fitness']]
        improvement = (best_fitness_per_gen[-1] - best_fitness_per_gen[0]) / abs(best_fitness_per_gen[0]) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("性能提升", f"{improvement:.1f}%")
        with col2:
            st.metric("利用事件", len(history['exploit_events']))
        with col3:
            avg_lr = np.mean(final_lrs)
            st.metric("平均学习率", f"{avg_lr:.4f}")
        
        st.success("""
        **PBT 的核心价值**：
        - **自动调参**：避免人工搜索超参数的灾难
        - **动态适应**：训练过程中自动调整超参数
        - **双层优化**：同时优化权重和超参数
        - **种群多样性**：保持探索能力，避免过早收敛
        """)
    
    @staticmethod
    def _render_cma_es():
        """CMA-ES协方差适应演示"""
        st.markdown("### 🎯 CMA-ES：协方差矩阵适应进化策略")
        
        st.markdown("""
        **核心突破**：
        - **椭球探索**：学习协方差矩阵，从圆形探索变为椭球探索
        - **参数相关性**：自动学习参数间的相关性
        - **自适应步长**：根据成功概率调整探索强度
        """)
        
        with st.sidebar:
            population_size = st.slider("种群大小", 10, 100, 30, 5)
            num_iterations = st.slider("迭代次数", 20, 100, 50, 5)
            initial_sigma = st.slider("初始步长", 0.1, 2.0, 0.5, 0.1)
            target_condition = st.selectbox("目标函数", 
                ["椭圆山谷", "旋转椭圆", "多峰函数"])
        
        # 定义测试函数
        def elliptical_valley(x):
            """椭圆山谷函数"""
            return 100 * x[0]**2 + x[1]**2
        
        def rotated_elliptical(x):
            """旋转椭圆函数"""
            theta = np.pi / 4  # 45度旋转
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
            x_rotated = rotation @ x
            return 100 * x_rotated[0]**2 + x_rotated[1]**2
        
        def multimodal_function(x):
            """多峰函数"""
            return (x[0]**2 + x[1]**2) * np.sin(5 * np.sqrt(x[0]**2 + x[1]**2))
        
        # 选择目标函数
        if target_condition == "椭圆山谷":
            target_func = elliptical_valley
            bounds = (-2, 2)
        elif target_condition == "旋转椭圆":
            target_func = rotated_elliptical
            bounds = (-2, 2)
        else:  # 多峰函数
            target_func = multimodal_function
            bounds = (-3, 3)
        
        # 简化的CMA-ES实现
        class CMAES:
            def __init__(self, dimension, initial_mean, initial_sigma, population_size):
                self.dimension = dimension
                self.mean = initial_mean.copy()
                self.sigma = initial_sigma
                self.population_size = population_size
                self.covariance = np.eye(dimension)
                self.evolution_path = np.zeros(dimension)
                
                # CMA-ES参数
                self.cc = 4 / (dimension + 4)
                self.cs = 2 / (dimension + 2)
                self.c1 = 2 / ((dimension + 1.3)**2 + 2)
                self.cmu = min(1 - self.c1, 2 * (2/17) / (dimension**2 + 2))
                self.damps = 1 + 2 * max(0, np.sqrt((population_size - 1) / (dimension + 1)) - 1) + self.cs
            
            def sample(self):
                """从当前分布采样"""
                samples = []
                # 确保协方差矩阵对称正定
                cov_sym = (self.covariance + self.covariance.T) / 2
                eigenvals, eigenvecs = np.linalg.eigh(cov_sym)
                eigenvals = np.maximum(eigenvals, 1e-8)  # 确保特征值正
                cov_stable = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                for _ in range(self.population_size):
                    # 使用Cholesky分解进行稳定采样
                    try:
                        L = np.linalg.cholesky(cov_stable)
                        z = L @ np.random.randn(self.dimension)
                    except np.linalg.LinAlgError:
                        # 如果Cholesky失败，使用特征分解方法
                        z = eigenvecs @ (np.sqrt(eigenvals) * np.random.randn(self.dimension))
                    
                    x = self.mean + self.sigma * z
                    samples.append(x)
                return np.array(samples)
            
            def update(self, samples, fitness_values):
                """更新分布参数"""
                # 选择最好的个体
                selected_indices = np.argsort(fitness_values)[:self.population_size//2]
                selected_samples = samples[selected_indices]
                
                # 计算新均值
                old_mean = self.mean.copy()
                self.mean = np.mean(selected_samples, axis=0)
                
                # 更新进化路径
                self.evolution_path = (1 - self.cc) * self.evolution_path + \
                                    np.sqrt(self.cc * (2 - self.cc) * self.population_size) * \
                                    (self.mean - old_mean) / self.sigma
                
                # 更新协方差矩阵
                y = selected_samples - old_mean
                rank_one_update = np.outer(self.evolution_path, self.evolution_path)
                
                # 手动计算协方差，避免np.cov的数值问题
                if y.shape[0] > 1:
                    rank_mu_update = np.dot(y.T, y) / (y.shape[0] - 1)
                    # 检查计算结果
                    if not np.all(np.isfinite(rank_mu_update)):
                        print("警告：rank_mu_update包含异常值，使用默认值")
                        rank_mu_update = np.eye(self.dimension) * 0.01
                else:
                    rank_mu_update = np.eye(self.dimension) * 0.01
                
                self.covariance = (1 - self.c1 - self.cmu) * self.covariance + \
                                self.c1 * rank_one_update + \
                                self.cmu * rank_mu_update
                
                # 强制数值稳定性检查
                if not np.all(np.isfinite(self.covariance)):
                    print("警告：协方差矩阵包含无穷大或NaN，重置为单位矩阵")
                    self.covariance = np.eye(self.dimension)
                
                # 确保协方差矩阵对称正定
                self.covariance = (self.covariance + self.covariance.T) / 2
                
                # 检查特征值并修正
                try:
                    eigenvals = np.linalg.eigvals(self.covariance)
                    if np.min(eigenvals) < 1e-8 or not np.all(np.isfinite(eigenvals)):
                        self.covariance += 1e-6 * np.eye(self.dimension)
                except np.linalg.LinAlgError:
                    print("警告：特征值分解失败，重置协方差矩阵")
                    self.covariance = np.eye(self.dimension)
                
                # 更新步长（带数值检查）
                path_norm = np.linalg.norm(self.evolution_path)
                if np.isfinite(path_norm) and path_norm > 0:
                    update_factor = self.cs / self.damps * (path_norm / np.sqrt(self.dimension) - 1)
                    if np.isfinite(update_factor):
                        self.sigma *= np.exp(update_factor)
                        # 限制步长范围
                        self.sigma = np.clip(self.sigma, 1e-8, 10.0)
                    else:
                        print("警告：步长更新因子异常，跳过更新")
                else:
                    print("警告：进化路径范数异常，跳过步长更新")
        
        # 运行CMA-ES
        np.random.seed(42)
        dimension = 2
        initial_mean = np.array([1.5, 1.5])
        
        cma = CMAES(dimension, initial_mean, initial_sigma, population_size)
        
        history = {
            'mean': [initial_mean.copy()],
            'covariance': [np.eye(dimension)],
            'sigma': [initial_sigma],
            'fitness': [target_func(initial_mean)]
        }
        
        for iteration in range(num_iterations):
            # 采样
            samples = cma.sample()
            
            # 评估
            fitness_values = np.array([target_func(sample) for sample in samples])
            
            # 更新
            cma.update(samples, fitness_values)
            
            # 记录历史
            history['mean'].append(cma.mean.copy())
            history['covariance'].append(cma.covariance.copy())
            history['sigma'].append(cma.sigma)
            history['fitness'].append(np.min(fitness_values))
        
        # 可视化CMA-ES过程
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "均值轨迹与协方差椭圆", "步长演化",
                "适应度收敛", "特征值分析"
            ]
        )
        
        # 均值轨迹
        means = np.array(history['mean'])
        fig.add_trace(
            go.Scatter(
                x=means[:, 0], y=means[:, 1],
                mode='lines+markers',
                name='均值轨迹',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # 绘制协方差椭圆 (每5步显示一个)
        for i in range(0, len(history['covariance']), 5):
            cov = history['covariance'][i]
            mean = history['mean'][i]
            
            # 生成椭圆点
            theta = np.linspace(0, 2*np.pi, 50)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            
            ellipse_points = []
            for t in theta:
                point = mean + 2 * history['sigma'][i] * (eigenvecs @ np.sqrt(eigenvals) * np.array([np.cos(t), np.sin(t)]))
                ellipse_points.append(point)
            
            ellipse_points = np.array(ellipse_points)
            
            fig.add_trace(
                go.Scatter(
                    x=ellipse_points[:, 0],
                    y=ellipse_points[:, 1],
                    mode='lines',
                    name=f'协方差椭圆 步骤{i}',
                    line=dict(width=1, color='gray'),
                    opacity=0.5,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 步长演化
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(history['sigma'])),
                y=history['sigma'],
                mode='lines',
                name='步长',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # 适应度收敛
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(history['fitness'])),
                y=history['fitness'],
                mode='lines',
                name='最佳适应度',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 特征值分析
        eigenvalues_history = []
        for cov in history['covariance']:
            eigenvals = np.linalg.eigvals(cov)
            eigenvalues_history.append(sorted(eigenvals, reverse=True))
        
        eigenvalues_history = np.array(eigenvalues_history)
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(eigenvalues_history)),
                y=eigenvalues_history[:, 0],
                mode='lines',
                name='最大特征值',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(eigenvalues_history)),
                y=eigenvalues_history[:, 1],
                mode='lines',
                name='最小特征值',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="CMA-ES 协方差适应过程",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 最终状态分析
        st.markdown("### 📊 最终状态分析")
        
        final_mean = history['mean'][-1]
        final_cov = history['covariance'][-1]
        final_sigma = history['sigma'][-1]
        final_fitness = history['fitness'][-1]
        
        eigenvals, eigenvecs = np.linalg.eig(final_cov)
        condition_number = max(eigenvals) / min(eigenvals)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最终适应度", f"{final_fitness:.4f}")
        with col2:
            st.metric("最终步长", f"{final_sigma:.4f}")
        with col3:
            st.metric("条件数", f"{condition_number:.2f}")
        with col4:
            st.metric("迭代次数", num_iterations)
        
        # 协方差矩阵可视化
        st.markdown("### 🔄 协方差矩阵演化")
        
        fig_cov = go.Figure()
        
        # 热力图显示最终协方差矩阵
        fig_cov.add_trace(
            go.Heatmap(
                z=final_cov,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="协方差值")
            )
        )
        
        fig_cov.update_layout(
            title="最终协方差矩阵",
            xaxis_title="参数维度",
            yaxis_title="参数维度",
            height=400
        )
        
        st.plotly_chart(fig_cov, use_container_width=True)
        
        st.success("""
        **CMA-ES 的核心优势**：
        - **椭球探索**：自适应协方差矩阵实现椭球探索
        - **参数相关性**：自动学习并利用参数间的相关性
        - **自适应性**：步长和形状自动调整，无需手动调参
        - **数学严谨性**：基于自然进化策略的严格数学推导
        """)


# 为了兼容性，添加缺少的导入
try:
    from scipy.stats import multivariate_normal
except ImportError:
    # 如果scipy不可用，使用numpy实现
    def multivariate_normal(mean, cov):
        class MVN:
            def rvs(self, size=1):
                return np.random.multivariate_normal(mean, cov, size)
        return MVN()

        # 添加交互式测验
