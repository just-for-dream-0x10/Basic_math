"""
交互式马尔可夫决策过程(MDP)可视化
严格按照 16.MDP.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

warnings.filterwarnings('ignore')


class InteractiveMDP:
    """交互式马尔可夫决策过程可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎯 马尔可夫决策过程与贝尔曼方程")
        st.markdown(r"""
        **核心思想**: 从预测到决策的范式转移，智能体通过动作改变环境并最大化累积奖励
        
        关键概念：
        - **五元组**: $M = \langle S, A, P, R, \gamma \rangle$
        - **贝尔曼方程**: $V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma V^*(s')]$
        - **Q-Learning**: $Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
        - **策略梯度**: $\nabla_{\theta} J(\theta) = \mathbb{E}[\sum_{t} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot G_t]$
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["MDP基础概念", "价值迭代算法", "Q-Learning", "策略梯度"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "MDP基础概念":
            InteractiveMDP._render_mdp_basics()
        elif viz_type == "价值迭代算法":
            InteractiveMDP._render_value_iteration()
        elif viz_type == "Q-Learning":
            InteractiveMDP._render_q_learning()
        elif viz_type == "策略梯度":
            InteractiveMDP._render_policy_gradient()
    

        # 添加交互式测验
        quiz_system = QuizSystem("mdp")
        quizzes = QuizTemplates.get_mdp_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_mdp_basics():
        """MDP基础概念演示"""
        st.markdown("### 🌍 MDP五元组：世界观的数学建模")
        
        st.latex(r"""
        M = \langle S, A, P, R, \gamma \rangle
        """)
        
        with st.sidebar:
            grid_size = st.slider("网格世界大小", 3, 8, 5, 1)
            gamma = st.slider("折扣因子 γ", 0.0, 1.0, 0.9, 0.05)
            show_probabilities = st.checkbox("显示转移概率", value=True)
            show_rewards = st.checkbox("显示奖励值", value=True)
        
        # 创建网格世界
        np.random.seed(42)
        grid = np.zeros((grid_size, grid_size))
        
        # 随机放置特殊格子
        # 终点（奖励+10）
        grid[grid_size-1, grid_size-1] = 10
        # 陷阱（奖励-10）
        trap_positions = np.random.choice(grid_size*grid_size-2, 2, replace=False)
        for pos in trap_positions:
            r, c = pos // grid_size, pos % grid_size
            if r != 0 or c != 0:  # 不在起点
                grid[r, c] = -10
        # 墙壁（不可通过）
        wall_positions = np.random.choice(grid_size*grid_size-4, 3, replace=False)
        for pos in wall_positions:
            r, c = pos // grid_size, pos % grid_size
            if r != 0 or c != 0 and (r != grid_size-1 or c != grid_size-1):
                grid[r, c] = -1
        
        # 定义动作
        actions = ['上', '下', '左', '右']
        action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 计算转移概率和奖励
        def get_transition_and_reward(state, action):
            r, c = state
            dr, dc = action_deltas[action]
            new_r, new_c = r + dr, c + dc
            
            # 检查边界和墙壁
            if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                grid[new_r, new_c] != -1):
                return (new_r, new_c), grid[new_r, new_c]
            else:
                # 撞墙或出界，留在原地
                return (r, c), -1
        
        # 可视化网格世界
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["网格世界", "转移概率示例"],
            specs=[[{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 网格世界热力图
        grid_display = grid.copy()
        grid_display[grid_display == -1] = 0  # 墙壁显示为灰色
        
        fig.add_trace(
            go.Heatmap(
                z=grid_display,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="奖励值"),
                text=np.array([['墙壁' if x == -1 else f'{x:.0f}' if x != 0 else '' 
                              for x in row] for row in grid]),
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=1, col=1
        )
        
        # 转移概率示例（从中心点）
        center_state = (grid_size // 2, grid_size // 2)
        transition_probs = []
        action_labels = []
        
        for i, (action, (dr, dc)) in enumerate(zip(actions, action_deltas)):
            new_state, reward = get_transition_and_reward(center_state, i)
            # 简化的转移概率（实际应该更复杂）
            prob = 0.8 if grid[new_state[0], new_state[1]] != -1 else 0.0
            
            if show_probabilities:
                transition_probs.append(prob)
                action_labels.append(f'{action}\\nP={prob:.1f}')
        
        if show_probabilities:
            fig.add_trace(
                go.Bar(
                    x=action_labels,
                    y=transition_probs,
                    marker_color='lightblue',
                    name='转移概率'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"MDP网格世界 (γ={gamma})",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 马尔可夫性质演示
        st.markdown("### 🔄 马尔可夫性质")
        
        st.markdown("""
        **核心假设**: 未来只取决于现在，与过去无关
        
        $P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)$
        
        这意味着状态 $S_t$ 必须包含决策所需的所有信息。
        """)
        
        # 状态序列演示
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**满足马尔可夫性质**")
            st.write("✅ 国际象棋：当前棋局包含所有信息")
            st.write("✅ 导航：当前位置和目标位置")
            st.write("✅ 金融：当前投资组合")
        
        with col2:
            st.markdown("**不满足马尔可夫性质**")
            st.write("❌ POMDP：部分可观测系统")
            st.write("❊ 需要历史信息才能决策")
            st.write("❊ 卡牌游戏：不知道对手手牌")
        
        # 折扣因子的影响
        st.markdown("### ⏰ 折扣因子的影响")
        
        gamma_values = [0.0, 0.5, 0.9, 0.99]
        future_values = []
        
        for g in gamma_values:
            # 计算未来10步的折现奖励
            discounted_sum = sum([g**i for i in range(10)])
            future_values.append(discounted_sum)
        
        fig_gamma = go.Figure()
        fig_gamma.add_trace(
            go.Scatter(
                x=gamma_values,
                y=future_values,
                mode='lines+markers',
                name='折现总和',
                line=dict(width=3),
                marker=dict(size=8)
            )
        )
        
        fig_gamma.update_layout(
            title="折扣因子对未来奖励的影响",
            xaxis_title="折扣因子 γ",
            yaxis_title="未来10步折现奖励总和",
            height=400
        )
        
        st.plotly_chart(fig_gamma, use_container_width=True)
        
        st.info("""
        **折扣因子的哲学含义**：
        - γ = 0：完全短视，只看当前奖励
        - γ = 0.5：平衡当前和未来
        - γ = 0.9：重视长期收益
        - γ → 1：极度远见，考虑千秋万代
        """)
    
    @staticmethod
    def _render_value_iteration():
        """价值迭代算法演示"""
        st.markdown("### 🧠 价值迭代：贝尔曼方程的数值求解")
        
        st.latex(r"""
        V_{k+1}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma V_k(s')]
        """)
        
        with st.sidebar:
            grid_size = st.slider("网格大小", 4, 8, 5, 1)
            gamma = st.slider("折扣因子", 0.5, 0.99, 0.9, 0.01)
            max_iterations = st.slider("最大迭代次数", 10, 100, 50, 5)
            convergence_threshold = st.slider("收敛阈值", 0.001, 0.1, 0.01, 0.001)
            show_convergence = st.checkbox("显示收敛过程", value=True)
        
        # 创建网格世界
        grid = np.zeros((grid_size, grid_size))
        grid[grid_size-1, grid_size-1] = 10  # 终点
        grid[1, 1] = -10  # 陷阱
        grid[2, 2] = -10  # 陷阱
        
        # 墙壁
        walls = [(1, 2), (3, 1)]
        for wall in walls:
            if wall[0] < grid_size and wall[1] < grid_size:
                grid[wall[0], wall[1]] = -1
        
        # 价值迭代算法
        def value_iteration(grid, gamma, max_iter, threshold):
            rows, cols = grid.shape
            V = np.zeros((rows, cols))
            policy = np.zeros((rows, cols), dtype=int)
            history = []
            
            actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
            action_names = ['上', '下', '左', '右']
            
            for iteration in range(max_iter):
                V_new = np.copy(V)
                max_change = 0
                
                for i in range(rows):
                    for j in range(cols):
                        if grid[i, j] == -1:  # 墙壁
                            continue
                        
                        if i == rows-1 and j == cols-1:  # 终点
                            V_new[i, j] = 0
                            continue
                        
                        best_value = float('-inf')
                        best_action = 0
                        
                        for action_idx, (di, dj) in enumerate(actions):
                            ni, nj = i + di, j + dj
                            
                            # 检查边界和墙壁
                            if (0 <= ni < rows and 0 <= nj < cols and 
                                grid[ni, nj] != -1):
                                reward = grid[ni, nj]
                                value = reward + gamma * V[ni, nj]
                            else:
                                # 撞墙
                                value = -1 + gamma * V[i, j]
                            
                            if value > best_value:
                                best_value = value
                                best_action = action_idx
                        
                        V_new[i, j] = best_value
                        policy[i, j] = best_action
                        max_change = max(max_change, abs(V_new[i, j] - V[i, j]))
                
                history.append(np.copy(V_new))
                V = V_new
                
                if max_change < threshold:
                    break
            
            return V, policy, history
        
        # 运行价值迭代
        V, policy, history = value_iteration(grid, gamma, max_iterations, convergence_threshold)
        
        # 可视化结果
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "最终价值函数", "最优策略", 
                "收敛过程", "价值函数演化"
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 最终价值函数
        V_display = V.copy()
        V_display[grid == -1] = np.nan  # 墙壁显示为空
        
        fig.add_trace(
            go.Heatmap(
                z=V_display,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="价值"),
                text=np.array([[f'{V[i,j]:.2f}' if not np.isnan(V[i,j]) else '' 
                              for j in range(grid_size)] for i in range(grid_size)]),
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=1, col=1
        )
        
        # 最优策略
        policy_display = policy.copy()
        policy_display[grid == -1] = -1  # 墙壁
        
        action_symbols = ['↑', '↓', '←', '→', '█']
        policy_text = np.array([[action_symbols[policy_display[i,j]] 
                                for j in range(grid_size)] for i in range(grid_size)])
        
        fig.add_trace(
            go.Heatmap(
                z=policy_display,
                colorscale='RdYlBu',
                showscale=False,
                text=policy_text,
                texttemplate="%{text}",
                textfont={"size": 16}
            ),
            row=1, col=2
        )
        
        # 收敛过程
        if show_convergence and len(history) > 1:
            changes = [np.max(np.abs(history[i] - history[i-1])) 
                      for i in range(1, len(history))]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(changes)+1)),
                    y=changes,
                    mode='lines+markers',
                    name='最大变化',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
            
            # 添加收敛线
            fig.add_hline(
                y=convergence_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"阈值: {convergence_threshold}"
            )
        
        # 价值函数演化（选择几个关键状态）
        if len(history) > 1:
            # 选择起点(0,0)和终点附近的状态
            start_values = [hist[0, 0] for hist in history]
            near_goal_values = [hist[grid_size-2, grid_size-2] for hist in history]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(start_values))),
                    y=start_values,
                    mode='lines',
                    name='起点 (0,0)',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(near_goal_values))),
                    y=near_goal_values,
                    mode='lines',
                    name='近终点',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="价值迭代算法分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 算法性能分析
        st.markdown("### 📊 算法性能分析")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最终迭代次数", len(history))
        with col2:
            final_change = np.max(np.abs(history[-1] - history[-2])) if len(history) > 1 else 0
            st.metric("最终变化", f"{final_change:.4f}")
        with col3:
            max_value = np.nanmax(V_display)
            st.metric("最大价值", f"{max_value:.2f}")
        with col4:
            min_value = np.nanmin(V_display)
            st.metric("最小价值", f"{min_value:.2f}")
        
        st.success("""
        **价值迭代的数学保证**：
        - **巴拿赫不动点定理**：贝尔曼算子是压缩映射
        - **收敛性**：γ < 1 时必然收敛到唯一解
        - **最优性**：收敛得到的策略是最优策略
        - **计算复杂度**：O(|S|²|A|)，适用于中小规模问题
        """)
    
    @staticmethod
    def _render_q_learning():
        """Q-Learning算法演示"""
        st.markdown("### 🎮 Q-Learning：无模型的强化学习")
        
        st.latex(r"""
        Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]
        """)
        
        with st.sidebar:
            grid_size = st.slider("网格大小", 4, 6, 4, 1)
            alpha = st.slider("学习率 α", 0.1, 1.0, 0.5, 0.1)
            gamma = st.slider("折扣因子 γ", 0.5, 0.99, 0.9, 0.05)
            epsilon = st.slider("探索率 ε", 0.1, 1.0, 0.3, 0.1)
            episodes = st.slider("训练回合数", 100, 2000, 1000, 100)
            show_q_table = st.checkbox("显示Q表演化", value=True)
        
        # 创建简单的网格世界
        grid = np.zeros((grid_size, grid_size))
        grid[grid_size-1, grid_size-1] = 10  # 终点
        grid[1, 1] = -5  # 小陷阱
        
        # Q-Learning实现
        def q_learning(grid, alpha, gamma, epsilon, episodes):
            rows, cols = grid.shape
            Q = np.zeros((rows, cols, 4))  # 4个动作
            rewards_history = []
            steps_history = []
            
            actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            action_names = ['上', '下', '左', '右']
            
            for episode in range(episodes):
                state = (0, 0)  # 从起点开始
                total_reward = 0
                steps = 0
                max_steps = rows * cols * 2  # 防止无限循环
                
                while steps < max_steps:
                    # ε-贪婪策略选择动作
                    if np.random.random() < epsilon:
                        action = np.random.randint(4)
                    else:
                        action = np.argmax(Q[state[0], state[1]])
                    
                    # 执行动作
                    di, dj = actions[action]
                    new_state = (state[0] + di, state[1] + dj)
                    
                    # 检查边界
                    if (0 <= new_state[0] < rows and 0 <= new_state[1] < cols):
                        reward = grid[new_state[0], new_state[1]]
                    else:
                        new_state = state  # 撞墙
                        reward = -1
                    
                    # Q-Learning更新
                    old_q = Q[state[0], state[1], action]
                    next_max_q = np.max(Q[new_state[0], new_state[1]])
                    td_error = reward + gamma * next_max_q - old_q
                    Q[state[0], state[1], action] = old_q + alpha * td_error
                    
                    total_reward += reward
                    steps += 1
                    state = new_state
                    
                    # 到达终点
                    if state == (rows-1, cols-1):
                        break
                
                rewards_history.append(total_reward)
                steps_history.append(steps)
                
                # 衰减探索率
                epsilon = max(0.01, epsilon * 0.995)
            
            return Q, rewards_history, steps_history
        
        # 运行Q-Learning
        Q, rewards, steps = q_learning(grid, alpha, gamma, epsilon, episodes)
        
        # 可视化结果
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "学习曲线", "步数变化",
                "最终Q表", "最优策略"
            ]
        )
        
        # 学习曲线（滑动平均）
        window = min(50, episodes // 10)
        if window > 1:
            rewards_smooth = pd.Series(rewards).rolling(window).mean().values
        else:
            rewards_smooth = rewards
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards_smooth))),
                y=rewards_smooth,
                mode='lines',
                name='平均奖励',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # 步数变化
        steps_smooth = pd.Series(steps).rolling(window).mean().values if window > 1 else steps
        fig.add_trace(
            go.Scatter(
                x=list(range(len(steps_smooth))),
                y=steps_smooth,
                mode='lines',
                name='平均步数',
                line=dict(width=2, color='orange')
            ),
            row=1, col=2
        )
        
        # 最终Q表热力图（选择最优动作的价值）
        Q_max = np.max(Q, axis=2)
        fig.add_trace(
            go.Heatmap(
                z=Q_max,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="最大Q值"),
                text=np.array([[f'{Q_max[i,j]:.1f}' for j in range(grid_size)] 
                              for i in range(grid_size)]),
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=2, col=1
        )
        
        # 最优策略
        policy = np.argmax(Q, axis=2)
        action_symbols = ['↑', '↓', '←', '→']
        policy_text = np.array([[action_symbols[policy[i,j]] 
                                for j in range(grid_size)] for i in range(grid_size)])
        
        fig.add_trace(
            go.Heatmap(
                z=policy,
                colorscale='RdYlBu',
                showscale=False,
                text=policy_text,
                texttemplate="%{text}",
                textfont={"size": 16}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Q-Learning算法分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能统计
        st.markdown("### 📈 学习性能统计")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            st.metric("最终平均奖励", f"{final_reward:.2f}")
        with col2:
            final_steps = np.mean(steps[-100:]) if len(steps) >= 100 else np.mean(steps)
            st.metric("最终平均步数", f"{final_steps:.1f}")
        with col3:
            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
            st.metric("成功率", f"{success_rate:.1%}")
        with col4:
            convergence_episode = next((i for i, r in enumerate(rewards_smooth) 
                                      if r > 0 and i > 100), len(rewards))
            st.metric("收敛回合", f"{convergence_episode}")
        
        st.success("""
        **Q-Learning的核心优势**：
        - **无模型**：不需要知道环境转移概率
        - **离策略**：可以从历史经验中学习
        - **收敛保证**：在适当条件下收敛到最优Q函数
        - **实用性强**：是DQN等深度强化学习算法的基础
        """)
    
    @staticmethod
    def _render_policy_gradient():
        """策略梯度算法演示"""
        st.markdown("### 🎯 策略梯度：直接优化策略")
        
        st.latex(r"""
        \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot G_t\right]
        """)
        
        with st.sidebar:
            num_states = st.slider("状态数量", 3, 10, 5, 1)
            num_actions = st.slider("动作数量", 2, 5, 3, 1)
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            episodes = st.slider("训练回合数", 500, 5000, 2000, 500)
            temperature = st.slider("温度参数", 0.1, 2.0, 1.0, 0.1)
            show_policy_evolution = st.checkbox("显示策略演化", value=True)
        
        # 简化的策略梯度实现
        class PolicyGradient:
            def __init__(self, num_states, num_actions, learning_rate, temperature):
                self.num_states = num_states
                self.num_actions = num_actions
                self.lr = learning_rate
                self.temperature = temperature
                
                # 策略参数
                self.theta = np.random.randn(num_states, num_actions) * 0.1
                
            def policy(self, state):
                """Softmax策略"""
                logits = self.theta[state] / self.temperature
                exp_logits = np.exp(logits - np.max(logits))
                return exp_logits / np.sum(exp_logits)
            
            def sample_action(self, state):
                """根据策略采样动作"""
                probs = self.policy(state)
                return np.random.choice(self.num_actions, p=probs)
            
            def update(self, states, actions, rewards):
                """策略梯度更新"""
                for state, action, reward in zip(states, actions, rewards):
                    # 计算梯度
                    probs = self.policy(state)
                    grad = np.zeros(self.num_actions)
                    
                    for a in range(self.num_actions):
                        if a == action:
                            grad[a] = (1 - probs[a]) * reward
                        else:
                            grad[a] = -probs[a] * reward
                    
                    # 更新参数
                    self.theta[state] += self.lr * grad
        
        # 创建简单的环境
        def create_environment(num_states):
            # 线性环境：从状态0到目标状态num_states-1
            rewards = {}
            for s in range(num_states):
                for a in range(3):  # 3个动作：前进、后退、不动
                    if a == 0:  # 前进
                        next_s = min(s + 1, num_states - 1)
                        rewards[(s, a, next_s)] = 1.0 if next_s == num_states - 1 else -0.1
                    elif a == 1:  # 后退
                        next_s = max(s - 1, 0)
                        rewards[(s, a, next_s)] = -0.1
                    else:  # 不动
                        rewards[(s, a, s)] = -0.05
            
            return rewards
        
        # 训练策略梯度
        env_rewards = create_environment(num_states)
        agent = PolicyGradient(num_states, 3, learning_rate, temperature)
        
        episode_rewards = []
        policy_history = []
        
        for episode in range(episodes):
            state = 0
            states, actions, rewards = [], [], []
            total_reward = 0
            
            # 一个回合
            for step in range(num_states * 2):
                action = agent.sample_action(state)
                next_state = min(max(state + action - 1, 0), num_states - 1)
                reward = env_rewards.get((state, action, next_state), -0.1)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                total_reward += reward
                state = next_state
                
                if state == num_states - 1:  # 到达目标
                    break
            
            # 更新策略
            agent.update(states, actions, rewards)
            episode_rewards.append(total_reward)
            
            # 记录策略演化
            if episode % 100 == 0:
                policy_snapshot = np.array([agent.policy(s) for s in range(num_states)])
                policy_history.append(policy_snapshot)
        
        # 可视化结果
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "学习曲线", "最终策略分布",
                "策略演化", "动作概率变化"
            ]
        )
        
        # 学习曲线
        window = min(100, episodes // 10)
        if window > 1:
            rewards_smooth = pd.Series(episode_rewards).rolling(window).mean().values
        else:
            rewards_smooth = episode_rewards
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rewards_smooth))),
                y=rewards_smooth,
                mode='lines',
                name='平均奖励',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # 最终策略分布
        final_policy = np.array([agent.policy(s) for s in range(num_states)])
        action_names = ['后退', '前进', '不动']
        
        for action in range(3):
            fig.add_trace(
                go.Bar(
                    x=list(range(num_states)),
                    y=final_policy[:, action],
                    name=action_names[action],
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 策略演化（选择几个状态）
        if show_policy_evolution and policy_history:
            for state_idx in [0, num_states//2, num_states-1]:
                evolution = [policy_hist[state_idx, 1] for policy_hist in policy_history]  # 前进动作
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(evolution))),
                        y=evolution,
                        mode='lines',
                        name=f'状态{state_idx}前进概率',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        # 动作概率随时间变化
        if len(policy_history) > 1:
            start_policy = policy_history[0]
            end_policy = policy_history[-1]
            
            x_pos = np.arange(num_states)
            width = 0.35
            
            for action in range(3):
                fig.add_trace(
                    go.Bar(
                        x=x_pos - width/2,
                        y=start_policy[:, action],
                        name=f'初始{action_names[action]}',
                        width=width,
                        opacity=0.7
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=x_pos + width/2,
                        y=end_policy[:, action],
                        name=f'最终{action_names[action]}',
                        width=width,
                        opacity=0.7
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="策略梯度算法分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 算法分析
        st.markdown("### 📊 算法分析")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_performance = np.mean(episode_rewards[-100:])
            st.metric("最终性能", f"{final_performance:.3f}")
        with col2:
            improvement = episode_rewards[-1] - episode_rewards[0]
            st.metric("性能提升", f"{improvement:.3f}")
        with col3:
            convergence_point = next((i for i, r in enumerate(rewards_smooth) 
                                    if r > 0 and i > 100), len(rewards_smooth))
            st.metric("收敛点", f"{convergence_point}")
        with col4:
            final_entropy = -np.sum(final_policy * np.log(final_policy + 1e-8), axis=1).mean()
            st.metric("策略熵", f"{final_entropy:.3f}")
        
        st.info("""
        **策略梯度的特点**：
        - **直接优化**：直接优化策略参数，不需要价值函数
        - **连续动作**：天然支持连续动作空间
        - **随机策略**：可以学习随机策略，适合探索
        - **高方差**：通常比价值方法方差更高，需要更多样本
        """)

        # 添加交互式测验
