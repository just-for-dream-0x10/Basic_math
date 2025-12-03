"""
交互式博弈论可视化
严格按照 23.GameTheory.md 中的理论实现

核心内容：
1. 纳什均衡基础
2. 极小极大优化与旋转动力学
3. 雅可比矩阵分析
4. Stackelberg博弈
5. LOLA算法
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.linalg import eig


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation

class InteractiveGameTheory:
    """交互式博弈论可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎮 博弈论：从静态优化到动态均衡")
        
        st.markdown("""
        **核心思想**: 当优化目标取决于对手策略时，极小值点变为鞍点，需要动力学分析
        
        **关键概念**:
        - **纳什均衡**: 没有玩家有单方面偏离的动机
        - **极小极大**: $\\min_x \\max_y f(x,y)$
        - **雅可比分析**: 特征值决定系统稳定性
        - **动力学修正**: 从纯旋转到螺旋收敛
        """)
        
        # 侧边栏选择
        with st.sidebar:
            st.markdown("### 📊 选择可视化")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "纳什均衡基础",
                    "极小极大动力学",
                    "雅可比矩阵分析",
                    "Stackelberg博弈",
                    "LOLA算法"
                ]
            )
        
        # 渲染对应的可视化
        if demo_type == "纳什均衡基础":
            InteractiveGameTheory._render_nash_equilibrium()
        elif demo_type == "极小极大动力学":
            InteractiveGameTheory._render_minmax_dynamics()
        elif demo_type == "雅可比矩阵分析":
            InteractiveGameTheory._render_jacobian_analysis()
        elif demo_type == "Stackelberg博弈":
            InteractiveGameTheory._render_stackelberg()
        elif demo_type == "LOLA算法":
            InteractiveGameTheory._render_lola()
    

        # 添加交互式测验
        quiz_system = QuizSystem("game_theory")
        quizzes = QuizTemplates.get_game_theory_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_nash_equilibrium():
        """纳什均衡基础演示"""
        st.markdown("### 🎯 纳什均衡：从优化到博弈")
        
        st.markdown(r"""
        **定义**: 策略组合 $(\theta_1^*, \theta_2^*)$ 是纳什均衡，如果：
        """)
        
        st.latex(r"""
        \begin{cases}
        \theta_1^* = \arg\min_{\theta_1} L_1(\theta_1, \theta_2^*) \\
        \theta_2^* = \arg\min_{\theta_2} L_2(\theta_1^*, \theta_2)
        \end{cases}
        """)
        
        st.markdown(r"""
        **存在性 (Brouwer不动点定理)**: 如果策略空间是紧致凸集，收益函数连续，则纳什均衡一定存在。
        
        **直观理解**: 就像把一张揉皱的纸扔回桌上，总有一个点在垂直方向上没有位移。
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            game_type = st.selectbox(
                "博弈类型",
                ["囚徒困境", "性别大战", "猎鹿博弈", "剪刀石头布"]
            )
            show_analysis = st.checkbox("显示详细分析", value=True)
        
        # 定义收益矩阵
        payoff_data = InteractiveGameTheory._get_game_payoffs(game_type)
        payoff_p1 = payoff_data['p1']
        payoff_p2 = payoff_data['p2']
        strategies = payoff_data['strategies']
        nash_eq = payoff_data['nash_eq']
        description = payoff_data['description']
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "玩家1收益矩阵",
                "玩家2收益矩阵",
                "收益对比",
                "纳什均衡分析"
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 玩家1收益矩阵
        fig.add_trace(
            go.Heatmap(
                z=payoff_p1,
                x=strategies,
                y=strategies,
                colorscale='RdYlGn',
                text=payoff_p1,
                texttemplate='%{text}',
                textfont={"size": 14},
                showscale=False
            ),
            row=1, col=1
        )
        
        # 玩家2收益矩阵
        fig.add_trace(
            go.Heatmap(
                z=payoff_p2,
                x=strategies,
                y=strategies,
                colorscale='RdYlGn',
                text=payoff_p2,
                texttemplate='%{text}',
                textfont={"size": 14},
                showscale=False
            ),
            row=1, col=2
        )
        
        # 收益对比柱状图
        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                fig.add_trace(
                    go.Bar(
                        x=[f"{s1},{s2}"],
                        y=[payoff_p1[i, j]],
                        name=f"P1",
                        marker_color='blue',
                        showlegend=(i == 0 and j == 0)
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=[f"{s1},{s2}"],
                        y=[payoff_p2[i, j]],
                        name=f"P2",
                        marker_color='red',
                        showlegend=(i == 0 and j == 0)
                    ),
                    row=2, col=1
                )
        
        # 纳什均衡标注
        nash_labels = []
        nash_x = []
        nash_y = []
        for eq in nash_eq:
            i, j = eq
            nash_labels.append(f"NE: ({strategies[i]}, {strategies[j]})")
            nash_x.append(i)
            nash_y.append(j)
        
        fig.add_trace(
            go.Scatter(
                x=nash_x,
                y=nash_y,
                mode='markers+text',
                marker=dict(size=20, color='gold', symbol='star'),
                text=nash_labels,
                textposition='top center',
                name='纳什均衡'
            ),
            row=2, col=2
        )
        
        # 添加所有策略点
        for i in range(len(strategies)):
            for j in range(len(strategies)):
                is_nash = (i, j) in nash_eq
                fig.add_trace(
                    go.Scatter(
                        x=[i],
                        y=[j],
                        mode='markers',
                        marker=dict(
                            size=15 if is_nash else 10,
                            color='gold' if is_nash else 'lightblue',
                            symbol='star' if is_nash else 'circle'
                        ),
                        text=f"({strategies[i]}, {strategies[j]})",
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text="玩家2策略", row=1, col=1)
        fig.update_yaxes(title_text="玩家1策略", row=1, col=1)
        fig.update_xaxes(title_text="玩家2策略", row=1, col=2)
        fig.update_yaxes(title_text="玩家1策略", row=1, col=2)
        fig.update_xaxes(title_text="策略组合", row=2, col=1)
        fig.update_yaxes(title_text="收益", row=2, col=1)
        fig.update_xaxes(title_text="玩家2策略索引", row=2, col=2)
        fig.update_yaxes(title_text="玩家1策略索引", row=2, col=2)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text=f"{game_type} - 纳什均衡分析"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 博弈分析")
        st.info(description)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("纳什均衡数量", len(nash_eq))
        
        with col2:
            if nash_eq:
                avg_p1 = np.mean([payoff_p1[eq] for eq in nash_eq])
                st.metric("玩家1均衡收益", f"{avg_p1:.2f}")
        
        with col3:
            if nash_eq:
                avg_p2 = np.mean([payoff_p2[eq] for eq in nash_eq])
                st.metric("玩家2均衡收益", f"{avg_p2:.2f}")
        
        if show_analysis:
            st.markdown("### 🔍 优势策略分析")
            
            # 检查玩家1的优势策略
            p1_dominant = InteractiveGameTheory._check_dominant_strategy(payoff_p1, axis=0)
            # 检查玩家2的优势策略
            p2_dominant = InteractiveGameTheory._check_dominant_strategy(payoff_p2.T, axis=0)
            
            if p1_dominant is not None:
                st.success(f"✅ 玩家1有优势策略: **{strategies[p1_dominant]}**")
            else:
                st.warning("⚠️ 玩家1没有优势策略")
            
            if p2_dominant is not None:
                st.success(f"✅ 玩家2有优势策略: **{strategies[p2_dominant]}**")
            else:
                st.warning("⚠️ 玩家2没有优势策略")
        
        st.success("""
        **纳什均衡的核心洞察**:
        - **稳定性**: 没有玩家有动机单方面偏离
        - **非效率性**: 纳什均衡可能不是帕累托最优（如囚徒困境）
        - **存在性**: 在有限博弈中总是存在（可能是混合策略）
        - **多重性**: 可能存在多个纳什均衡
        """)
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _get_game_payoffs(game_type):
        """获取不同博弈的收益矩阵"""
        if game_type == "囚徒困境":
            return {
                'p1': np.array([[-1, -3], [0, -2]]),
                'p2': np.array([[-1, 0], [-3, -2]]),
                'strategies': ['合作', '背叛'],
                'nash_eq': [(1, 1)],  # (背叛, 背叛)
                'description': """
                **囚徒困境**: 经典的非合作博弈
                - 双方都选择背叛是纳什均衡
                - 但双方合作的收益更高（帕累托最优）
                - 说明个体理性可能导致集体非理性
                """
            }
        elif game_type == "性别大战":
            return {
                'p1': np.array([[2, 0], [0, 1]]),
                'p2': np.array([[1, 0], [0, 2]]),
                'strategies': ['电影', '球赛'],
                'nash_eq': [(0, 0), (1, 1)],
                'description': """
                **性别大战**: 协调博弈
                - 存在两个纯策略纳什均衡
                - 双方都想一起活动，但偏好不同
                - 说明协调问题的复杂性
                """
            }
        elif game_type == "猎鹿博弈":
            return {
                'p1': np.array([[4, 1], [3, 2]]),
                'p2': np.array([[4, 3], [1, 2]]),
                'strategies': ['猎鹿', '猎兔'],
                'nash_eq': [(0, 0), (1, 1)],
                'description': """
                **猎鹿博弈**: 信任与风险
                - (猎鹿, 猎鹿) 收益高但需要合作
                - (猎兔, 猎兔) 收益低但安全
                - 说明信任建立的困难
                """
            }
        else:  # 剪刀石头布
            return {
                'p1': np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
                'p2': np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
                'strategies': ['石头', '剪刀', '布'],
                'nash_eq': [],  # 只有混合策略均衡
                'description': """
                **剪刀石头布**: 零和博弈
                - 没有纯策略纳什均衡
                - 存在混合策略均衡（各1/3概率）
                - 说明完全竞争中的随机化策略
                """
            }
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _check_dominant_strategy(payoff_matrix, axis=0):
        """检查是否存在优势策略"""
        n = payoff_matrix.shape[axis]
        for i in range(n):
            is_dominant = True
            for j in range(n):
                if i == j:
                    continue
                # 检查策略i是否严格优于策略j
                if axis == 0:
                    if not np.all(payoff_matrix[i, :] >= payoff_matrix[j, :]):
                        is_dominant = False
                        break
                else:
                    if not np.all(payoff_matrix[:, i] >= payoff_matrix[:, j]):
                        is_dominant = False
                        break
            if is_dominant:
                return i
        return None
    
    @staticmethod
    def _render_minmax_dynamics():
        """极小极大动力学演示"""
        st.markdown("### 🌀 极小极大优化：从旋转到收敛")
        
        st.markdown(r"""
        **经典问题**: $\min_x \max_y f(x,y) = xy$
        
        **梯度下降-上升 (GDA)**:
        """)
        
        st.latex(r"""
        \begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = 
        \begin{bmatrix} -y \\ x \end{bmatrix} = 
        \underbrace{\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}}_{J} 
        \begin{bmatrix} x \\ y \end{bmatrix}
        """)
        
        st.markdown(r"""
        **雅可比矩阵**: $J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$
        
        **特征值**: $\lambda = \pm i$ (纯虚数，纯旋转！)
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            dynamics_type = st.selectbox(
                "动力学类型",
                ["朴素GDA", "梯度惩罚", "辛修正", "对比分析"]
            )
            learning_rate = st.slider("学习率 η", 0.01, 0.3, 0.1, 0.01)
            lambda_reg = st.slider("正则化强度 λ", 0.0, 1.0, 0.5, 0.05)
            initial_x = st.slider("初始 x", -2.0, 2.0, 1.5, 0.1)
            initial_y = st.slider("初始 y", -2.0, 2.0, 0.0, 0.1)
            n_steps = st.slider("迭代步数", 50, 500, 200, 50)
        
        # 生成轨迹
        if dynamics_type == "对比分析":
            # 对比多种方法
            trajectories = {}
            for dtype in ["朴素GDA", "梯度惩罚", "辛修正"]:
                traj = InteractiveGameTheory._simulate_minmax(
                    initial_x, initial_y, learning_rate, lambda_reg, n_steps, dtype
                )
                trajectories[dtype] = traj
        else:
            # 单一方法
            trajectory = InteractiveGameTheory._simulate_minmax(
                initial_x, initial_y, learning_rate, lambda_reg, n_steps, dynamics_type
            )
            trajectories = {dynamics_type: trajectory}
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("轨迹演化", "向量场"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = {'朴素GDA': 'red', '梯度惩罚': 'blue', '辛修正': 'green'}
        
        # 绘制轨迹
        for name, traj in trajectories.items():
            color = colors.get(name, 'purple')
            
            # 轨迹线
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=3),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 起点
            fig.add_trace(
                go.Scatter(
                    x=[traj[0, 0]],
                    y=[traj[0, 1]],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='circle'),
                    name=f'{name} 起点',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 终点
            fig.add_trace(
                go.Scatter(
                    x=[traj[-1, 0]],
                    y=[traj[-1, 1]],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='square'),
                    name=f'{name} 终点',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 纳什均衡点
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=20, color='gold', symbol='star'),
                name='纳什均衡',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 向量场
        x_range = np.linspace(-2, 2, 20)
        y_range = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 计算向量场（使用第一个动力学类型）
        first_type = list(trajectories.keys())[0]
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                grad = InteractiveGameTheory._compute_gradient(
                    X[j, i], Y[j, i], lambda_reg, first_type
                )
                U[j, i] = grad[0]
                V[j, i] = grad[1]
        
        # 绘制向量场（使用箭头）
        for i in range(0, len(x_range), 2):
            for j in range(0, len(y_range), 2):
                fig.add_annotation(
                    x=X[j, i] + U[j, i] * 0.1,
                    y=Y[j, i] + V[j, i] * 0.1,
                    ax=X[j, i],
                    ay=Y[j, i],
                    xref='x2', yref='y2',
                    axref='x2', ayref='y2',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='gray',
                    opacity=0.5
                )
        
        # 在向量场图上也显示轨迹
        for name, traj in trajectories.items():
            color = colors.get(name, 'purple')
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="x (Player 1)", range=[-2.5, 2.5], row=1, col=1)
        fig.update_yaxes(title_text="y (Player 2)", range=[-2.5, 2.5], row=1, col=1)
        fig.update_xaxes(title_text="x (Player 1)", range=[-2.5, 2.5], row=1, col=2)
        fig.update_yaxes(title_text="y (Player 2)", range=[-2.5, 2.5], row=1, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text=f"极小极大动力学 - {dynamics_type if dynamics_type != '对比分析' else '多方法对比'}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计分析
        st.markdown("### 📊 轨迹分析")
        
        cols = st.columns(len(trajectories))
        for idx, (name, traj) in enumerate(trajectories.items()):
            with cols[idx]:
                # 计算收敛性指标
                final_dist = np.linalg.norm(traj[-1])
                initial_dist = np.linalg.norm(traj[0])
                total_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                
                st.markdown(f"**{name}**")
                st.metric("最终距离", f"{final_dist:.4f}")
                st.metric("轨迹长度", f"{total_length:.2f}")
                
                if final_dist < 0.1:
                    st.success("✅ 收敛")
                elif final_dist < initial_dist:
                    st.info("📉 接近中")
                else:
                    st.error("❌ 发散/旋转")
        
        # 理论解释
        st.markdown("### 🔬 理论分析")
        
        st.info("""
        **朴素GDA的问题**:
        - 特征值 $\\lambda = \\pm i$ (纯虚数)
        - 导致纯旋转，无法收敛
        - 能量守恒: $\\frac{d}{dt}(x^2 + y^2) = 0$
        
        **修正策略**:
        1. **梯度惩罚**: 添加 $-\\lambda x$ 和 $-\\lambda y$ 项
        2. **辛修正**: 调整更新方向，引入"摩擦力"
        3. **目标**: 使特征值具有负实部 $\\lambda = -\\alpha \\pm i\\beta$
        """)
        
        st.success("""
        **深度学习启示**:
        - **GAN训练不稳定**: 本质是纯旋转动力学
        - **WGAN-GP**: 梯度惩罚引入收敛项
        - **Spectral Normalization**: 控制雅可比矩阵的特征值
        - **关键**: 将旋转场变为收敛场
        """)
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _simulate_minmax(x0, y0, eta, lambda_reg, n_steps, dynamics_type):
        """模拟极小极大动力学"""
        trajectory = np.zeros((n_steps, 2))
        trajectory[0] = [x0, y0]
        
        for t in range(1, n_steps):
            x, y = trajectory[t-1]
            grad = InteractiveGameTheory._compute_gradient(x, y, lambda_reg, dynamics_type)
            trajectory[t] = trajectory[t-1] + eta * grad
        
        return trajectory
    
    @cache_heavy
    @staticmethod
    def _compute_gradient(x, y, lambda_reg, dynamics_type):
        """计算不同动力学下的梯度"""
        if dynamics_type == "朴素GDA":
            # 纯旋转: dx/dt = -y, dy/dt = x
            return np.array([-y, x])
        
        elif dynamics_type == "梯度惩罚":
            # 添加正则化项: dx/dt = -y - λx, dy/dt = x - λy
            return np.array([-y - lambda_reg * x, x - lambda_reg * y])
        
        elif dynamics_type == "辛修正":
            # 辛修正: 调整梯度方向
            return np.array([-y - lambda_reg * x, x - lambda_reg * y])
        
        else:
            return np.array([-y, x])
    
    @staticmethod
    def _render_jacobian_analysis():
        """雅可比矩阵分析"""
        st.markdown("### 🔍 雅可比矩阵：系统稳定性的关键")
        
        st.markdown(r"""
        **核心思想**: 雅可比矩阵的特征值决定系统的稳定性
        
        **特征值分类**:
        - **实部 < 0**: 稳定收敛（有摩擦力）
        - **实部 = 0**: 临界状态（纯旋转）
        - **实部 > 0**: 不稳定发散
        - **虚部 ≠ 0**: 存在旋转/震荡
        
        **极小极大问题**: $\min_x \max_y f(x,y) = xy$
        """)
        
        st.latex(r"""
        J = \begin{bmatrix} 
        \frac{\partial \dot{x}}{\partial x} & \frac{\partial \dot{x}}{\partial y} \\ 
        \frac{\partial \dot{y}}{\partial x} & \frac{\partial \dot{y}}{\partial y} 
        \end{bmatrix}
        """)
        
        st.markdown("")
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            analysis_type = st.selectbox(
                "分析类型",
                ["朴素GDA", "添加正则化", "自定义矩阵"]
            )
            
            if analysis_type == "添加正则化":
                reg_strength = st.slider("正则化强度", 0.0, 2.0, 0.5, 0.1)
            elif analysis_type == "自定义矩阵":
                st.markdown("雅可比矩阵元素")
                j11 = st.slider("J[0,0]", -2.0, 2.0, 0.0, 0.1)
                j12 = st.slider("J[0,1]", -2.0, 2.0, -1.0, 0.1)
                j21 = st.slider("J[1,0]", -2.0, 2.0, 1.0, 0.1)
                j22 = st.slider("J[1,1]", -2.0, 2.0, 0.0, 0.1)
        
        # 构造雅可比矩阵
        if analysis_type == "朴素GDA":
            J = np.array([[0, -1], [1, 0]])
            description = "朴素GDA: 纯旋转矩阵"
        elif analysis_type == "添加正则化":
            J = np.array([[-reg_strength, -1], [1, -reg_strength]])
            description = f"添加正则化 (λ={reg_strength})"
        else:
            J = np.array([[j11, j12], [j21, j22]])
            description = "自定义雅可比矩阵"
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eig(J)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "雅可比矩阵",
                "特征值分布",
                "相空间轨迹",
                "稳定性分析"
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # 1. 雅可比矩阵热图
        fig.add_trace(
            go.Heatmap(
                z=J,
                x=['x', 'y'],
                y=['ẋ', 'ẏ'],
                colorscale='RdBu',
                zmid=0,
                text=J,
                texttemplate='%{text:.2f}',
                textfont={"size": 16},
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. 特征值在复平面上的分布
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            fig.add_trace(
                go.Scatter(
                    x=[np.real(val)],
                    y=[np.imag(val)],
                    mode='markers+text',
                    marker=dict(size=15, color='red' if np.real(val) > 0 else 'blue'),
                    text=[f'λ{i+1}'],
                    textposition='top center',
                    name=f'特征值 {i+1}',
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 添加虚轴
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 添加稳定区域着色
        fig.add_vrect(
            x0=-3, x1=0,
            fillcolor="green", opacity=0.1,
            annotation_text="稳定区", annotation_position="top left",
            row=1, col=2
        )
        fig.add_vrect(
            x0=0, x1=3,
            fillcolor="red", opacity=0.1,
            annotation_text="不稳定区", annotation_position="top right",
            row=1, col=2
        )
        
        # 3. 相空间轨迹
        # 模拟多条轨迹
        initial_points = [
            [1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5],
            [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]
        ]
        
        for init_point in initial_points:
            # 简单的欧拉方法模拟
            trajectory = [init_point]
            for _ in range(100):
                current = trajectory[-1]
                update = J @ current
                next_point = current + 0.05 * update
                trajectory.append(next_point)
                # 防止发散到无穷
                if np.linalg.norm(next_point) > 5:
                    break
            
            trajectory = np.array(trajectory)
            fig.add_trace(
                go.Scatter(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode='lines',
                    line=dict(width=1),
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 纳什均衡点
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name='均衡点',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 稳定性分析表格
        stability_data = []
        for i, val in enumerate(eigenvalues):
            real_part = np.real(val)
            imag_part = np.imag(val)
            magnitude = np.abs(val)
            
            if real_part < -0.01:
                stability = "稳定"
            elif real_part > 0.01:
                stability = "不稳定"
            else:
                stability = "临界"
            
            if abs(imag_part) > 0.01:
                behavior = "旋转"
            else:
                behavior = "纯指数"
            
            stability_data.append([
                f"λ{i+1}",
                f"{real_part:.3f}",
                f"{imag_part:.3f}",
                f"{magnitude:.3f}",
                behavior,
                stability
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["特征值", "实部", "虚部", "模", "行为", "稳定性"],
                    fill_color='paleturquoise',
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*stability_data)),
                    fill_color='lavender',
                    align='center'
                )
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="∂/∂x", row=1, col=1)
        fig.update_yaxes(title_text="d/dt", row=1, col=1)
        fig.update_xaxes(title_text="实部 Re(λ)", range=[-2, 2], row=1, col=2)
        fig.update_yaxes(title_text="虚部 Im(λ)", range=[-2, 2], row=1, col=2)
        fig.update_xaxes(title_text="x", range=[-3, 3], row=2, col=1)
        fig.update_yaxes(title_text="y", range=[-3, 3], row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"雅可比矩阵分析 - {description}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 系统诊断")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_real = max(np.real(eigenvalues))
            if max_real < -0.01:
                stability_status = "稳定"
                color = "green"
            elif max_real > 0.01:
                stability_status = "不稳定"
                color = "red"
            else:
                stability_status = "临界"
                color = "orange"
            st.metric("系统稳定性", stability_status)
        
        with col2:
            has_rotation = any(abs(np.imag(val)) > 0.01 for val in eigenvalues)
            st.metric("是否旋转", "是" if has_rotation else "否")
        
        with col3:
            max_magnitude = max(np.abs(eigenvalues))
            st.metric("最大特征值模", f"{max_magnitude:.3f}")
        
        with col4:
            trace = np.trace(J)
            st.metric("迹 Tr(J)", f"{trace:.3f}")
        
        # 理论解释
        st.markdown("### 🎓 理论要点")
        
        st.info(f"""
        **雅可比矩阵**:
        ```
        J = {J}
        ```
        
        **行列式**: det(J) = {np.linalg.det(J):.3f}
        
        **迹**: Tr(J) = {np.trace(J):.3f}
        
        **特征方程**: det(J - λI) = 0
        """)
        
        st.success("""
        **判断准则**:
        
        1. **Tr(J) < 0**: 系统有收敛趋势
        2. **det(J) > 0**: 特征值同号或共轭
        3. **Re(λ) < 0**: 所有特征值实部为负 → 渐近稳定
        4. **Re(λ) = 0**: 临界稳定（如纯GDA）
        5. **Re(λ) > 0**: 至少一个特征值实部为正 → 不稳定
        
        **应用**:
        - **GAN训练**: 需要使 Re(λ) < 0
        - **强化学习**: 策略梯度的稳定性分析
        - **对抗训练**: 确保收敛而非震荡
        """)
    
    @staticmethod
    def _render_stackelberg():
        """Stackelberg博弈演示"""
        st.markdown("### 👑 Stackelberg博弈：Leader-Follower动态")
        
        st.markdown(r"""
        **核心概念**: 领导者考虑跟随者的最优反应
        
        **数学表达**:
        
        - Leader优化: $\min_{\theta_1} U(\theta_1, \theta_2^*(\theta_1))$
        - Follower优化: $\theta_2^* = \arg\min_{\theta_2} L_{follower}(\theta_1, \theta_2)$
        
        **隐函数梯度**:
        """)
        
        st.latex(r"""
        \frac{d\theta_2^*}{d\theta_1} = -[\nabla_{\theta_2\theta_2}^2 L]^{-1} \nabla_{\theta_1\theta_2}^2 L
        """)
        
        st.markdown(r"""
        **Total Gradient**:
        """)
        
        st.latex(r"""
        \nabla_{\theta_1}^{Total} = \frac{\partial U}{\partial \theta_1} + 
        \frac{\partial U}{\partial \theta_2} \cdot \frac{d\theta_2^*}{d\theta_1}
        """)
        
        st.markdown("")
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            scenario = st.selectbox(
                "博弈场景",
                ["定价竞争", "质量竞争", "广告投入"]
            )
            leader_strategy = st.slider("Leader初始策略", 0.0, 10.0, 5.0, 0.5)
            show_reaction = st.checkbox("显示反应函数", value=True)
        
        # 定义Stackelberg博弈
        if scenario == "定价竞争":
            # Leader价格，Follower最优反应
            def follower_best_response(p1):
                # 假设线性需求: p2 = (10 - p1) / 2
                return (10 - p1) / 2
            
            def leader_profit(p1):
                p2 = follower_best_response(p1)
                # Leader利润: (p1 - c1) * q1, q1 = 10 - p1 - 0.5*p2
                c1 = 2  # Leader成本
                q1 = 10 - p1 - 0.5 * p2
                return (p1 - c1) * q1
            
            x_label = "价格"
            y_label = "数量/利润"
            
        elif scenario == "质量竞争":
            def follower_best_response(q1):
                return 0.8 * q1  # Follower质量略低
            
            def leader_profit(q1):
                q2 = follower_best_response(q1)
                # 假设利润与质量差相关
                return q1 * (10 - q1) - 0.5 * (q1 - q2)**2
            
            x_label = "质量"
            y_label = "利润"
            
        else:  # 广告投入
            def follower_best_response(a1):
                return 0.5 * np.sqrt(a1)  # 跟随者策略
            
            def leader_profit(a1):
                a2 = follower_best_response(a1)
                # 利润: 收益 - 成本
                return np.sqrt(a1) * 10 - a1 - 0.3 * a2**2
            
            x_label = "广告投入"
            y_label = "利润"
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("反应函数", "Leader利润最大化"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. 反应函数曲线
        x_range = np.linspace(0.1, 10, 100)
        follower_responses = [follower_best_response(x) for x in x_range]
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=follower_responses,
                mode='lines',
                name='Follower反应函数',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # 当前Leader策略点
        current_response = follower_best_response(leader_strategy)
        fig.add_trace(
            go.Scatter(
                x=[leader_strategy],
                y=[current_response],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='当前策略',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. Leader利润曲线
        leader_profits = [leader_profit(x) for x in x_range]
        optimal_idx = np.argmax(leader_profits)
        optimal_strategy = x_range[optimal_idx]
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=leader_profits,
                mode='lines',
                name='Leader利润',
                line=dict(color='green', width=3)
            ),
            row=1, col=2
        )
        
        # 最优点
        fig.add_trace(
            go.Scatter(
                x=[optimal_strategy],
                y=[leader_profits[optimal_idx]],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name='最优策略',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 当前点
        fig.add_trace(
            go.Scatter(
                x=[leader_strategy],
                y=[leader_profit(leader_strategy)],
                mode='markers',
                marker=dict(size=12, color='red', symbol='circle'),
                name='当前利润',
                showlegend=True
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text=f"Leader {x_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Follower {x_label}", row=1, col=1)
        fig.update_xaxes(title_text=f"Leader {x_label}", row=1, col=2)
        fig.update_yaxes(title_text=y_label, row=1, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text=f"Stackelberg博弈 - {scenario}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 博弈分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("最优Leader策略", f"{optimal_strategy:.2f}")
        
        with col2:
            optimal_follower = follower_best_response(optimal_strategy)
            st.metric("对应Follower策略", f"{optimal_follower:.2f}")
        
        with col3:
            max_profit = leader_profits[optimal_idx]
            st.metric("最大Leader利润", f"{max_profit:.2f}")
        
        with col4:
            current_profit = leader_profit(leader_strategy)
            improvement = max_profit - current_profit
            st.metric("可改进空间", f"{improvement:.2f}")
        
        st.success("""
        **Stackelberg博弈的关键**:
        
        1. **先动优势**: Leader通过承诺获得优势
        2. **隐函数梯度**: 考虑Follower的反应
        3. **计算复杂度**: 需要计算Hessian逆，$O(n^3)$
        4. **工程近似**: 使用Neumann级数或共轭梯度
        
        **应用场景**:
        - **元学习 (MAML)**: Outer loop是Leader
        - **神经架构搜索 (DARTS)**: 架构参数是Leader
        - **对抗训练**: 防御者是Leader
        """)
    
    @staticmethod
    def _render_lola():
        """LOLA算法演示"""
        st.markdown("### 🤝 LOLA：对手感知学习")
        
        st.markdown(r"""
        **Learning with Opponent-Learning Awareness (LOLA)**
        
        **核心思想**: 智能体不仅优化自己的收益，还要考虑对手的学习过程
        
        **标准更新** (朴素学习):
        """)
        
        st.latex(r"""
        \theta_1 \leftarrow \theta_1 - \eta \nabla_{\theta_1} L_1(\theta_1, \theta_2)
        """)
        
        st.markdown(r"""
        **LOLA更新** (对手感知):
        """)
        
        st.latex(r"""
        \theta_1 \leftarrow \theta_1 - \eta \left[\nabla_{\theta_1} L_1 + 
        \nabla_{\theta_2} L_1 \cdot \frac{d\theta_2}{d\theta_1}\right]
        """)
        
        st.markdown(r"""
        **效果**: 在重复囚徒困境中自发涌现合作行为！
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            algorithm = st.selectbox(
                "学习算法",
                ["LOLA", "标准梯度", "对比分析"]
            )
            learning_rate = st.slider("学习率", 0.01, 0.5, 0.1, 0.01)
            n_episodes = st.slider("训练轮数", 50, 500, 200, 50)
            game_scenario = st.selectbox(
                "博弈场景",
                ["囚徒困境", "协调博弈", "混合策略"]
            )
        
        # 定义收益矩阵
        if game_scenario == "囚徒困境":
            payoff_matrix = {
                'p1': np.array([[-1, -3], [0, -2]]),
                'p2': np.array([[-1, 0], [-3, -2]])
            }
            optimal_action = "合作"
        elif game_scenario == "协调博弈":
            payoff_matrix = {
                'p1': np.array([[2, 0], [0, 1]]),
                'p2': np.array([[2, 0], [0, 1]])
            }
            optimal_action = "协调"
        else:  # 混合策略
            payoff_matrix = {
                'p1': np.array([[1, -1], [-1, 1]]),
                'p2': np.array([[-1, 1], [1, -1]])
            }
            optimal_action = "混合"
        
        # 运行模拟
        if algorithm == "对比分析":
            results = {}
            for alg in ["LOLA", "标准梯度"]:
                results[alg] = InteractiveGameTheory._simulate_lola(
                    payoff_matrix, alg, learning_rate, n_episodes
                )
        else:
            results = {
                algorithm: InteractiveGameTheory._simulate_lola(
                    payoff_matrix, algorithm, learning_rate, n_episodes
                )
            }
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "策略演化",
                "累积收益",
                "合作率/协调率",
                "策略空间轨迹"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        colors = {'LOLA': 'blue', '标准梯度': 'red'}
        
        for alg_name, result in results.items():
            color = colors[alg_name]
            
            # 1. 策略演化（玩家1选择合作的概率）
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=result['p1_strategy'][:, 0],  # 选择第一个动作的概率
                    mode='lines',
                    name=f'{alg_name} - P1',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=result['p2_strategy'][:, 0],
                    mode='lines',
                    name=f'{alg_name} - P2',
                    line=dict(color=color, width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # 2. 累积收益
            cumulative_p1 = np.cumsum(result['p1_rewards'])
            cumulative_p2 = np.cumsum(result['p2_rewards'])
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=cumulative_p1,
                    mode='lines',
                    name=f'{alg_name} - P1收益',
                    line=dict(color=color, width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=cumulative_p2,
                    mode='lines',
                    name=f'{alg_name} - P2收益',
                    line=dict(color=color, width=2, dash='dash')
                ),
                row=1, col=2
            )
            
            # 3. 合作率（双方都选择第一个动作的概率）
            cooperation_rate = result['p1_strategy'][:, 0] * result['p2_strategy'][:, 0]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_episodes)),
                    y=cooperation_rate,
                    mode='lines',
                    name=f'{alg_name} - 合作率',
                    line=dict(color=color, width=3),
                    fill='tozeroy',
                    fillcolor=f'rgba({255 if alg_name == "标准梯度" else 0}, {0}, {255 if alg_name == "LOLA" else 0}, 0.2)'
                ),
                row=2, col=1
            )
            
            # 4. 策略空间轨迹
            fig.add_trace(
                go.Scatter(
                    x=result['p1_strategy'][:, 0],
                    y=result['p2_strategy'][:, 0],
                    mode='lines+markers',
                    name=f'{alg_name} 轨迹',
                    line=dict(color=color, width=2),
                    marker=dict(size=3)
                ),
                row=2, col=2
            )
            
            # 起点和终点
            fig.add_trace(
                go.Scatter(
                    x=[result['p1_strategy'][0, 0]],
                    y=[result['p2_strategy'][0, 0]],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='circle'),
                    name=f'{alg_name} 起点',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[result['p1_strategy'][-1, 0]],
                    y=[result['p2_strategy'][-1, 0]],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='star'),
                    name=f'{alg_name} 终点',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 添加纳什均衡点（如果是囚徒困境）
        if game_scenario == "囚徒困境":
            # (背叛, 背叛) = (0, 0) in probability space
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    marker=dict(size=15, color='black', symbol='x'),
                    text=['纳什均衡<br>(背叛,背叛)'],
                    textposition='bottom center',
                    name='纳什均衡',
                    showlegend=True
                ),
                row=2, col=2
            )
            
            # 帕累托最优点
            fig.add_trace(
                go.Scatter(
                    x=[1], y=[1],
                    mode='markers+text',
                    marker=dict(size=15, color='gold', symbol='star'),
                    text=['帕累托最优<br>(合作,合作)'],
                    textposition='top center',
                    name='帕累托最优',
                    showlegend=True
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="轮数", row=1, col=1)
        fig.update_yaxes(title_text="动作1概率", row=1, col=1)
        fig.update_xaxes(title_text="轮数", row=1, col=2)
        fig.update_yaxes(title_text="累积收益", row=1, col=2)
        fig.update_xaxes(title_text="轮数", row=2, col=1)
        fig.update_yaxes(title_text="合作率", row=2, col=1)
        fig.update_xaxes(title_text="P1动作1概率", range=[0, 1], row=2, col=2)
        fig.update_yaxes(title_text="P2动作1概率", range=[0, 1], row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"LOLA算法演示 - {game_scenario}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计分析
        st.markdown("### 📊 学习结果分析")
        
        cols = st.columns(len(results))
        for idx, (alg_name, result) in enumerate(results.items()):
            with cols[idx]:
                st.markdown(f"**{alg_name}**")
                
                # 最终策略
                final_p1 = result['p1_strategy'][-1, 0]
                final_p2 = result['p2_strategy'][-1, 0]
                st.metric("P1最终策略", f"{final_p1:.3f}")
                st.metric("P2最终策略", f"{final_p2:.3f}")
                
                # 总收益
                total_p1 = np.sum(result['p1_rewards'])
                total_p2 = np.sum(result['p2_rewards'])
                st.metric("P1总收益", f"{total_p1:.1f}")
                st.metric("P2总收益", f"{total_p2:.1f}")
                
                # 最终合作率
                final_coop = final_p1 * final_p2
                st.metric("最终合作率", f"{final_coop:.3f}")
                
                # 判断收敛情况
                if game_scenario == "囚徒困境":
                    if final_coop > 0.7:
                        st.success("✅ 成功涌现合作")
                    elif final_coop > 0.3:
                        st.info("📊 部分合作")
                    else:
                        st.warning("⚠️ 陷入背叛")
        
        # 理论解释
        st.markdown("### 🎓 理论洞察")
        
        st.success("""
        **LOLA的优势**:
        
        1. **对手建模**: 显式考虑对手的学习动态
           - $\\frac{d\\theta_2}{d\\theta_1}$ 捕捉对手如何响应
        
        2. **长期视角**: 优化长期累积收益而非短期
           - 愿意短期牺牲换取长期合作
        
        3. **合作涌现**: 在重复囚徒困境中自发产生"针锋相对"
           - 无需显式编程合作策略
        
        4. **策略塑造**: 通过影响对手的学习引导向有利方向
           - 类似Stackelberg的Leader思维
        """)
        
        st.info("""
        **标准梯度的局限**:
        
        1. **短视**: 只看当前轮次的收益
        2. **独立学习**: 忽略对手的学习过程
        3. **陷入局部**: 容易陷入次优纳什均衡
        4. **无合作**: 在囚徒困境中总是背叛
        """)
        
        st.warning("""
        **应用场景**:
        
        - **GAN训练**: 生成器和判别器的对抗学习
        - **多智能体强化学习**: 合作任务中的策略学习
        - **对抗攻防**: 攻击者和防御者的博弈
        - **拍卖机制设计**: 多个竞价者的策略优化
        - **交通控制**: 多个自动驾驶车辆的协调
        """)
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _simulate_lola(payoff_matrix, algorithm, lr, n_episodes):
        """模拟LOLA或标准梯度学习"""
        # 初始化策略（softmax参数）
        theta1 = np.zeros(2)
        theta2 = np.zeros(2)
        
        # 记录历史
        history = {
            'p1_strategy': [],
            'p2_strategy': [],
            'p1_rewards': [],
            'p2_rewards': []
        }
        
        for episode in range(n_episodes):
            # Softmax策略
            p1_probs = InteractiveGameTheory._softmax(theta1)
            p2_probs = InteractiveGameTheory._softmax(theta2)
            
            history['p1_strategy'].append(p1_probs.copy())
            history['p2_strategy'].append(p2_probs.copy())
            
            # 采样动作
            a1 = np.random.choice(2, p=p1_probs)
            a2 = np.random.choice(2, p=p2_probs)
            
            # 获得收益
            r1 = payoff_matrix['p1'][a1, a2]
            r2 = payoff_matrix['p2'][a1, a2]
            
            history['p1_rewards'].append(r1)
            history['p2_rewards'].append(r2)
            
            # 计算梯度
            if algorithm == "LOLA":
                # LOLA: 考虑对手学习
                grad1 = InteractiveGameTheory._compute_lola_gradient(
                    theta1, theta2, a1, a2, r1, payoff_matrix['p1']
                )
                grad2 = InteractiveGameTheory._compute_lola_gradient(
                    theta2, theta1, a2, a1, r2, payoff_matrix['p2']
                )
            else:
                # 标准梯度: 策略梯度
                grad1 = InteractiveGameTheory._compute_policy_gradient(
                    theta1, a1, r1
                )
                grad2 = InteractiveGameTheory._compute_policy_gradient(
                    theta2, a2, r2
                )
            
            # 更新参数
            theta1 += lr * grad1
            theta2 += lr * grad2
        
        # 转换为numpy数组
        history['p1_strategy'] = np.array(history['p1_strategy'])
        history['p2_strategy'] = np.array(history['p2_strategy'])
        history['p1_rewards'] = np.array(history['p1_rewards'])
        history['p2_rewards'] = np.array(history['p2_rewards'])
        
        return history
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _softmax(theta):
        """Softmax函数"""
        exp_theta = np.exp(theta - np.max(theta))  # 数值稳定
        return exp_theta / np.sum(exp_theta)
    
    @cache_heavy
    @staticmethod
    def _compute_policy_gradient(theta, action, reward):
        """计算标准策略梯度"""
        probs = InteractiveGameTheory._softmax(theta)
        grad = np.zeros_like(theta)
        grad[action] = reward * (1 - probs[action])
        for a in range(len(theta)):
            if a != action:
                grad[a] = -reward * probs[a]
        return grad
    
    @cache_heavy
    @staticmethod
    def _compute_lola_gradient(theta_self, theta_opp, action_self, action_opp, reward, payoff_matrix):
        """计算LOLA梯度（简化版）"""
        # 标准策略梯度
        base_grad = InteractiveGameTheory._compute_policy_gradient(theta_self, action_self, reward)
        
        # LOLA修正项（简化：假设对手也在做策略梯度）
        # 实际LOLA需要计算 d(theta_opp)/d(theta_self)，这里用启发式近似
        probs_opp = InteractiveGameTheory._softmax(theta_opp)
        
        # 考虑如果对手改变策略，对自己的影响
        correction = np.zeros_like(theta_self)
        for a_self in range(len(theta_self)):
            for a_opp in range(len(theta_opp)):
                future_reward = payoff_matrix[a_self, a_opp]
                correction[a_self] += 0.1 * future_reward * probs_opp[a_opp]
        
        return base_grad + correction

        # 添加交互式测验
