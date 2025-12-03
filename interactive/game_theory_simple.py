"""
交互式博弈论可视化 - 简化版本
严格按照 23.GameTheory.md 中的理论实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

warnings.filterwarnings('ignore')


class InteractiveGameTheory:
    """交互式博弈论可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎮 博弈论：从静态优化到动态均衡")
        st.markdown(r"""
        **核心思想**: 当优化目标取决于对手策略时，极小值点变为鞍点，需要动力学分析
        
        关键概念：
        - **纳什均衡**: $\theta_1^* = \arg\min_{\theta_1} L_1(\theta_1, \theta_2^*)$, $\theta_2^* = \arg\min_{\theta_2} L_2(\theta_1^*, \theta_2)$
        - **极小极大优化**: $\min_x \max_y f(x,y) = xy$
        - **雅可比分析**: 特征值决定系统稳定性
        - **动力学修正**: 从纯旋转到螺旋收敛
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["纳什均衡基础", "零和博弈动力学", "Stackelberg博弈", "多智能体学习"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "纳什均衡基础":
            InteractiveGameTheory._render_nash_equilibrium()
        elif viz_type == "零和博弈动力学":
            InteractiveGameTheory._render_zero_sum_dynamics()
        elif viz_type == "Stackelberg博弈":
            InteractiveGameTheory._render_stackelberg()
        elif viz_type == "多智能体学习":
            InteractiveGameTheory._render_multi_agent()
    

        # 添加交互式测验
        quiz_system = QuizSystem("game_theory_simple")
        quizzes = QuizTemplates.get_game_theory_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_nash_equilibrium():
        """纳什均衡基础演示"""
        st.markdown("### 🎯 纳什均衡：从优化到博弈")
        
        st.markdown(r"""
        **核心概念**：
        - **传统优化**: $\min_\theta L(\theta)$
        - **博弈优化**: 寻找 $(\theta_1^*, \theta_2^*)$ 使得双方都无法单方面获益
        - **存在性**: Brouwer不动点定理保证纳什均衡存在
        """)
        
        with st.sidebar:
            game_type = st.selectbox("博弈类型", 
                ["囚徒困境", "性别大战", "零和博弈"])
            show_best_responses = st.checkbox("显示最优反应函数", value=True)
        
        # 定义不同博弈的收益矩阵
        if game_type == "囚徒困境":
            # 囚徒困境：背叛是优势策略
            payoff_matrix = {
                'player1': np.array([[3, 0], [5, 1]]),  # 行：合作/背叛，列：合作/背叛
                'player2': np.array([[3, 5], [0, 1]])
            }
            strategies = ['合作', '背叛']
            nash_eq = [(1, 1)]  # (背叛, 背叛)
            
        elif game_type == "性别大战":
            # 性别大战：协调博弈
            payoff_matrix = {
                'player1': np.array([[2, 0], [0, 1]]),
                'player2': np.array([[1, 0], [0, 2]])
            }
            strategies = ['电影', '球赛']
            nash_eq = [(0, 0), (1, 1)]  # 两个纳什均衡
            
        else:  # 零和博弈
            payoff_matrix = {
                'player1': np.array([[1, -1], [-1, 1]]),
                'player2': np.array([[-1, 1], [1, -1]])
            }
            strategies = ['策略A', '策略B']
            nash_eq = [(0, 0), (1, 1)]  # 混合策略纳什均衡
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "收益矩阵 (玩家1)", "收益矩阵 (玩家2)",
                "最优反应函数", "策略空间分析"
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 玩家1收益矩阵
        fig.add_trace(
            go.Heatmap(
                z=payoff_matrix['player1'],
                x=strategies,
                y=strategies,
                colorscale='RdBu',
                name='玩家1收益',
                showscale=False
            ),
            row=1, col=1
        )
        
        # 玩家2收益矩阵
        fig.add_trace(
            go.Heatmap(
                z=payoff_matrix['player2'],
                x=strategies,
                y=strategies,
                colorscale='RdBu',
                name='玩家2收益',
                showscale=False
            ),
            row=1, col=2
        )
        
        if show_best_responses:
            # 计算最优反应
            best_responses_p1 = []
            best_responses_p2 = []
            
            for i in range(2):
                # 玩家1的最优反应（对玩家2的每个策略）
                best_response_p1 = np.argmax(payoff_matrix['player1'][:, i])
                best_responses_p1.append((best_response_p1, i))
                
                # 玩家2的最优反应（对玩家1的每个策略）
                best_response_p2 = np.argmax(payoff_matrix['player2'][i, :])
                best_responses_p2.append((i, best_response_p2))
            
            # 绘制最优反应函数
            for br in best_responses_p1:
                fig.add_trace(
                    go.Scatter(
                        x=[br[1]], y=[br[0]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='circle'),
                        name='P1最优反应',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            for br in best_responses_p2:
                fig.add_trace(
                    go.Scatter(
                        x=[br[1]], y=[br[0]],
                        mode='markers',
                        marker=dict(size=15, color='blue', symbol='diamond'),
                        name='P2最优反应',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # 标记纳什均衡
            for eq in nash_eq:
                fig.add_trace(
                    go.Scatter(
                        x=[eq[1]], y=[eq[0]],
                        mode='markers',
                        marker=dict(size=20, color='green', symbol='star'),
                        name='纳什均衡',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 策略分析表格
        analysis_data = []
        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                analysis_data.append([
                    f"{s1} vs {s2}",
                    f"{payoff_matrix['player1'][i,j]:.1f}",
                    f"{payoff_matrix['player2'][i,j]:.1f}",
                    "是" if (i,j) in nash_eq else "否"
                ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=["策略组合", "玩家1收益", "玩家2收益", "纳什均衡"]),
                cells=dict(values=list(zip(*analysis_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"{game_type} - 纳什均衡分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 博弈分析")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("纳什均衡数量", len(nash_eq))
        with col2:
            total_payoff_p1 = sum(payoff_matrix['player1'][eq[0], eq[1]] for eq in nash_eq)
            st.metric("P1均衡收益", f"{total_payoff_p1/len(nash_eq):.2f}")
        with col3:
            total_payoff_p2 = sum(payoff_matrix['player2'][eq[0], eq[1]] for eq in nash_eq)
            st.metric("P2均衡收益", f"{total_payoff_p2/len(nash_eq):.2f}")
        with col4:
            pareto_optimal = 0  # 简化：假设(0,0)是帕累托最优
            st.metric("帕累托最优", f"{pareto_optimal}")
        
        st.success("""
        **纳什均衡的核心洞察**：
        - **稳定性**: 没有玩家有动机单方面偏离
        - **非效率性**: 纳什均衡可能不是帕累托最优
        - **存在性**: 在有限博弈中总是存在
        - **多重性**: 可能存在多个纳什均衡
        """)
    
    @staticmethod
    def _render_zero_sum_dynamics():
        """零和博弈动力学演示"""
        st.markdown("### 🌀 零和博弈动力学：从旋转到收敛")
        
        st.markdown(r"""
        **核心问题**: $\min_x \max_y f(x,y) = xy$
        
        **梯度下降-上升 (GDA)**:
        - $\dot{x} = -\nabla_x f = -y$
        - $\dot{y} = +\nabla_y f = x$
        - 雅可比矩阵: $J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$
        - 特征值: $\lambda = \pm i$ (纯旋转！)
        
        **向量场分解**: $v(\theta) = -\nabla \phi(\theta) + H(\theta)$
        - **势能流**: 雅可比矩阵对称部分，驱动收敛
        - **汉密尔顿流**: 雅可比矩阵反对称部分，能量守恒
        """)
        
        with st.sidebar:
            dynamics_type = st.selectbox("动力学类型", 
                ["朴素GDA", "辛修正", "对比分析"])
            learning_rate = st.slider("学习率 η", 0.01, 0.5, 0.1, 0.01)
            lambda_reg = st.slider("正则化强度 λ", 0.0, 1.0, 0.1, 0.01)
            initial_x = st.slider("初始位置 x", -2.0, 2.0, 1.5, 0.1)
            initial_y = st.slider("初始位置 y", -2.0, 2.0, 0.0, 0.1)
            show_eigenvalues = st.checkbox("显示特征值分析", value=True)
        
        # 定义动力学函数
        def game_dynamics(state, dynamics_type, lambda_reg=0.1):
            x, y = state
            
            if dynamics_type == "朴素GDA":
                # 纯旋转动力学
                return np.array([-y, x])
            
            elif dynamics_type == "辛修正":
                # 添加阻尼项
                v = np.array([-y, x])
                correction = -lambda_reg * np.array([x, y])
                return v + correction
            
            else:  # 对比分析
                return np.array([-y, x])
        
        # 模拟轨迹
        def simulate_trajectory(initial_state, dynamics_type, steps=500):
            trajectory = [initial_state]
            state = np.array(initial_state)
            
            for _ in range(steps):
                update = game_dynamics(state, dynamics_type, lambda_reg)
                state = state + learning_rate * update
                trajectory.append(state.copy())
            
            return np.array(trajectory)
        
        # 计算雅可比矩阵和特征值
        def compute_stability(state, dynamics_type, lambda_reg=0.1):
            x, y = state
            
            if dynamics_type == "朴素GDA":
                J = np.array([[0, -1], [1, 0]])
            elif dynamics_type == "辛修正":
                J = np.array([[-lambda_reg, -1], [1, -lambda_reg]])
            else:  # 对比分析
                J = np.array([[0, -1], [1, 0]])
            
            try:
                eigenvalues = np.linalg.eigvals(J)
            except:
                eigenvalues = np.array([1j, -1j])
            
            # 分解雅可比矩阵
            J_symmetric = (J + J.T) / 2  # 对称部分（势能流）
            J_antisymmetric = (J - J.T) / 2  # 反对称部分（汉密尔顿流）
            
            return J, eigenvalues, J_symmetric, J_antisymmetric
        
        # 可视化
        if dynamics_type == "对比分析":
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "朴素GDA: 纯旋转", "辛修正: 螺旋收敛",
                    "轨迹对比", "特征值分析"
                ]
            )
            
            # 模拟不同动力学
            initial_state = [initial_x, initial_y]
            
            # 朴素GDA
            traj_naive = simulate_trajectory(initial_state, "朴素GDA")
            fig.add_trace(
                go.Scatter(
                    x=traj_naive[:, 0], y=traj_naive[:, 1],
                    mode='lines',
                    name='朴素GDA',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # 辛修正
            traj_symp = simulate_trajectory(initial_state, "辛修正")
            fig.add_trace(
                go.Scatter(
                    x=traj_symp[:, 0], y=traj_symp[:, 1],
                    mode='lines',
                    name='辛修正',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
            # 轨迹对比
            fig.add_trace(
                go.Scatter(
                    x=traj_naive[:, 0], y=traj_naive[:, 1],
                    mode='lines',
                    name='朴素GDA',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=traj_symp[:, 0], y=traj_symp[:, 1],
                    mode='lines',
                    name='辛修正',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 标记纳什均衡
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='star'),
                    name='纳什均衡',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 特征值分析
            methods = ['朴素GDA', '辛修正']
            colors = ['red', 'blue']
            
            for i, method in enumerate(methods):
                J, eigenvals, J_sym, J_antisym = compute_stability([0, 0], method, lambda_reg)
                
                for j, eig in enumerate(eigenvals):
                    fig.add_trace(
                        go.Scatter(
                            x=[np.real(eig)], y=[np.imag(eig)],
                            mode='markers',
                            marker=dict(size=10, color=colors[i]),
                            name=f'{method} λ{j+1}',
                            showlegend=True
                        ),
                        row=2, col=2
                    )
            
            # 添加虚轴
            fig.add_trace(
                go.Scatter(
                    x=[-2, 2], y=[0, 0],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dash'),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0, 0], y=[-2, 2],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dash'),
                    showlegend=False
                ),
                row=2, col=2
            )
            
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "相空间轨迹", "向量场与流线"
                ]
            )
            
            # 模拟轨迹
            initial_state = [initial_x, initial_y]
            trajectory = simulate_trajectory(initial_state, dynamics_type)
            
            # 绘制轨迹
            fig.add_trace(
                go.Scatter(
                    x=trajectory[:, 0], y=trajectory[:, 1],
                    mode='lines',
                    name='优化轨迹',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            # 标记起点和终点
            fig.add_trace(
                go.Scatter(
                    x=[initial_state[0]], y=[initial_state[1]],
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='起点',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[-1, 0]], y=[trajectory[-1, 1]],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='终点',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 向量场
            x_range = np.linspace(-2, 2, 15)
            y_range = np.linspace(-2, 2, 15)
            X, Y = np.meshgrid(x_range, y_range)
            
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            
            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    state = [X[i, j], Y[i, j]]
                    update = game_dynamics(state, dynamics_type, lambda_reg)
                    U[i, j] = update[0]
                    V[i, j] = update[1]
            
            fig.add_trace(
                go.Scatter(
                    x=X.flatten(), y=Y.flatten(),
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=np.sqrt(U.flatten()**2 + V.flatten()**2),
                        colorscale='Viridis',
                        showscale=True
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"{dynamics_type} - 零和博弈动力学分析",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 稳定性分析
        if show_eigenvalues:
            st.markdown("### 🔍 稳定性分析")
            
            J, eigenvalues, J_sym, J_antisym = compute_stability([0, 0], dynamics_type, lambda_reg)
            
            # 确保trajectory变量存在（用于后面的距离计算）
            if 'trajectory' not in locals():
                initial_state = [initial_x, initial_y]
                trajectory = simulate_trajectory(initial_state, dynamics_type)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                real_parts = [np.real(eig) for eig in eigenvalues]
                max_real = max(real_parts)
                stability = "稳定" if max_real < 0 else "不稳定" if max_real > 0 else "临界"
                st.metric("系统稳定性", stability)
            
            with col2:
                max_abs_eig = max(abs(eig) for eig in eigenvalues)
                st.metric("最大特征值模", f"{max_abs_eig:.3f}")
            
            with col3:
                imaginary_parts = [np.imag(eig) for eig in eigenvalues]
                has_rotation = any(abs(imag) > 0.01 for imag in imaginary_parts)
                st.metric("存在旋转", "是" if has_rotation else "否")
            
            with col4:
                # 确保trajectory变量存在
                if 'trajectory' in locals() and len(trajectory) > 0:
                    final_distance = np.linalg.norm(trajectory[-1])
                else:
                    # 如果没有trajectory，计算初始状态到原点的距离
                    initial_state = np.array([initial_x, initial_y])
                    final_distance = np.linalg.norm(initial_state)
                st.metric("到原点距离", f"{final_distance:.3f}")
            
            # 特征值详情
            st.markdown("### 📊 雅可比矩阵特征值")
            
            eigen_data = []
            for i, eig in enumerate(eigenvalues):
                eigen_data.append([
                    f"λ{i+1}",
                    f"{np.real(eig):.3f}",
                    f"{np.imag(eig):.3f}",
                    f"{np.abs(eig):.3f}",
                    "旋转" if abs(np.imag(eig)) > 0.01 else "纯指数"
                ])
            
            st.table(pd.DataFrame(eigen_data, 
                                columns=["特征值", "实部", "虚部", "模", "类型"]))
            
            st.markdown("### 📈 雅可比矩阵")
            st.code(f"J = {J}")
            
            # 向量场分解分析
            st.markdown("### 🌊 向量场分解")
            
            col1, col2 = st.columns(2)
            with col1:
                sym_norm = np.linalg.norm(J_sym)
                st.metric("势能流强度", f"{sym_norm:.3f}")
                st.code(f"J_sym = {J_sym}")
            
            with col2:
                antisym_norm = np.linalg.norm(J_antisym)
                st.metric("汉密尔顿流强度", f"{antisym_norm:.3f}")
                st.code(f"J_antisym = {J_antisym}")
            
            # 结构稳定性分析
            st.markdown("### 🏗️ 结构稳定性分析")
            
            max_real = max(np.real(eigenvalues))
            if max_real < -0.01:
                stability_status = "渐进稳定"
                stability_color = "🟢"
            elif max_real > 0.01:
                stability_status = "不稳定"
                stability_color = "🔴"
            else:
                stability_status = "临界稳定"
                stability_color = "🟡"
            
            st.info(f"""
            **稳定性状态**: {stability_color} {stability_status}
            
            **结构稳定性原理**:
            - 特征值实部 < 0: 系统在有扰动时仍收敛
            - 特征值实部 = 0: 临界情况，微小扰动可能导致发散
            - 特征值实部 > 0: 系统必然发散
            
            **在博弈论中的意义**:
            - 纳什均衡点通常是鞍点，需要特殊处理
            - 朴素的梯度下降在零和博弈中失效
            - 正则化和修正算法是必要的
            """)
        
        st.success("""
        **动力学分析的核心洞察**：
        - **纯虚数特征值**: 导致持续旋转，无法收敛
        - **负实部特征值**: 确保系统收敛到均衡
        - **正实部特征值**: 系统发散，训练不稳定
        - **修正策略**: 通过正则化引入"摩擦力"
        - **向量场分解**: 势能流驱动收敛，汉密尔顿流导致旋转
        - **结构稳定性**: 临界系统需要鲁棒性修正
        """)
    
    @staticmethod
    def _render_stackelberg():
        """Stackelberg博弈演示"""
        st.markdown("### 👑 Stackelberg博弈：Leader-Follower动态")
        
        st.markdown(r"""
        **核心概念**：
        - **Leader**: 先行动，预判Follower反应
        - **Follower**: 后行动，对Leader策略做出最优反应
        - **双层优化**: $\min_{\theta_1} U(\theta_1, \theta_2^*(\theta_1))$
        - **隐函数定理**: $\frac{d\theta_2^*}{d\theta_1} = -[\nabla_{\theta_2^2}^2 L]^{-1} \nabla_{\theta_1\theta_2}^2 L$
        """)
        
        with st.sidebar:
            problem_type = st.selectbox("问题类型", 
                ["简单二次", "非凸博弈", "元学习示例"])
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            iterations = st.slider("迭代次数", 50, 500, 200, 10)
            show_hessian = st.checkbox("显示海森矩阵分析", value=True)
        
        # 定义不同的Stackelberg问题
        if problem_type == "简单二次":
            # Leader: min_x (x^2 + 2xy + y^2)
            # Follower: min_y (x^2 + y^2 + 2xy)
            def leader_objective(x, y):
                return x**2 + 2*x*y + y**2
            
            def follower_objective(x, y):
                return x**2 + y**2 + 2*x*y
            
            def follower_best_response(x):
                # ∂L/∂y = 2y + 2x = 0 => y = -x
                return -x
            
            def leader_gradient(x, y):
                # 考虑follower反应的全梯度
                # ∇_x U = 2x + 2y + 2x*(dy/dx) = 2x + 2y - 2x = 2y
                return np.array([2*y])
            
        elif problem_type == "非凸博弈":
            # 更复杂的非凸问题
            def leader_objective(x, y):
                return x**4 - 2*x**2 + y**2 + x*y
            
            def follower_objective(x, y):
                return y**3 - y + x**2 + 2*x*y
            
            def follower_best_response(x):
                # 数值求解follower的最优反应
                y_vals = np.linspace(-2, 2, 100)
                best_y = y_vals[np.argmin([follower_objective(x, y) for y in y_vals])]
                return best_y
            
            def leader_gradient(x, y):
                # 数值梯度
                eps = 1e-5
                grad = np.zeros(1)
                for i in range(1):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    y_plus = follower_best_response(x_plus[0])
                    
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    y_minus = follower_best_response(x_minus[0])
                    
                    grad[i] = (leader_objective(x_plus, y_plus) - 
                             leader_objective(x_minus, y_minus)) / (2*eps)
                return grad
            
        else:  # 元学习示例
            # 模拟MAML风格的元学习
            def leader_objective(theta, phi):
                # 元损失：在多个任务上的平均损失
                return (theta - phi)**2 + 0.1*theta**2
            
            def follower_objective(theta, phi):
                # 任务损失：内层优化
                return (theta - phi)**2 + 0.5*phi**2
            
            def follower_best_response(theta):
                # 内层最优：phi = theta
                return theta
            
            def leader_gradient(x, y):
                # 元梯度
                return np.array([2*(x-y) + 0.2*x])
        
        # 模拟Stackelberg学习过程
        def simulate_stackelberg(initial_theta, initial_phi, iterations):
            theta_history = []
            phi_history = []
            
            theta = initial_theta.copy()
            
            for _ in range(iterations):
                # Follower对Leader策略做出最优反应
                phi = follower_best_response(theta[0])
                
                # Leader更新（考虑Follower反应）
                grad = leader_gradient(theta, phi)
                theta = theta - learning_rate * grad
                
                theta_history.append(theta.flatten())
                phi_history.append(phi)
            
            return np.array(theta_history), np.array(phi_history)
        
        # 模拟普通博弈（不考虑Stackelberg层次）
        def simulate_simultaneous(initial_theta, initial_phi, iterations):
            theta_history = [initial_theta]
            phi_history = [initial_phi]
            
            theta = initial_theta.copy()
            phi = initial_phi.copy()
            
            for _ in range(iterations):
                # 同时更新（不考虑对方反应）
                grad_theta = np.array([2*theta[0] + 2*phi[0]])  # 简化梯度
                grad_phi = np.array([2*phi[0] + 2*theta[0]])
                
                theta = theta - learning_rate * grad_theta
                phi = phi - learning_rate * grad_phi
                
                theta_history.append(theta.copy())
                phi_history.append(phi)
            
            return np.array(theta_history), np.array(phi_history)
        
        # 运行模拟
        initial_theta = np.array([1.0])
        initial_phi = np.array([-0.5])
        
        theta_stackelberg, phi_stackelberg = simulate_stackelberg(
            initial_theta, initial_phi, iterations
        )
        
        theta_simultaneous, phi_simultaneous = simulate_simultaneous(
            initial_theta, initial_phi, iterations
        )
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "参数演化轨迹", "收敛速度对比",
                "Follower反应函数", "损失函数等高线"
            ]
        )
        
        # 参数演化
        fig.add_trace(
            go.Scatter(
                x=list(range(iterations+1)), y=theta_stackelberg.flatten(),
                mode='lines',
                name='Leader (Stackelberg)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(iterations+1)), y=phi_stackelberg.flatten(),
                mode='lines',
                name='Follower (Stackelberg)',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # 收敛速度对比
        losses_stackelberg = [leader_objective(t, f) for t, f in zip(theta_stackelberg, phi_stackelberg)]
        losses_simultaneous = [leader_objective(t, f) for t, f in zip(theta_simultaneous, phi_simultaneous)]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(iterations+1)), y=losses_stackelberg,
                mode='lines',
                name='Stackelberg',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(iterations+1)), y=losses_simultaneous,
                mode='lines',
                name='Simultaneous',
                line=dict(color='orange', width=2)
            ),
            row=1, col=2
        )
        
        # Follower反应函数
        theta_range = np.linspace(-2, 2, 50)
        phi_responses = [follower_best_response(t) for t in theta_range]
        
        fig.add_trace(
            go.Scatter(
                x=theta_range, y=phi_responses,
                mode='lines',
                name='最优反应',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 标记学习轨迹上的反应点
        fig.add_trace(
            go.Scatter(
                x=theta_stackelberg.flatten(), y=phi_stackelberg.flatten(),
                mode='markers',
                name='学习轨迹',
                marker=dict(color='red', size=4),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 损失函数等高线
        theta_grid = np.linspace(-2, 2, 30)
        phi_grid = np.linspace(-2, 2, 30)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        LOSS = np.zeros_like(THETA)
        for i in range(len(theta_grid)):
            for j in range(len(phi_grid)):
                LOSS[i, j] = leader_objective(THETA[i, j], PHI[i, j])
        
        fig.add_trace(
            go.Contour(
                x=theta_grid, y=phi_grid, z=LOSS,
                contours_coloring='heatmap',
                showscale=False,
                name='损失等高线'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=theta_stackelberg.flatten(), y=phi_stackelberg.flatten(),
                mode='lines+markers',
                name='Stackelberg路径',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Stackelberg博弈 - {problem_type}",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 学习分析")
        
        final_loss_stackelberg = float(losses_stackelberg[-1])
        final_loss_simultaneous = float(losses_simultaneous[-1])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stackelberg最终损失", f"{final_loss_stackelberg:.4f}")
        with col2:
            st.metric("Simultaneous最终损失", f"{final_loss_simultaneous:.4f}")
        with col3:
            improvement = (final_loss_simultaneous - final_loss_stackelberg) / final_loss_simultaneous * 100
            st.metric("改进幅度", f"{improvement:.1f}%")
        with col4:
            if len(losses_stackelberg) > 50:
                # 确保diff后的序列不为空
                diff_values = np.diff(losses_stackelberg[-50:])
                if len(diff_values) > 0:
                    convergence_iter = len(losses_stackelberg) - 50 + np.argmin(np.abs(diff_values))
                    st.metric("收敛迭代", f"{iterations-convergence_iter}")
                else:
                    st.metric("收敛迭代", "计算失败")
            else:
                st.metric("收敛迭代", "数据不足")
        
        if show_hessian:
            st.markdown("### 🔬 海森矩阵分析")
            
            # 简化的海森矩阵计算
            x_final = theta_stackelberg[-1, 0]
            y_final = phi_stackelberg[-1]
            
            # 数值海森矩阵
            eps = 1e-5
            hessian = np.zeros((2, 2))
            
            # 计算二阶导数
            for i in range(2):
                for j in range(2):
                    if i == 0 and j == 0:  # ∂²/∂x²
                        f_plus = follower_objective(x_final + eps, y_final)
                        f_minus = follower_objective(x_final - eps, y_final)
                        hessian[i, j] = (f_plus - 2*follower_objective(x_final, y_final) + f_minus) / eps**2
                    elif i == 1 and j == 1:  # ∂²/∂y²
                        f_plus = follower_objective(x_final, y_final + eps)
                        f_minus = follower_objective(x_final, y_final - eps)
                        hessian[i, j] = (f_plus - 2*follower_objective(x_final, y_final) + f_minus) / eps**2
                    else:  # 混合偏导
                        f_pp = follower_objective(x_final + eps, y_final + eps)
                        f_pm = follower_objective(x_final + eps, y_final - eps)
                        f_mp = follower_objective(x_final - eps, y_final + eps)
                        f_mm = follower_objective(x_final - eps, y_final - eps)
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            
            st.code(f"海森矩阵 H = {hessian}")
            
            # 计算条件数
            eigenvals = np.linalg.eigvals(hessian)
            condition_number = max(abs(eigenvals)) / min(abs(eigenvals)) if min(abs(eigenvals)) > 1e-10 else float('inf')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最大特征值", f"{max(abs(eigenvals)):.3f}")
            with col2:
                st.metric("条件数", f"{condition_number:.2e}")
        
        st.success("""
        **Stackelberg博弈的核心洞察**：
        - **层次优势**: Leader通过预判获得先动优势
        - **计算复杂度**: 需要计算隐函数梯度，复杂度较高
        - **应用场景**: 元学习、超参数优化、生成对抗网络
        - **工程挑战**: 海森矩阵求逆的数值稳定性
        """)
    
    @staticmethod
    def _render_multi_agent():
        """多智能体学习演示"""
        st.markdown("### 🤝 多智能体学习：LOLA算法与合作涌现")
        
        st.markdown(r"""
        **核心概念**：
        - **对手学习感知**: 不仅优化自己，还要塑造对手的学习过程
        - **LOLA算法**: $\Delta\theta_i = -\eta \nabla_{\theta_i} L_i - \beta \nabla_{\theta_i} L_j \frac{d\theta_j}{d\theta_i}$
        - **合作涌现**: 在囚徒困境中自发产生"针锋相对"策略
        - **塑造效应**: 通过梯度项影响对手的学习动态
        """)
        
        with st.sidebar:
            game_type = st.selectbox("博弈类型", 
                ["囚徒困境", "合作博弈", "协调博弈"])
            learning_rate = st.slider("学习率", 0.01, 0.2, 0.1, 0.01)
            lolalpha = st.slider("LOLA系数 α", 0.0, 0.5, 0.1, 0.01)
            episodes = st.slider("训练回合数", 100, 2000, 500, 100)
            show_shaping = st.checkbox("显示塑造效应", value=True)
        
        # 定义囚徒困境的收益矩阵
        if game_type == "囚徒困境":
            payoff_matrix = {
                'player1': np.array([[3, 0], [5, 1]]),  # 行：合作/背叛，列：合作/背叛
                'player2': np.array([[3, 5], [0, 1]])
            }
            strategies = ['合作', '背叛']
            
        elif game_type == "合作博弈":
            payoff_matrix = {
                'player1': np.array([[4, 1], [2, 3]]),
                'player2': np.array([[4, 2], [1, 3]])
            }
            strategies = ['合作', '背叛']
            
        else:  # 协调博弈
            payoff_matrix = {
                'player1': np.array([[2, 0], [0, 1]]),
                'player2': np.array([[1, 0], [0, 2]])
            }
            strategies = ['策略A', '策略B']
        
        # LOLA算法实现
        def lola_learning(payoff_matrix, learning_rate, lolalpha, episodes):
            # 策略参数（softmax）
            theta1 = np.random.randn(2) * 0.1
            theta2 = np.random.randn(2) * 0.1
            
            history = {
                'theta1': [],
                'theta2': [],
                'cooperation_rate': [],
                'payoffs': []
            }
            
            for episode in range(episodes):
                # 计算策略概率
                pi1 = np.exp(theta1) / np.sum(np.exp(theta1))
                pi2 = np.exp(theta2) / np.sum(np.exp(theta2))
                
                # 记录合作率（选择第一个策略的概率）
                history['cooperation_rate'].append(pi1[0])
                history['theta1'].append(theta1.copy())
                history['theta2'].append(theta2.copy())
                
                # 计算期望收益
                expected_payoff1 = np.sum(pi1[:, None] * pi2[None, :] * payoff_matrix['player1'])
                expected_payoff2 = np.sum(pi1[:, None] * pi2[None, :] * payoff_matrix['player2'])
                history['payoffs'].append([expected_payoff1, expected_payoff2])
                
                # 计算梯度（简化版本）
                grad1 = np.zeros(2)
                grad2 = np.zeros(2)
                
                for i in range(2):
                    for j in range(2):
                        # 基础梯度
                        grad1[i] += pi2[j] * payoff_matrix['player1'][i, j] * (1 - pi1[i])
                        grad2[j] += pi1[i] * payoff_matrix['player2'][i, j] * (1 - pi2[j])
                
                # LOLA修正项（塑造对手学习）
                if lolalpha > 0 and show_shaping:
                    # 简化的塑造项：考虑对手策略变化对自己收益的影响
                    shaping1 = lolalpha * np.sum(grad2) * (pi1 - 0.5)  # 塑造对手向合作
                    shaping2 = lolalpha * np.sum(grad1) * (pi2 - 0.5)
                    
                    grad1 += shaping1
                    grad2 += shaping2
                
                # 更新参数
                theta1 += learning_rate * grad1
                theta2 += learning_rate * grad2
            
            return history
        
        # 运行LOLA学习
        history = lola_learning(payoff_matrix, learning_rate, lolalpha, episodes)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "策略演化", "合作率变化",
                "收益变化", "策略空间轨迹"
            ]
        )
        
        # 策略演化
        theta1_history = np.array(history['theta1'])
        theta2_history = np.array(history['theta2'])
        
        fig.add_trace(
            go.Scatter(
                x=list(range(episodes)), y=theta1_history[:, 0],
                mode='lines',
                name='玩家1策略1',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(episodes)), y=theta2_history[:, 0],
                mode='lines',
                name='玩家2策略1',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 合作率变化
        cooperation_rates = history['cooperation_rate']
        fig.add_trace(
            go.Scatter(
                x=list(range(episodes)), y=cooperation_rates,
                mode='lines',
                name='合作率',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # 收益变化
        payoffs = np.array(history['payoffs'])
        fig.add_trace(
            go.Scatter(
                x=list(range(episodes)), y=payoffs[:, 0],
                mode='lines',
                name='玩家1收益',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(episodes)), y=payoffs[:, 1],
                mode='lines',
                name='玩家2收益',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # 策略空间轨迹
        if len(theta1_history) > 0 and len(theta2_history) > 0:
            pi1_history = np.exp(theta1_history) / np.sum(np.exp(theta1_history), axis=1, keepdims=True)
            pi2_history = np.exp(theta2_history) / np.sum(np.exp(theta2_history), axis=1, keepdims=True)
            
            fig.add_trace(
                go.Scatter(
                    x=pi1_history[:, 0], y=pi2_history[:, 0],
                    mode='lines+markers',
                    name='学习轨迹',
                    line=dict(color='purple', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=2
            )
            
            # 标记起点和终点
            if len(pi1_history) > 0 and len(pi2_history) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[pi1_history[0, 0]], y=[pi2_history[0, 0]],
                        mode='markers',
                        marker=dict(size=10, color='green', symbol='circle'),
                        name='起点',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[pi1_history[-1, 0]], y=[pi2_history[-1, 0]],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='star'),
                        name='终点',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f"LOLA算法 - {game_type}",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析结果
        st.markdown("### 📊 学习分析")
        
        # 安全检查数组访问
        if len(cooperation_rates) > 0 and len(payoffs) > 0:
            final_cooperation = cooperation_rates[-1]
            final_payoff1 = payoffs[-1, 0]
            final_payoff2 = payoffs[-1, 1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最终合作率", f"{final_cooperation:.3f}")
            with col2:
                st.metric("玩家1最终收益", f"{final_payoff1:.3f}")
            with col3:
                st.metric("玩家2最终收益", f"{final_payoff2:.3f}")
            with col4:
                if len(cooperation_rates) >= 100:
                    total_cooperation = np.mean(cooperation_rates[-100:])
                else:
                    total_cooperation = np.mean(cooperation_rates)
                st.metric("平均合作率", f"{total_cooperation:.3f}")
        else:
            st.error("数据不足，无法进行分析")
        
        # 塑造效应分析
        if show_shaping and lolalpha > 0:
            st.markdown("### 🎯 塑造效应分析")
            
            # 计算不同LOLA系数下的合作率
            alphas = [0.0, 0.05, 0.1, 0.2, 0.3]
            final_cooperations = []
            
            for alpha in alphas:
                hist_temp = lola_learning(payoff_matrix, learning_rate, alpha, 200)
                final_cooperations.append(hist_temp['cooperation_rate'][-1])
            
            fig_shaping = go.Figure()
            fig_shaping.add_trace(
                go.Scatter(
                    x=alphas,
                    y=final_cooperations,
                    mode='lines+markers',
                    name='合作率 vs LOLA系数',
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
            
            fig_shaping.update_layout(
                title="LOLA系数对合作率的影响",
                xaxis_title="LOLA系数 α",
                yaxis_title="最终合作率",
                height=400
            )
            
            st.plotly_chart(fig_shaping, use_container_width=True)
        
        st.success("""
        **LOLA算法的核心洞察**：
        - **超越自我**: 不仅优化自身，还要考虑对手的学习过程
        - **合作涌现**: 在适当条件下，自私智能体也能产生合作行为
        - **长期主义**: 短期牺牲可能带来长期收益
        - **应用前景**: 多智能体系统、算法博弈论、社交智能
        """)


# 为了兼容性，添加缺少的导入
try:
    from scipy.linalg import eig as scipy_eig
except ImportError:
    # 提供eig函数作为备选
    def eig(matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

        # 添加交互式测验
