"""
交互式最优传输理论可视化
严格按照 22.OptimalTransport.md 中的理论实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.optimize import linear_sum_assignment
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates


class InteractiveOptimalTransport:
    """交互式最优传输理论可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🚛 最优传输理论：从搬土问题到生成模型")
        st.markdown(r"""
        **核心思想**: 将概率分布视为几何对象，通过最优传输路径定义分布间的度量
        
        关键概念：
        - **Monge问题**: 寻找确定性映射 $T: \mathcal{X} \to \mathcal{Y}$
        - **Kantorovich松弛**: 引入耦合矩阵 $\pi(x,y)$
        - **Wasserstein距离**: $W_p(\mu, \nu) = (\inf_{\pi} \mathbb{E}_{(x,y) \sim \pi} [||x-y||^p])^{1/p}$
        - **Sinkhorn算法**: 熵正则化的GPU友好求解
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["Wasserstein距离", "传输问题", "Sinkhorn算法", "生成模型应用"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "Wasserstein距离":
            InteractiveOptimalTransport._render_wasserstein_distance()
        elif viz_type == "传输问题":
            InteractiveOptimalTransport._render_transport_problem()
        elif viz_type == "Sinkhorn算法":
            InteractiveOptimalTransport._render_sinkhorn()
        elif viz_type == "生成模型应用":
            InteractiveOptimalTransport._render_generative_models()
    

        # 添加交互式测验
        quiz_system = QuizSystem("optimal_transport")
        quizzes = QuizTemplates.get_optimal_transport_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod

        # 添加交互式测验

    def _render_wasserstein_distance():
        """Wasserstein距离可视化"""
        st.markdown("### 📏 Wasserstein距离：几何视角的分布度量")
        
        st.markdown("""
        **为什么需要Wasserstein距离？**
        - **KL散度问题**: 当分布互不重叠时，$D_{KL}(P||Q) = +\infty$
        - **梯度消失**: KL散度无法提供有效的梯度信号
        - **Wasserstein优势**: 即使分布分离，仍能提供有意义的距离和梯度
        """)
        
        with st.sidebar:
            distance_type = st.selectbox("距离类型", 
                ["1-Wasserstein", "2-Wasserstein", "KL散度对比"])
            distribution_type = st.selectbox("分布类型", 
                ["高斯分布", "均匀分布", "混合高斯"])
            separation = st.slider("分布分离程度", 0.0, 5.0, 2.0, 0.1)
            show_contours = st.checkbox("显示等高线", value=True)
        
        # 生成分布数据
        np.random.seed(42)
        n_samples = 1000
        
        if distribution_type == "高斯分布":
            # 两个高斯分布
            mu1 = np.array([-separation, 0])
            mu2 = np.array([separation, 0])
            cov = np.eye(2) * 0.5
            
            samples1 = np.random.multivariate_normal(mu1, cov, n_samples)
            samples2 = np.random.multivariate_normal(mu2, cov, n_samples)
            
        elif distribution_type == "均匀分布":
            # 两个均匀分布（矩形）
            x1 = np.random.uniform(-separation-1, -separation+1, n_samples)
            y1 = np.random.uniform(-1, 1, n_samples)
            samples1 = np.column_stack([x1, y1])
            
            x2 = np.random.uniform(separation-1, separation+1, n_samples)
            y2 = np.random.uniform(-1, 1, n_samples)
            samples2 = np.column_stack([x2, y2])
            
        else:  # 混合高斯
            # 第一个分布：两个高斯混合
            mix1_samples = n_samples // 2
            samples1_part1 = np.random.multivariate_normal([-separation-1, -1], 0.3*np.eye(2), mix1_samples)
            samples1_part2 = np.random.multivariate_normal([-separation+1, 1], 0.3*np.eye(2), n_samples-mix1_samples)
            samples1 = np.vstack([samples1_part1, samples1_part2])
            
            # 第二个分布：两个高斯混合
            samples2_part1 = np.random.multivariate_normal([separation-1, -1], 0.3*np.eye(2), mix1_samples)
            samples2_part2 = np.random.multivariate_normal([separation+1, 1], 0.3*np.eye(2), n_samples-mix1_samples)
            samples2 = np.vstack([samples2_part1, samples2_part2])
        
        # 计算距离
        if distance_type in ["1-Wasserstein", "2-Wasserstein"]:
            # 简化的Wasserstein距离计算（使用质心距离作为近似）
            p = 1 if distance_type == "1-Wasserstein" else 2
            centroid1 = np.mean(samples1, axis=0)
            centroid2 = np.mean(samples2, axis=0)
            w_distance = np.linalg.norm(centroid1 - centroid2) ** p
            if distance_type == "2-Wasserstein":
                w_distance = np.sqrt(w_distance)
        else:
            # KL散度（简化计算）
            def estimate_kl(samples1, samples2, bandwidth=0.5):

        # 添加交互式测验
                from scipy.stats import gaussian_kde
                
                kde1 = gaussian_kde(samples1.T, bw_method=bandwidth)
                kde2 = gaussian_kde(samples2.T, bw_method=bandwidth)
                
                # 在样本点上计算
                log_ratio = np.log(kde1(samples1.T) + 1e-10) - np.log(kde2(samples1.T) + 1e-10)
                return np.mean(log_ratio)
            
            try:
                w_distance = estimate_kl(samples1, samples2)
                if np.isnan(w_distance) or np.isinf(w_distance):
                    w_distance = 999.0  # 表示无穷大
            except:
                w_distance = 999.0
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "分布可视化", "距离度量对比",
                "传输路径示意", "距离随分离程度变化"
            ]
        )
        
        # 分布可视化
        fig.add_trace(
            go.Scatter(
                x=samples1[:, 0], y=samples1[:, 1],
                mode='markers',
                name='分布 μ',
                marker=dict(color='red', size=4, opacity=0.6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=samples2[:, 0], y=samples2[:, 1],
                mode='markers',
                name='分布 ν',
                marker=dict(color='blue', size=4, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # 添加质心
        fig.add_trace(
            go.Scatter(
                x=[np.mean(samples1[:, 0])], y=[np.mean(samples1[:, 1])],
                mode='markers',
                name='μ质心',
                marker=dict(color='darkred', size=10, symbol='x')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[np.mean(samples2[:, 0])], y=[np.mean(samples2[:, 1])],
                mode='markers',
                name='ν质心',
                marker=dict(color='darkblue', size=10, symbol='x')
            ),
            row=1, col=1
        )
        
        # 距离对比
        distances = {
            'Wasserstein': w_distance if "Wasserstein" in distance_type else np.linalg.norm(centroid1 - centroid2),
            'KL散度': w_distance if distance_type == "KL散度对比" else 0,
            '欧氏距离': np.linalg.norm(centroid1 - centroid2)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(distances.keys()),
                y=list(distances.values()),
                marker_color=['green', 'red', 'orange']
            ),
            row=1, col=2
        )
        
        # 传输路径示意
        n_paths = 20
        indices1 = np.random.choice(len(samples1), n_paths, replace=False)
        indices2 = np.random.choice(len(samples2), n_paths, replace=False)
        
        for i in range(min(n_paths, 10)):  # 限制显示数量
            fig.add_trace(
                go.Scatter(
                    x=[samples1[indices1[i], 0], samples2[indices2[i], 0]],
                    y=[samples1[indices1[i], 1], samples2[indices2[i], 1]],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 距离随分离程度变化
        separations = np.linspace(0, 5, 20)
        w_distances = []
        kl_distances = []
        
        for sep in separations:
            # 重新生成数据
            mu1_temp = np.array([-sep, 0])
            mu2_temp = np.array([sep, 0])
            
            samples1_temp = np.random.multivariate_normal(mu1_temp, cov, 200)
            samples2_temp = np.random.multivariate_normal(mu2_temp, cov, 200)
            
            # Wasserstein距离
            w_dist_temp = np.linalg.norm(mu1_temp - mu2_temp)
            w_distances.append(w_dist_temp)
            
            # KL散度（简化）
            overlap = np.exp(-sep**2 / (4 * 0.5))  # 高斯重叠度
            kl_dist = sep**2 / (2 * 0.5) if overlap > 0.01 else 999
            kl_distances.append(min(kl_dist, 50))  # 限制显示范围
        
        fig.add_trace(
            go.Scatter(
                x=separations, y=w_distances,
                mode='lines',
                name='Wasserstein',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=separations, y=kl_distances,
                mode='lines',
                name='KL散度',
                line=dict(color='red', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"{distance_type} vs KL散度对比分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 距离度量详细分析")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if "Wasserstein" in distance_type:
                st.metric(f"{distance_type}", f"{w_distance:.3f}")
            else:
                st.metric("当前距离", f"{w_distance:.3f}")
        
        with col2:
            euclidean_dist = np.linalg.norm(centroid1 - centroid2)
            st.metric("欧氏距离", f"{euclidean_dist:.3f}")
        
        with col3:
            overlap_score = np.exp(-separation**2 / 2)  # 简化的重叠度
            st.metric("分布重叠", f"{overlap_score:.3f}")
        
        if distance_type == "KL散度对比" and w_distance > 100:
            st.error("🚨 KL散度趋于无穷大！分布重叠太少，导致梯度消失")
        else:
            st.success("✅ 距离度量正常，可以提供有效的梯度信号")
        
        st.info("""
        **Wasserstein距离的优势**：
        - **连续性**: 即使分布不重叠，距离仍然有限
        - **几何意义**: 反映分布间的物理"搬运成本"
        - **梯度友好**: 提供稳定的梯度信号
        - **理论保证**: 满足距离公理
        """)
    
    @staticmethod
    def _render_transport_problem():
        """传输问题演示"""
        st.markdown("### 🚛 最优传输问题：从Monge到Kantorovich")
        
        st.markdown("""
        **问题演进**：
        - **Monge问题 (1781)**: 寻找确定性映射 $T: \mathcal{X} \to \mathcal{Y}$
        - **Kantorovich松弛 (1942)**: 引入耦合矩阵 $\pi(x,y)$，允许概率分配
        - **约束条件**: 行和为源分布，列和为目标分布，$\pi \ge 0$
        """)
        
        with st.sidebar:
            problem_type = st.selectbox("问题类型", 
                ["Monge映射", "Kantorovich耦合", "对比分析"])
            n_sources = st.slider("源点数量", 5, 15, 8, 1)
            n_targets = st.slider("目标点数量", 5, 15, 8, 1)
            cost_function = st.selectbox("代价函数", 
                ["欧氏距离", "平方距离", "曼哈顿距离"])
            show_matrix = st.checkbox("显示传输矩阵", value=True)
        
        # 生成数据
        np.random.seed(42)
        
        # 源分布点
        sources = np.random.randn(n_sources, 2) * 2
        source_weights = np.ones(n_sources) / n_sources
        
        # 目标分布点
        targets = np.random.randn(n_targets, 2) * 2 + 3  # 偏移
        target_weights = np.ones(n_targets) / n_targets
        
        # 计算代价矩阵
        if cost_function == "欧氏距离":
            C = np.sqrt(((sources[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2))
        elif cost_function == "平方距离":
            C = ((sources[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
        else:  # 曼哈顿距离
            C = np.abs(sources[:, None, :] - targets[None, :, :]).sum(axis=2)
        
        # 求解传输问题
        if problem_type == "Monge映射":
            # 简化：使用匈牙利算法（一对一映射）
            row_ind, col_ind = linear_sum_assignment(C)
            transport_matrix = np.zeros_like(C)
            transport_matrix[row_ind, col_ind] = 1.0 / min(n_sources, n_targets)
            
            # 计算总代价
            total_cost = np.sum(C[row_ind, col_ind]) / min(n_sources, n_targets)
            
        elif problem_type == "Kantorovich耦合":
            # 使用简化的Sinkhorn算法
            epsilon = 0.1
            K = np.exp(-C / epsilon)
            
            u = np.ones(n_sources)
            v = np.ones(n_targets)
            
            for _ in range(50):
                u = source_weights / (K @ v + 1e-8)
                v = target_weights / (K.T @ u + 1e-8)
            
            transport_matrix = np.diag(u) @ K @ np.diag(v)
            total_cost = np.sum(transport_matrix * C)
            
        else:  # 对比分析
            # Monge解
            row_ind, col_ind = linear_sum_assignment(C)
            monge_matrix = np.zeros_like(C)
            monge_matrix[row_ind, col_ind] = 1.0 / min(n_sources, n_targets)
            monge_cost = np.sum(C[row_ind, col_ind]) / min(n_sources, n_targets)
            
            # Kantorovich解
            epsilon = 0.1
            K = np.exp(-C / epsilon)
            u = np.ones(n_sources)
            v = np.ones(n_targets)
            
            for _ in range(50):
                u = source_weights / (K @ v + 1e-8)
                v = target_weights / (K.T @ u + 1e-8)
            
            transport_matrix = np.diag(u) @ K @ np.diag(v)
            total_cost = np.sum(transport_matrix * C)
            kantorovich_cost = total_cost
        
        # 可视化
        if problem_type == "对比分析":
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Monge映射 (一对一)", "Kantorovich耦合 (多对多)",
                    "传输矩阵对比", "代价对比"
                ]
            )
            
            # Monge映射可视化
            fig.add_trace(
                go.Scatter(
                    x=sources[:, 0], y=sources[:, 1],
                    mode='markers',
                    name='源点',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=targets[:, 0], y=targets[:, 1],
                    mode='markers',
                    name='目标点',
                    marker=dict(color='blue', size=8)
                ),
                row=1, col=1
            )
            
            # Monge连线
            for i, j in zip(row_ind, col_ind):
                fig.add_trace(
                    go.Scatter(
                        x=[sources[i, 0], targets[j, 0]],
                        y=[sources[i, 1], targets[j, 1]],
                        mode='lines',
                        line=dict(width=2, color='green'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Kantorovich耦合可视化
            fig.add_trace(
                go.Scatter(
                    x=sources[:, 0], y=sources[:, 1],
                    mode='markers',
                    name='源点',
                    marker=dict(color='red', size=8),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=targets[:, 0], y=targets[:, 1],
                    mode='markers',
                    name='目标点',
                    marker=dict(color='blue', size=8),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Kantorovich连线（只显示主要的）
            threshold = np.percentile(transport_matrix, 80)
            for i in range(n_sources):
                for j in range(n_targets):
                    if transport_matrix[i, j] > threshold:
                        fig.add_trace(
                            go.Scatter(
                                x=[sources[i, 0], targets[j, 0]],
                                y=[sources[i, 1], targets[j, 1]],
                                mode='lines',
                                line=dict(width=1, color='orange'),
                    marker=dict(opacity=0.5),
                                showlegend=False
                            ),
                            row=1, col=2
                        )
            
            # 传输矩阵对比
            fig.add_trace(
                go.Heatmap(
                    z=monge_matrix,
                    colorscale='Reds',
                    name='Monge'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=transport_matrix,
                    colorscale='Blues',
                    name='Kantorovich'
                ),
                row=2, col=2
            )
            
            # 代价对比
            fig.add_trace(
                go.Bar(
                    x=['Monge', 'Kantorovich'],
                    y=[monge_cost, kantorovich_cost],
                    marker_color=['red', 'blue']
                ),
                row=2, col=2
            )
            
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "传输方案可视化", "传输矩阵热力图"
                ]
            )
            
            # 传输方案可视化
            fig.add_trace(
                go.Scatter(
                    x=sources[:, 0], y=sources[:, 1],
                    mode='markers',
                    name='源点',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=targets[:, 0], y=targets[:, 1],
                    mode='markers',
                    name='目标点',
                    marker=dict(color='blue', size=8)
                ),
                row=1, col=1
            )
            
            # 传输连线
            if problem_type == "Monge映射":
                for i, j in zip(row_ind, col_ind):
                    fig.add_trace(
                        go.Scatter(
                            x=[sources[i, 0], targets[j, 0]],
                            y=[sources[i, 1], targets[j, 1]],
                            mode='lines',
                            line=dict(width=2, color='green'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            else:  # Kantorovich
                threshold = np.percentile(transport_matrix, 70)
                for i in range(n_sources):
                    for j in range(n_targets):
                        if transport_matrix[i, j] > threshold:
                            fig.add_trace(
                                go.Scatter(
                                    x=[sources[i, 0], targets[j, 0]],
                                    y=[sources[i, 1], targets[j, 1]],
                                    mode='lines',
                                    line=dict(width=transport_matrix[i, j]*5, 
                                            color='orange', opacity=0.7),
                                    showlegend=False
                                ),
                                row=1, col=1
                            )
            
            # 传输矩阵
            fig.add_trace(
                go.Heatmap(
                    z=transport_matrix,
                    colorscale='Viridis',
                    showscale=True
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"{problem_type} - 最优传输方案",
            height=500 if problem_type != "对比分析" else 600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 传输方案分析")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总传输代价", f"{total_cost:.3f}")
        with col2:
            st.metric("源点数量", n_sources)
        with col3:
            st.metric("目标点数量", n_targets)
        
        if show_matrix:
            st.markdown("### 📈 传输矩阵详情")
            st.dataframe(pd.DataFrame(transport_matrix, 
                                    index=[f"源{i}" for i in range(n_sources)],
                                    columns=[f"目标{j}" for j in range(n_targets)]))
        
        if problem_type == "对比分析":
            st.markdown("### 🔍 Monge vs Kantorovich 对比")
            
            comparison_data = [
                ["Monge映射", "确定性一对一映射", "简单但受限", "当源目标数量不等时无解"],
                ["Kantorovich", "概率耦合矩阵", "灵活且通用", "适用于所有情况，计算稍复杂"]
            ]
            
            st.table(pd.DataFrame(comparison_data, 
                                columns=["方法", "特点", "优势", "限制"]))
        
        st.success("""
        **最优传输的核心价值**：
        - **统一框架**: 将各种度量问题统一为传输优化
        - **几何直观**: 提供物理世界的搬运类比
        - **计算可行**: 通过松弛和正则化实现高效计算
        """)
    
    @staticmethod
    def _render_sinkhorn():
        """Sinkhorn算法演示"""
        st.markdown("### ⚡ Sinkhorn算法：熵正则化的GPU友好求解")
        
        st.markdown("""
        **核心思想**：
        - **熵正则化**: $\min_{\pi} \langle C, \pi \rangle - \epsilon H(\pi)$
        - **形式解**: $\pi_{ij} = u_i e^{-C_{ij}/\epsilon} v_j$ (类似Softmax)
        - **交替迭代**: 行归一化 → 列归一化 → 收敛
        """)
        
        with st.sidebar:
            epsilon = st.slider("熵正则化系数 ε", 0.01, 1.0, 0.1, 0.01)
            max_iter = st.slider("最大迭代次数", 10, 200, 50, 10)
            n_points = st.slider("点数量", 5, 20, 10, 1)
            show_convergence = st.checkbox("显示收敛过程", value=True)
            show_animation = st.checkbox("显示迭代动画", value=True)
        
        # 生成数据
        np.random.seed(42)
        
        # 源分布和目标分布
        sources = np.random.randn(n_points, 2) * 1.5 - 2
        targets = np.random.randn(n_points, 2) * 1.5 + 2
        
        # 权重（均匀分布）
        source_weights = np.ones(n_points) / n_points
        target_weights = np.ones(n_points) / n_points
        
        # 计算代价矩阵
        C = ((sources[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
        
        # Sinkhorn算法
        def sinkhorn_algorithm(C, mu, nu, epsilon, max_iter):
            K = np.exp(-C / epsilon)
            u = np.ones(n_points)
            v = np.ones(n_points)
            
            history = []
            
            for iteration in range(max_iter):
                # 行归一化
                u_new = mu / (K @ v + 1e-8)
                # 列归一化
                v_new = nu / (K.T @ u_new + 1e-8)
                
                # 计算传输矩阵
                P = np.diag(u_new) @ K @ np.diag(v_new)
                
                # 记录历史
                cost = np.sum(P * C)
                row_error = np.max(np.abs(P.sum(axis=1) - mu))
                col_error = np.max(np.abs(P.sum(axis=0) - nu))
                
                history.append({
                    'iteration': iteration,
                    'cost': cost,
                    'row_error': row_error,
                    'col_error': col_error,
                    'P': P.copy()
                })
                
                u, v = u_new, v_new
            
            return history
        
        # 运行算法
        history = sinkhorn_algorithm(C, source_weights, target_weights, epsilon, max_iter)
        
        # 可视化
        if show_animation:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    "初始状态", "迭代中", "最终结果",
                    "代价收敛", "约束误差", "熵正则化效应"
                ]
            )
            
            # 选择几个关键迭代
            key_iterations = [0, max_iter//4, max_iter//2, -1]
            
            for idx, iter_idx in enumerate(key_iterations[:3]):
                P = history[iter_idx]['P']
                
                # 点分布
                fig.add_trace(
                    go.Scatter(
                        x=sources[:, 0], y=sources[:, 1],
                        mode='markers',
                        name='源点' if idx == 0 else '',
                        marker=dict(color='red', size=8),
                        showlegend=(idx == 0)
                    ),
                    row=1, col=idx+1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=targets[:, 0], y=targets[:, 1],
                        mode='markers',
                        name='目标点' if idx == 0 else '',
                        marker=dict(color='blue', size=8),
                        showlegend=False
                    ),
                    row=1, col=idx+1
                )
                
                # 传输连线
                threshold = np.percentile(P, 60)
                for i in range(n_points):
                    for j in range(n_points):
                        if P[i, j] > threshold:
                            fig.add_trace(
                    go.Scatter(
                        x=[sources[i, 0], targets[j, 0]],
                        y=[sources[i, 1], targets[j, 1]],
                        mode='lines',
                        line=dict(width=P[i, j]*3, color='green'),
                        marker=dict(opacity=0.6),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 收敛曲线
            iterations = [h['iteration'] for h in history]
            costs = [h['cost'] for h in history]
            row_errors = [h['row_error'] for h in history]
            col_errors = [h['col_error'] for h in history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations, y=costs,
                    mode='lines',
                    name='传输代价',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=iterations, y=row_errors,
                    mode='lines',
                    name='行约束误差',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=iterations, y=col_errors,
                    mode='lines',
                    name='列约束误差',
                    line=dict(color='green', width=2)
                ),
                row=2, col=2
            )
            
            # 熵正则化效应
            epsilons = np.linspace(0.01, 1.0, 20)
            final_costs = []
            
            for eps in epsilons:
                hist = sinkhorn_algorithm(C, source_weights, target_weights, eps, 50)
                final_costs.append(hist[-1]['cost'])
            
            fig.add_trace(
                go.Scatter(
                    x=epsilons, y=final_costs,
                    mode='lines',
                    name='ε vs 最终代价',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=3
            )
            
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "最终传输方案", "收敛过程"
                ]
            )
            
            # 最终传输方案
            P_final = history[-1]['P']
            
            fig.add_trace(
                go.Scatter(
                    x=sources[:, 0], y=sources[:, 1],
                    mode='markers',
                    name='源点',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=targets[:, 0], y=targets[:, 1],
                    mode='markers',
                    name='目标点',
                    marker=dict(color='blue', size=8)
                ),
                row=1, col=1
            )
            
            # 传输连线
            threshold = np.percentile(P_final, 70)
            for i in range(n_points):
                for j in range(n_points):
                    if P_final[i, j] > threshold:
                        fig.add_trace(
                            go.Scatter(
                                x=[sources[i, 0], targets[j, 0]],
                                y=[sources[i, 1], targets[j, 1]],
                                mode='lines',
                                line=dict(width=P_final[i, j]*5, color='green'),
                    marker=dict(opacity=0.7),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
            
            # 收敛过程
            iterations = [h['iteration'] for h in history]
            costs = [h['cost'] for h in history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations, y=costs,
                    mode='lines',
                    name='传输代价',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"Sinkhorn算法演示 (ε={epsilon})",
            height=400 if not show_animation else 600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 算法性能分析")
        
        final_result = history[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最终代价", f"{final_result['cost']:.4f}")
        with col2:
            st.metric("行约束误差", f"{final_result['row_error']:.6f}")
        with col3:
            st.metric("列约束误差", f"{final_result['col_error']:.6f}")
        with col4:
            st.metric("迭代次数", len(history))
        
        # 熵正则化分析
        st.markdown("### 🌡️ 熵正则化效应分析")
        
        if epsilon < 0.1:
            st.warning("⚠️ ε较小：传输方案更精确，但可能收敛较慢")
        elif epsilon > 0.5:
            st.info("ℹ️ ε较大：传输方案更平滑，但精度略低")
        else:
            st.success("✅ ε适中：在精度和稳定性间取得平衡")
        
        st.success("""
        **Sinkhorn算法的优势**：
        - **GPU友好**: 纯矩阵运算，可并行化
        - **收敛保证**: 熵正则化确保凸性
        - **Softmax联系**: 与深度学习的激活函数相关
        - **Attention机制**: Transformer的理论基础
        """)
    
    @staticmethod
    def _render_generative_models():
        """生成模型应用"""
        st.markdown("### 🎨 生成模型应用：从WGAN到Flow Matching")
        
        st.markdown("""
        **核心应用**：
        - **WGAN**: 利用Wasserstein距离解决模式崩溃和梯度消失
        - **Flow Matching**: 沿最优传输路径的确定性生成
        - **扩散模型**: 随机路径 vs 最优路径的对比
        """)
        
        with st.sidebar:
            application = st.selectbox("应用类型", 
                ["WGAN原理", "Flow Matching", "路径对比"])
            n_samples = st.slider("样本数量", 100, 1000, 500, 50)
            noise_level = st.slider("噪声水平", 0.1, 2.0, 0.5, 0.1)
            show_paths = st.checkbox("显示生成路径", value=True)
        
        # 生成数据
        np.random.seed(42)
        
        # 简化的2D数据分布（如月牙、环形等）
        theta = np.linspace(0, 2*np.pi, n_samples)
        r = 2 + np.random.normal(0, noise_level, n_samples)
        
        data_x = r * np.cos(theta)
        data_y = r * np.sin(theta) + np.random.normal(0, noise_level, n_samples)
        
        data_samples = np.column_stack([data_x, data_y])
        noise_samples = np.random.randn(n_samples, 2)
        
        # 计算数据中心（用于所有分支）
        center_data = np.mean(data_samples, axis=0)
        center_noise = np.mean(noise_samples, axis=0)
        
        # 可视化
        if application == "WGAN原理":
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "真实数据分布", "噪声分布", 
                    "Wasserstein梯度", "KL散度梯度"
                ]
            )
            
            # 真实数据
            fig.add_trace(
                go.Scatter(
                    x=data_samples[:, 0], y=data_samples[:, 1],
                    mode='markers',
                    name='真实数据',
                    marker=dict(color='blue', size=4, opacity=0.6)
                ),
                row=1, col=1
            )
            
            # 噪声数据
            fig.add_trace(
                go.Scatter(
                    x=noise_samples[:, 0], y=noise_samples[:, 1],
                    mode='markers',
                    name='噪声数据',
                    marker=dict(color='red', size=4, opacity=0.6)
                ),
                row=1, col=2
            )
            
            # Wasserstein梯度（简化示意）
            gradient_direction = center_data - center_noise
            
            # 显示梯度方向
            fig.add_trace(
                go.Scatter(
                    x=[center_noise[0], center_noise[0] + gradient_direction[0]],
                    y=[center_noise[1], center_noise[1] + gradient_direction[1]],
                    mode='lines+markers',
                    name='W梯度',
                    line=dict(width=3, color='green'),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
            
            # KL散度梯度（示意：局部梯度）
            fig.add_trace(
                go.Scatter(
                    x=[center_noise[0], center_noise[0] + 0.1],
                    y=[center_noise[1], center_noise[1] + 0.1],
                    mode='lines+markers',
                    name='KL梯度',
                    line=dict(width=3, color='orange', dash='dash'),
                    marker=dict(size=8)
                ),
                row=2, col=2
            )
            
        elif application == "Flow Matching":
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "最优传输路径", "速度场可视化"
                ]
            )
            
            # 最优传输路径（线性插值）
            n_paths = min(20, n_samples)
            indices = np.random.choice(n_samples, n_paths, replace=False)
            
            for idx in indices:
                # 线性插值路径
                t_values = np.linspace(0, 1, 10)
                path_x = (1-t_values) * noise_samples[idx, 0] + t_values * data_samples[idx, 0]
                path_y = (1-t_values) * noise_samples[idx, 1] + t_values * data_samples[idx, 1]
                
                fig.add_trace(
                    go.Scatter(
                        x=path_x, y=path_y,
                        mode='lines',
                        line=dict(width=1, color='green'),
                        marker=dict(opacity=0.5),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 起点和终点
            fig.add_trace(
                go.Scatter(
                    x=noise_samples[indices, 0], y=noise_samples[indices, 1],
                    mode='markers',
                    name='起点(噪声)',
                    marker=dict(color='red', size=6)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data_samples[indices, 0], y=data_samples[indices, 1],
                    mode='markers',
                    name='终点(数据)',
                    marker=dict(color='blue', size=6)
                ),
                row=1, col=1
            )
            
            # 速度场可视化
            # 创建网格
            x_range = np.linspace(-4, 4, 15)
            y_range = np.linspace(-4, 4, 15)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            
            # 简化的速度场（指向数据中心）
            Vx = center_data[0] - X_grid
            Vy = center_data[1] - Y_grid
            
            fig.add_trace(
                go.Scatter(
                    x=X_grid.flatten(), y=Y_grid.flatten(),
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=np.sqrt(Vx.flatten()**2 + Vy.flatten()**2),
                        colorscale='Viridis',
                        showscale=True
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
            
        else:  # 路径对比
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "扩散模型路径(随机)", "最优传输路径(确定)"
                ]
            )
            
            # 扩散模型路径（随机游走）
            n_paths = min(15, n_samples)
            indices = np.random.choice(n_samples, n_paths, replace=False)
            
            for idx in indices:
                # 随机路径模拟
                t_values = np.linspace(0, 1, 10)
                path_x = []
                path_y = []
                
                current = noise_samples[idx].copy()
                for t in t_values:
                    path_x.append(current[0])
                    path_y.append(current[1])
                    # 添加随机扰动
                    current += 0.3 * (data_samples[idx] - current) * 0.1 + 0.1 * np.random.randn(2)
                
                fig.add_trace(
                    go.Scatter(
                        x=path_x, y=path_y,
                        mode='lines',
                        line=dict(width=1, color='red'),
                        marker=dict(opacity=0.5),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 最优传输路径
            for idx in indices:
                t_values = np.linspace(0, 1, 10)
                path_x = (1-t_values) * noise_samples[idx, 0] + t_values * data_samples[idx, 0]
                path_y = (1-t_values) * noise_samples[idx, 1] + t_values * data_samples[idx, 1]
                
                fig.add_trace(
                    go.Scatter(
                        x=path_x, y=path_y,
                        mode='lines',
                        line=dict(width=1, color='green'),
                        marker=dict(opacity=0.5),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            title=f"{application} - 生成模型可视化",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 理论解释
        st.markdown("### 📚 理论基础")
        
        if application == "WGAN原理":
            st.latex(r"""
            W_1(P_{data}, P_G) = \sup_{||f||_L \le 1} \left( \mathbb{E}_{x \sim P_{data}}[f(x)] - \mathbb{E}_{z \sim P_z}[f(G(z))] \right)
            """)
            
            st.markdown("""
            **WGAN的核心思想**：
            - **判别器角色**: 从二分类器变为Lipschitz约束的回归器
            - **梯度稳定**: 即使分布不重叠，仍有有效梯度
            - **模式崩溃**: 通过Wasserstein距离的自然特性缓解
            """)
            
        elif application == "Flow Matching":
            st.latex(r"""
            v_t(x_t) = \frac{d}{dt} x_t = x_1 - x_0
            """)
            
            st.markdown("""
            **Flow Matching的优势**：
            - **直线路径**: 沿最优传输测地线演化
            - **确定生成**: 避免扩散模型的随机性
            - **快速收敛**: 比传统扩散模型更高效
            """)
        
        else:
            st.markdown("""
            **路径对比分析**：
            - **扩散模型**: 随机游走路径，探索性强但效率低
            - **最优传输**: 确定性最短路径，高效但需要精确配对
            - **实际应用**: 现代方法结合两者优势
            """)
        
        st.success("""
        **最优传输在生成模型中的价值**：
        - **理论指导**: 为生成模型提供几何直觉
        - **算法优化**: 启发更高效的训练和采样方法
        - **性能提升**: 解决模式崩溃和梯度消失问题
        """)


# 为了兼容性，添加缺少的导入
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    st.error("❌ SciPy库未安装，请运行: pip install scipy")

        # 添加交互式测验
