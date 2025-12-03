"""
交互式核回归与注意力机制可视化
严格按照 13.KernelRegression.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import cdist
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation
from common.quiz_system import QuizSystem, QuizTemplates

warnings.filterwarnings('ignore')


class InteractiveKernelRegression:
    """交互式核回归与注意力机制可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎯 注意力机制的数学本质：核回归、几何与秩")
        st.markdown("""
        **核心思想**: Attention 是 Nadaraya-Watson 核回归的现代变体，是可微的字典查询
        
        关键概念：
        - **核回归本质**: $\\hat{{v}} = \\sum_i \\frac{{K(q,k_i)}}{{\\sum_j K(q,k_j)}} v_i$
        - **缩放因子**: $\\sqrt{{d_k}}$ 用于方差稳定
        - **多头机制**: 突破单次投影的秩亏问题
        - **低秩瓶颈**: $Rank(QK^T) \\leq d_k < N$
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["核回归基础", "注意力机制", "多头注意力", "线性Attention"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "核回归基础":
            InteractiveKernelRegression._render_kernel_regression()
        elif viz_type == "注意力机制":
            InteractiveKernelRegression._render_attention_mechanism()
        elif viz_type == "多头注意力":
            InteractiveKernelRegression._render_multi_head_attention()
        elif viz_type == "线性Attention":
            InteractiveKernelRegression._render_linear_attention()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("kernel_regression")
        quizzes = QuizTemplates.get_kernel_regression_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_kernel_regression():
        """核回归基础演示"""
        st.markdown("### 📈 核回归基础：Nadaraya-Watson 估计器")
        
        st.latex(r"""
        \hat{v} = \sum_{i=1}^{n} \frac{K(q, k_i)}{\sum_{j=1}^{n} K(q, k_j)} v_i
        """)
        
        with st.sidebar:
            kernel_type = st.selectbox("核函数类型", 
                ["高斯核", "拉普拉斯核", "多项式核"])
            bandwidth = st.slider("带宽 h", 0.1, 2.0, 0.5, 0.1)
            num_points = st.slider("数据点数量", 10, 50, 20, 5)
            noise_level = st.slider("噪声水平", 0.0, 1.0, 0.2, 0.1)
        
        # 生成数据
        np.random.seed(42)
        
        # 真实函数 (正弦波 + 多项式)
        x_true = np.linspace(-3, 3, 200)
        y_true = np.sin(x_true) + 0.5 * x_true**2
        
        # 训练数据
        x_train = np.sort(np.random.uniform(-3, 3, num_points))
        y_train = np.sin(x_train) + 0.5 * x_train**2 + np.random.normal(0, noise_level, num_points)
        
        # 核函数
        def kernel_function(x1, x2):
            if kernel_type == "高斯核":
                return np.exp(-((x1 - x2)**2 / (2 * bandwidth**2)))
            elif kernel_type == "拉普拉斯核":
                return np.exp(-np.abs(x1 - x2) / bandwidth)
            else:  # 多项式核
                return (1 + x1 * x2) ** 3
        
        # Nadaraya-Watson 估计
        def nadaraya_watson(x_query, x_train, y_train):
            weights = np.array([kernel_function(x_query, xi) for xi in x_train])
            weights = weights / np.sum(weights)  # 归一化
            return np.sum(weights * y_train)
        
        # 预测
        y_pred = np.array([nadaraya_watson(x, x_train, y_train) for x in x_true])
        
        # 可视化
        fig = go.Figure()
        
        # 真实函数
        fig.add_trace(go.Scatter(
            x=x_true, y=y_true,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='真实函数'
        ))
        
        # 训练数据
        fig.add_trace(go.Scatter(
            x=x_train, y=y_train,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='训练数据'
        ))
        
        # 核回归预测
        fig.add_trace(go.Scatter(
            x=x_true, y=y_pred,
            mode='lines',
            line=dict(color='red', width=2),
            name='核回归预测'
        ))
        
        fig.update_layout(
            title=f"核回归估计 ({kernel_type}, h={bandwidth})",
            xaxis_title="x",
            yaxis_title="y",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 核函数可视化
        st.markdown("### 🎯 核函数形状")
        
        # 选择一个查询点
        query_point = 0.0
        kernel_values = np.array([kernel_function(query_point, xi) for xi in x_train])
        
        fig_kernel = go.Figure()
        
        # 核函数值
        fig_kernel.add_trace(go.Scatter(
            x=x_train, y=kernel_values,
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=6),
            name='核函数值'
        ))
        
        # 查询点
        fig_kernel.add_trace(go.Scatter(
            x=[query_point], y=[0],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='查询点'
        ))
        
        fig_kernel.update_layout(
            title=f"核函数形状 (查询点 x={query_point})",
            xaxis_title="训练点 x",
            yaxis_title="核函数值 K(q,x_i)",
            height=400
        )
        
        st.plotly_chart(fig_kernel, use_container_width=True)
        
        # 权重分布
        st.markdown("### ⚖️ 权重分布分析")
        
        weights_normalized = kernel_values / np.sum(kernel_values)
        
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            x=x_train,
            y=weights_normalized,
            marker_color='lightblue',
            name='归一化权重'
        ))
        
        fig_weights.update_layout(
            title="训练点权重分布",
            xaxis_title="训练点",
            yaxis_title="权重",
            height=300
        )
        
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # 误差分析
        mse = np.mean((y_pred - y_true)**2)
        st.metric("均方误差 (MSE)", f"{mse:.4f}")
        
        st.info("""
        **关键洞察**：
        - 核函数决定了相似度的度量方式
        - 带宽控制了局部性：小带宽=局部，大带宽=全局
        - 权重和为1，形成概率分布
        - 这是Attention机制的数学基础
        """)
    
    @staticmethod
    def _render_attention_mechanism():
        """注意力机制演示"""
        st.markdown("### 🧠 注意力机制：从核回归到动态权重")
        
        st.latex(r"""
        Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
        """)
        
        with st.sidebar:
            seq_length = st.slider("序列长度", 5, 20, 8, 1)
            d_k = st.slider("维度 d_k", 16, 128, 64, 16)
            temperature = st.slider("温度参数", 0.1, 2.0, 1.0, 0.1)
            is_causal = st.checkbox("因果遮蔽", value=True)
        
        # 生成随机数据
        np.random.seed(42)
        Q = np.random.randn(seq_length, d_k)
        K = np.random.randn(seq_length, d_k)
        V = np.random.randn(seq_length, d_k)
        
        # 计算注意力分数
        scores = np.dot(Q, K.T) / np.sqrt(d_k * temperature)
        
        # 因果遮蔽
        if is_causal:
            mask = np.triu(np.ones((seq_length, seq_length)), k=1)
            scores = scores - mask * 1e9  # 用大负数遮蔽
        
        # Softmax
        attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # 输出
        output = np.dot(attention_weights, V)
        
        # 可视化注意力权重
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["注意力权重矩阵", "权重分布直方图"],
            specs=[[{"type": "heatmap"}, {"type": "histogram"}]]
        )
        
        # 热力图
        fig.add_trace(
            go.Heatmap(
                z=attention_weights,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="权重")
            ),
            row=1, col=1
        )
        
        # 直方图
        fig.add_trace(
            go.Histogram(
                x=attention_weights.flatten(),
                nbinsx=20,
                marker_color='lightblue',
                name='权重分布'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"注意力机制分析 (序列长度={seq_length}, d_k={d_k})",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 缩放因子的影响
        st.markdown("### ⚖️ 缩放因子的影响")
        
        fig_scaling = go.Figure()
        
        # 不同缩放因子的效果
        scales = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for scale in scales:
            scores_scaled = np.dot(Q, K.T) / np.sqrt(d_k * scale)
            if is_causal:
                scores_scaled = scores_scaled - mask * 1e9
            
            weights_scaled = np.exp(scores_scaled - np.max(scores_scaled, axis=1, keepdims=True))
            weights_scaled = weights_scaled / np.sum(weights_scaled, axis=1, keepdims=True)
            
            # 计算熵
            entropies = -np.sum(weights_scaled * np.log(weights_scaled + 1e-9), axis=1)
            mean_entropy = np.mean(entropies)
            
            fig_scaling.add_trace(go.Scatter(
                x=[scale], y=[mean_entropy],
                mode='markers+lines',
                marker=dict(size=8),
                name=f'缩放因子 {scale}'
            ))
        
        fig_scaling.update_layout(
            title="缩放因子对注意力熵的影响",
            xaxis_title="缩放因子",
            yaxis_title="平均熵",
            height=400
        )
        
        st.plotly_chart(fig_scaling, use_container_width=True)
        
        # 置换不变性演示
        st.markdown("### 🔄 置换不变性分析")
        
        # 打乱序列顺序
        indices = np.random.permutation(seq_length)
        Q_shuffled = Q[indices]
        K_shuffled = K[indices]
        V_shuffled = V[indices]
        
        # 计算打乱后的注意力
        scores_shuffled = np.dot(Q_shuffled, K_shuffled.T) / np.sqrt(d_k)
        if is_causal:
            mask_shuffled = np.triu(np.ones((seq_length, seq_length)), k=1)
            scores_shuffled = scores_shuffled - mask_shuffled * 1e9
        
        weights_shuffled = np.exp(scores_shuffled - np.max(scores_shuffled, axis=1, keepdims=True))
        weights_shuffled = weights_shuffled / np.sum(weights_shuffled, axis=1, keepdims=True)
        
        # 比较原始和打乱的权重
        fig_permutation = make_subplots(
            rows=1, cols=2,
            subplot_titles=["原始序列", "打乱序列"],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        fig_permutation.add_trace(
            go.Heatmap(z=attention_weights, colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        fig_permutation.add_trace(
            go.Heatmap(z=weights_shuffled, colorscale='Reds', showscale=False),
            row=1, col=2
        )
        
        fig_permutation.update_layout(
            title="置换不变性演示",
            height=400
        )
        
        st.plotly_chart(fig_permutation, use_container_width=True)
        
        st.warning("""
        **重要观察**：
        - 打乱序列后，注意力权重的数值完全相同，只是位置变了
        - 这证明了Attention的置换不变性
        - 因此需要位置编码来打破这种对称性
        """)
    
    @staticmethod
    def _render_multi_head_attention():
        """多头注意力演示"""
        st.markdown("### 🐠 多头注意力：突破秩亏瓶颈")
        
        st.latex(r"""
        MultiHead(Q,K,V) = Concat(head_1, \cdots, head_h) W^O
        """)
        
        with st.sidebar:
            num_heads = st.slider("头数量", 1, 8, 4, 1)
            seq_length = st.slider("序列长度", 8, 32, 16, 1)
            d_model = st.slider("模型维度", 64, 256, 128, 32)
            show_rank_analysis = st.checkbox("显示秩分析", value=True)
        
        d_k = d_model // num_heads
        
        # 生成随机数据
        np.random.seed(42)
        
        # 单头注意力
        Q_single = np.random.randn(seq_length, d_model)
        K_single = np.random.randn(seq_length, d_model)
        V_single = np.random.randn(seq_length, d_model)
        
        # 多头注意力
        Q_multi = np.random.randn(seq_length, num_heads, d_k)
        K_multi = np.random.randn(seq_length, num_heads, d_k)
        V_multi = np.random.randn(seq_length, num_heads, d_k)
        
        # 计算单头注意力
        scores_single = np.dot(Q_single, K_single.T) / np.sqrt(d_model)
        weights_single = np.exp(scores_single - np.max(scores_single, axis=1, keepdims=True))
        weights_single = weights_single / np.sum(weights_single, axis=1, keepdims=True)
        output_single = np.dot(weights_single, V_single)
        
        # 计算多头注意力
        outputs_multi = []
        for head in range(num_heads):
            scores = np.dot(Q_multi[:, head, :], K_multi[:, head, :].T) / np.sqrt(d_k)
            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            output = np.dot(weights, V_multi[:, head, :])
            outputs_multi.append(output)
        
        output_multi = np.concatenate(outputs_multi, axis=1)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "单头注意力权重", "多头注意力权重(头1)",
                "单头输出维度", "多头输出维度"
            ]
        )
        
        # 单头权重
        fig.add_trace(
            go.Heatmap(z=weights_single, colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # 多头权重(第一个头)
        weights_head1 = np.exp(np.dot(Q_multi[:, 0, :], K_multi[:, 0, :].T) / np.sqrt(d_k))
        weights_head1 = weights_head1 / np.sum(weights_head1, axis=1, keepdims=True)
        fig.add_trace(
            go.Heatmap(z=weights_head1, colorscale='Reds', showscale=False),
            row=1, col=2
        )
        
        # 输出维度对比
        fig.add_trace(
            go.Bar(
                x=['单头'], y=[output_single.shape[1]],
                marker_color='blue',
                name='维度'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=['多头'], y=[output_multi.shape[1]],
                marker_color='red',
                name='维度'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"单头 vs 多头注意力对比 (头数={num_heads})",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 秩分析
        if show_rank_analysis:
            st.markdown("### 📊 秩分析")
            
            # 计算矩阵的秩
            def matrix_rank(matrix, tol=1e-10):
                return np.linalg.matrix_rank(matrix, tol)
            
            # QK^T 的秩
            QK_single_rank = matrix_rank(np.dot(Q_single, K_single.T))
            QK_multi_ranks = []
            
            for head in range(num_heads):
                rank = matrix_rank(np.dot(Q_multi[:, head, :], K_multi[:, head, :].T))
                QK_multi_ranks.append(rank)
            
            # 可视化
            fig_rank = go.Figure()
            
            fig_rank.add_trace(go.Bar(
                x=['单头'] + [f'头{i+1}' for i in range(num_heads)],
                y=[QK_single_rank] + QK_multi_ranks,
                marker_color=['blue'] + ['red'] * num_heads
            ))
            
            # 添加理论最大秩线
            max_rank_single = min(d_model, seq_length)
            max_rank_multi = min(d_k, seq_length)
            
            fig_rank.add_hline(y=max_rank_single, line_dash="dash", line_color="blue", 
                             annotation_text=f"单头最大秩: {max_rank_single}")
            fig_rank.add_hline(y=max_rank_multi, line_dash="dash", line_color="red", 
                             annotation_text=f"多头最大秩: {max_rank_multi}")
            
            fig_rank.update_layout(
                title="Attention矩阵的秩分析",
                xaxis_title="头",
                yaxis_title="秩",
                height=400
            )
            
            st.plotly_chart(fig_rank, use_container_width=True)
            
            st.info(f"""
            **秩分析结果**：
            - 单头Attention秩: {QK_single_rank} (最大: {max_rank_single})
            - 多头平均秩: {np.mean(QK_multi_ranks):.1f} (最大: {max_rank_multi})
            - 多头通过拼接可以突破单头的秩限制
            - 总表达能力: {max_rank_multi * num_heads} > {max_rank_single}
            """)
        
        # 头间多样性分析
        st.markdown("### 🎭 头间多样性分析")
        
        # 计算不同头的注意力模式差异
        head_similarities = []
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                weights_i = np.exp(np.dot(Q_multi[:, i, :], K_multi[:, i, :].T) / np.sqrt(d_k))
                weights_i = weights_i / np.sum(weights_i, axis=1, keepdims=True)
                
                weights_j = np.exp(np.dot(Q_multi[:, j, :], K_multi[:, j, :].T) / np.sqrt(d_k))
                weights_j = weights_j / np.sum(weights_j, axis=1, keepdims=True)
                
                # 计算相似度
                similarity = np.corrcoef(weights_i.flatten(), weights_j.flatten())[0, 1]
                head_similarities.append((f"头{i+1}-头{j+1}", similarity))
        
        if head_similarities:
            fig_diversity = go.Figure()
            
            labels, similarities = zip(*head_similarities)
            fig_diversity.add_trace(go.Bar(
                x=list(labels),
                y=list(similarities),
                marker_color='lightgreen'
            ))
            
            fig_diversity.update_layout(
                title="头间注意力模式相似度",
                xaxis_title="头对",
                yaxis_title="相关系数",
                height=400
            )
            
            st.plotly_chart(fig_diversity, use_container_width=True)
            
            avg_similarity = np.mean(similarities)
            st.metric("平均相似度", f"{avg_similarity:.3f}")
        
        st.success("""
        **多头注意力的优势**：
        - 突破单头投影的秩亏瓶颈
        - 不同头学习不同的相似度度量
        - 通过拼接恢复高秩表达能力
        - 实现功能分工（语法、位置、长距离依赖等）
        """)
    
    @staticmethod
    def _render_linear_attention():
        """线性Attention演示"""
        st.markdown("### ⚡ 线性Attention：复杂度优化")
        
        st.latex(r"""
        \text{标准}: (QK^T)V \quad \text{vs} \quad \text{线性}: Q(K^TV)
        """)
        
        with st.sidebar:
            seq_length = st.slider("序列长度", 100, 2000, 500, 100)
            d_k = st.slider("特征维度", 32, 256, 128, 32)
            kernel_type = st.selectbox("线性核", ["ELU", "ReLU", "Softmax"])
        
        # 生成随机数据
        np.random.seed(42)
        Q = np.random.randn(seq_length, d_k)
        K = np.random.randn(seq_length, d_k)
        V = np.random.randn(seq_length, d_k)
        
        # 标准Attention
        def standard_attention(Q, K, V):
            scores = np.dot(Q, K.T) / np.sqrt(d_k)
            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            return np.dot(weights, V), weights
        
        # 线性Attention
        def linear_attention(Q, K, V, kernel_type="ELU"):
            # K^TV: 先计算 K^TV
            KTV = np.dot(K.T, V)
            
            # Q(K^TV): 然后与Q相乘
            if kernel_type == "ELU":
                features = np.dot(Q, KTV)
                output = features / (np.dot(Q, K.sum(axis=0, keepdims=True)) + 1e-6)
            elif kernel_type == "ReLU":
                features = np.dot(Q, KTV)
                output = np.maximum(0, features)
            else:  # Softmax approximation
                features = np.dot(Q, KTV)
                output = features / (np.sum(features, axis=1, keepdims=True) + 1e-6)
            
            return output, None
        
        # 计算两种方法
        import time
        
        # 标准Attention (可能很慢)
        start_time = time.time()
        try:
            output_std, weights_std = standard_attention(Q, K, V)
            std_time = time.time() - start_time
            std_success = True
        except MemoryError:
            std_time = float('inf')
            std_success = False
            output_std = None
        
        # 线性Attention
        start_time = time.time()
        output_linear, _ = linear_attention(Q, K, V, kernel_type)
        linear_time = time.time() - start_time
        
        # 复杂度分析
        fig_complexity = go.Figure()
        
        methods = ['标准Attention', '线性Attention']
        times = [std_time if std_success else None, linear_time]
        complexities = [f"O(N²d)", f"O(Nd²)"]
        
        for i, (method, time_val, complexity) in enumerate(zip(methods, times, complexities)):
            if time_val is not None:
                fig_complexity.add_trace(go.Bar(
                    x=[method],
                    y=[time_val],
                    name=f"{method} ({complexity})",
                    text=f"{time_val:.4f}s"
                ))
        
        fig_complexity.update_layout(
            title=f"计算时间对比 (N={seq_length}, d={d_k})",
            xaxis_title="方法",
            yaxis_title="时间 (秒)",
            height=400
        )
        
        st.plotly_chart(fig_complexity, use_container_width=True)
        
        # 复杂度理论分析
        st.markdown("### 📊 复杂度理论分析")
        
        N_values = np.logspace(1, 3.5, 20)  # 10 to ~3000
        d_fixed = 128
        
        # 理论复杂度
        standard_complexity = N_values**2 * d_fixed
        linear_complexity = N_values * d_fixed**2
        
        fig_theory = go.Figure()
        
        fig_theory.add_trace(go.Scatter(
            x=N_values, y=standard_complexity,
            mode='lines',
            name='标准Attention O(N²d)',
            line=dict(color='red', width=3)
        ))
        
        fig_theory.add_trace(go.Scatter(
            x=N_values, y=linear_complexity,
            mode='lines',
            name='线性Attention O(Nd²)',
            line=dict(color='blue', width=3)
        ))
        
        fig_theory.update_layout(
            title="理论复杂度对比 (对数尺度)",
            xaxis_title="序列长度 N",
            yaxis_title="操作数",
            xaxis_type="log",
            yaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig_theory, use_container_width=True)
        
        # 交叉点分析
        crossover_point = d_fixed
        st.info(f"""
        **复杂度分析**：
        - 交叉点: N = {d_fixed}
        - N < {d_fixed}: 标准Attention更快
        - N > {d_fixed}: 线性Attention更快
        - 当前N={seq_length}: {'线性Attention' if seq_length > d_fixed else '标准Attention'}更优
        """)
        
        # 精度对比
        if std_success and output_std is not None:
            st.markdown("### 🎯 精度对比")
            
            # 计算输出差异
            diff = np.mean(np.abs(output_std - output_linear))
            relative_diff = diff / (np.mean(np.abs(output_std)) + 1e-8)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("绝对差异", f"{diff:.6f}")
            with col2:
                st.metric("相对差异", f"{relative_diff:.2%}")
            
            # 可视化输出对比
            fig_comparison = go.Figure()
            
            # 选择前10个输出维度进行可视化
            dims_to_show = min(10, d_k)
            
            fig_comparison.add_trace(go.Scatter(
                x=np.arange(dims_to_show),
                y=output_std[0, :dims_to_show],
                mode='lines+markers',
                name='标准Attention',
                line=dict(color='red')
            ))
            
            fig_comparison.add_trace(go.Scatter(
                x=np.arange(dims_to_show),
                y=output_linear[0, :dims_to_show],
                mode='lines+markers',
                name='线性Attention',
                line=dict(color='blue')
            ))
            
            fig_comparison.update_layout(
                title="输出对比 (前10个维度)",
                xaxis_title="维度",
                yaxis_title="输出值",
                height=400
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # 线性核函数对比
        st.markdown("### 🔧 线性核函数对比")
        
        # 测试不同核函数
        kernels = ["ELU", "ReLU", "Softmax"]
        kernel_results = {}
        
        for kernel in kernels:
            _, output = linear_attention(Q, K, V, kernel)
            kernel_results[kernel] = np.mean(output)
        
        fig_kernels = go.Figure()
        
        fig_kernels.add_trace(go.Bar(
            x=list(kernel_results.keys()),
            y=list(kernel_results.values()),
            marker_color=['lightblue', 'lightgreen', 'lightcoral']
        ))
        
        fig_kernels.update_layout(
            title="不同线性核的输出均值",
            xaxis_title="核函数",
            yaxis_title="输出均值",
            height=400
        )
        
        st.plotly_chart(fig_kernels, use_container_width=True)
        
        st.warning("""
        **线性Attention的权衡**：
        - 优势：复杂度从O(N²)降到O(N)，适合长序列
        - 劣势：失去Softmax的"聚焦"能力，可能影响精度
        - 应用：长文本处理、高分辨率图像等场景
        - 发展：FlashAttention等工程优化进一步提升了实用性
        """)


# 为了兼容性，添加缺少的导入
try:
    from scipy.stats import norm
    from scipy.spatial.distance import cdist
except ImportError:
    # 如果scipy不可用，使用numpy实现
    def norm(*args, **kwargs):
        pass
    
    def cdist(XA, XB):
        from numpy.linalg import norm
        n = XA.shape[0]
        m = XB.shape[0]
        dm = np.empty((n, m))
        for i in range(n):
            for j in range(m):
                dm[i, j] = norm(XA[i] - XB[j])
        return dm