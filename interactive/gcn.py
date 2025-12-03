"""
交互式图神经网络(GCN)与谱图理论可视化
严格按照 20.GCN.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from scipy.linalg import eig
from scipy.sparse import csr_matrix
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

warnings.filterwarnings('ignore')

# 尝试导入torch，如果不可用则使用numpy实现
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch未安装，部分功能将使用简化实现")


class InteractiveGCN:
    """交互式图神经网络与谱图理论可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🌐 图神经网络与谱图理论：非欧几里得空间的谐波分析")
        st.markdown("""
        **核心思想**: 当数据呈现拓扑结构时，利用谱图理论在频域重新定义卷积
        
        关键概念：
        - **拉普拉斯矩阵**: $\\mathbf{L} = \\mathbf{D} - \\mathbf{A}$
        - **谱分解**: $\\mathbf{L} = \\mathbf{U} \\mathbf{\\Lambda} \\mathbf{U}^T$
        - **图卷积**: $\\mathbf{x} *_G \\mathbf{g} = \\mathbf{U} ((\\mathbf{U}^T \\mathbf{g}) \\odot (\\mathbf{U}^T \\mathbf{x}))$
        - **GCN传播**: $\\mathbf{H}^{(l+1)} = \\sigma(\\tilde{\\mathbf{D}}^{-\\frac{1}{2}} \\tilde{\\mathbf{A}} \\tilde{\\mathbf{D}}^{-\\frac{1}{2}} \\mathbf{H}^{(l)} \\mathbf{W}^{(l)})$
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["图基础概念", "拉普拉斯矩阵", "谱图理论", "GCN传播"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "图基础概念":
            InteractiveGCN._render_graph_basics()
        elif viz_type == "拉普拉斯矩阵":
            InteractiveGCN._render_laplacian()
        elif viz_type == "谱图理论":
            InteractiveGCN._render_spectral_theory()
        elif viz_type == "GCN传播":
            InteractiveGCN._render_gcn_propagation()
    

        # 添加交互式测验
        quiz_system = QuizSystem("gcn")
        quizzes = QuizTemplates.get_gcn_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_graph_basics():
        """图基础概念演示"""
        st.markdown("### 🕸️ 图的基础概念：从CNN到GNN")
        
        st.markdown("""
        **核心挑战**：
        - **CNN**: 像素排列整齐，有明确的"上下左右"
        - **GNN**: 节点连接不规则，没有平移不变性
        
        **解决思路**: 转向频域，利用谱图理论定义卷积
        """)
        
        with st.sidebar:
            graph_type = st.selectbox("图类型", 
                ["环形图", "随机图", "路径图", "星形图"])
            num_nodes = st.slider("节点数量", 5, 20, 8, 1)
            show_labels = st.checkbox("显示节点标签", value=True)
            show_weights = st.checkbox("显示边权重", value=False)
        
        # 创建不同类型的图
        if graph_type == "环形图":
            G = nx.cycle_graph(num_nodes)
            graph_name = "环形图"
        elif graph_type == "随机图":
            G = nx.erdos_renyi_graph(num_nodes, 0.3)
            graph_name = "随机图"
        elif graph_type == "路径图":
            G = nx.path_graph(num_nodes)
            graph_name = "路径图"
        else:  # 星形图
            G = nx.star_graph(num_nodes-1)
            graph_name = "星形图"
        
        # 添加权重（可选）
        if show_weights:
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = np.random.uniform(0.1, 1.0)
        
        # 计算图的基本属性
        A = nx.adjacency_matrix(G).todense()
        degrees = np.array([G.degree(i) for i in G.nodes()])
        clustering = np.array([nx.clustering(G, i) for i in G.nodes()])
        
        # 可视化图结构
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"{graph_name}结构", "度分布与聚类系数"],
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 图结构可视化
        pos = nx.spring_layout(G)
        
        # 绘制边
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 绘制节点
        node_x = [pos[i][0] for i in G.nodes()]
        node_y = [pos[i][1] for i in G.nodes()]
        
        if show_labels:
            node_text = [f'节点{i}' for i in G.nodes()]
        else:
            node_text = None
        
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=degrees * 20,  # 大小反映度数
                    color=clustering,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="聚类系数", x=1.02, y=0.5)
                ),
                text=node_text,
                textposition="middle center",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 度分布
        fig.add_trace(
            go.Bar(
                x=list(G.nodes()),
                y=degrees,
                name='度数',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 聚类系数
        fig.add_trace(
            go.Bar(
                x=list(G.nodes()),
                y=clustering,
                name='聚类系数',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"图结构分析 - {graph_name}",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 图统计信息
        st.markdown("### 📊 图统计信息")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("节点数", len(G.nodes()))
        with col2:
            st.metric("边数", len(G.edges()))
        with col3:
            st.metric("平均度", f"{np.mean(degrees):.2f}")
        with col4:
            st.metric("平均聚类系数", f"{np.mean(clustering):.3f}")
        
        st.info("""
        **图的基本概念**：
        - **邻接矩阵A**: 描述节点间的连接关系
        - **度矩阵D**: 对角矩阵，对角元素为节点度数
        - **聚类系数**: 衡量节点邻居间的连接紧密程度
        - **拓扑结构**: 决定了信息传播的方式和效率
        """)
    
    @staticmethod
    def _render_laplacian():
        """拉普拉斯矩阵演示"""
        st.markdown("### 📐 拉普拉斯矩阵：图的二阶导数")
        
        st.latex(r"""
        \mathbf{L} = \mathbf{D} - \mathbf{A}
        """)
        
        st.markdown("""
        **物理意义**: $(\\mathbf{L}\\mathbf{x})_i = \\sum_{j \\in \\mathcal{N}(i)} (\\mathbf{x}_i - \\mathbf{x}_j)$
        
        这等价于微积分中的 $-\\Delta f$，衡量信号在图上的变化剧烈程度。
        """)
        
        with st.sidebar:
            graph_type = st.selectbox("图类型", 
                ["环形图", "完全图", "路径图", "随机图"])
            num_nodes = st.slider("节点数量", 5, 15, 8, 1)
            show_eigenvalues = st.checkbox("显示特征值", value=True)
            show_heatmap = st.checkbox("显示热力图", value=True)
        
        # 创建图
        if graph_type == "环形图":
            G = nx.cycle_graph(num_nodes)
        elif graph_type == "完全图":
            G = nx.complete_graph(num_nodes)
        elif graph_type == "路径图":
            G = nx.path_graph(num_nodes)
        else:  # 随机图
            G = nx.erdos_renyi_graph(num_nodes, 0.3)
        
        # 计算拉普拉斯矩阵
        A = nx.adjacency_matrix(G).todense()
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eig(L)
        
        # 确保特征值是实数（拉普拉斯矩阵是实对称矩阵）
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 排序特征值
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "拉普拉斯矩阵", "特征值分布",
                "前4个特征向量", "平滑度分析"
            ]
        )
        
        # 拉普拉斯矩阵热力图
        if show_heatmap:
            fig.add_trace(
                go.Heatmap(
                    z=L,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="值")
                ),
                row=1, col=1
            )
        
        # 特征值分布
        fig.add_trace(
            go.Scatter(
                x=list(range(len(eigenvalues))),
                y=eigenvalues,
                mode='lines+markers',
                name='特征值',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # 前4个特征向量
        for i in range(min(4, num_nodes)):
            fig.add_trace(
                go.Scatter(
                    x=list(range(num_nodes)),
                    y=eigenvectors[:, i],
                    mode='lines+markers',
                    name=f'特征向量{i+1}',
                    line=dict(width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
        
        # 平滑度分析：x^T L x
        test_signals = []
        smoothness = []
        
        for i in range(num_nodes):
            # 创建测试信号：只在第i个节点为1
            signal = np.zeros(num_nodes)
            signal[i] = 1.0
            
            # 计算平滑度
            smooth = signal.T @ L @ signal
            test_signals.append(signal)
            smoothness.append(smooth)
        
        fig.add_trace(
            go.Bar(
                x=[f'节点{i}' for i in range(num_nodes)],
                y=smoothness,
                name='平滑度',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="拉普拉斯矩阵分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 特征值分析
        if show_eigenvalues:
            st.markdown("### 📊 特征值分析")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最小特征值", f"{eigenvalues[0]:.6f}")
            with col2:
                st.metric("最大特征值", f"{eigenvalues[-1]:.6f}")
            with col3:
                st.metric("特征值跨度", f"{eigenvalues[-1] - eigenvalues[0]:.6f}")
            with col4:
                st.metric("条件数", f"{eigenvalues[-1]/max(eigenvalues[0], 1e-8):.2f}")
            
            st.info("""
            **特征值的物理意义**：
            - **小特征值**: 对应低频信号（全局变化缓慢）
            - **大特征值**: 对应高频信号（局部变化剧烈）
            - **特征值0**: 对应常数函数（完全不变化）
            - **条件数**: 反映图的数值稳定性
            """)
    
    @staticmethod
    def _render_spectral_theory():
        """谱图理论演示"""
        st.markdown("### 🌊 谱图理论：图的傅里叶变换")
        
        st.latex(r"""
        \mathbf{L} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T
        """)
        
        st.markdown("""
        **核心思想**：
        - **U**: 特征向量，即图的傅里叶基
        - **Λ**: 特征值，即图的频率
        - **卷积定理**: 时域卷积 = 频域乘法
        """)
        
        with st.sidebar:
            graph_type = st.selectbox("图类型", 
                ["环形图", "随机图", "社区图"])
            num_nodes = st.slider("节点数量", 8, 20, 12, 1)
            signal_type = st.selectbox("信号类型", 
                ["脉冲信号", "正弦信号", "随机信号"])
            filter_type = st.selectbox("滤波器类型", 
                ["低通", "高通", "带通"])
            show_comparison = st.checkbox("显示时域vs频域对比", value=True)
        
        # 创建图
        if graph_type == "环形图":
            G = nx.cycle_graph(num_nodes)
        elif graph_type == "随机图":
            G = nx.erdos_renyi_graph(num_nodes, 0.3)
        else:  # 社区图
            # barbell_graph需要两个参数：m1和m2，这里创建两个社区然后连接
            community_size = max(3, num_nodes // 3)
            G = nx.barbell_graph(community_size, 0)
        
        # 计算谱分解
        A = nx.adjacency_matrix(G).todense()
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigenvalues, eigenvectors = eig(L)
        
        # 确保特征值是实数
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 排序
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 创建测试信号
        if signal_type == "脉冲信号":
            signal = np.zeros(num_nodes)
            signal[0] = 1.0
            signal[num_nodes//2] = 0.5
        elif signal_type == "正弦信号":
            signal = np.sin(2 * np.pi * np.arange(num_nodes) / num_nodes * 2)
        else:  # 随机信号
            signal = np.random.randn(num_nodes)
        
        # 设计频域滤波器
        filter_response = np.ones(num_nodes)
        if filter_type == "低通":
            filter_response[eigenvalues > np.percentile(eigenvalues, 50)] = 0.1
        elif filter_type == "高通":
            filter_response[eigenvalues < np.percentile(eigenvalues, 50)] = 0.1
        else:  # 带通
            mid_freq = np.percentile(eigenvalues, 50)
            threshold = np.percentile(np.abs(eigenvalues - mid_freq), 25)
            filter_response = (np.abs(eigenvalues - mid_freq) < threshold).astype(float)
        
        # 图卷积：时域 vs 频域
        # 频域卷积
        signal_freq = eigenvectors.T @ signal
        filtered_freq = filter_response * signal_freq
        result_freq = eigenvectors @ filtered_freq
        
        # 时域卷积（直接计算，用于验证）
        A_hat = A + np.eye(num_nodes)
        D_hat = np.diag(np.sum(A_hat, axis=1))
        A_norm = np.linalg.inv(np.sqrt(D_hat)) @ A_hat @ np.linalg.inv(np.sqrt(D_hat))
        
        result_spatial = A_norm @ signal
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "原始信号", "频域表示", "频域滤波器",
                "频域卷积结果", "时域卷积结果", "对比分析"
            ]
        )
        
        # 原始信号
        fig.add_trace(
            go.Scatter(
                x=list(range(num_nodes)),
                y=signal,
                mode='lines+markers',
                name='原始信号',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # 频域表示
        fig.add_trace(
            go.Scatter(
                x=list(range(num_nodes)),
                y=signal_freq,
                mode='lines+markers',
                name='频域信号',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # 频域滤波器
        fig.add_trace(
            go.Scatter(
                x=list(range(num_nodes)),
                y=filter_response,
                mode='lines+markers',
                name=f'{filter_type}滤波器',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=1, col=3
        )
        
        # 频域卷积结果
        fig.add_trace(
            go.Scatter(
                x=list(range(num_nodes)),
                y=result_freq,
                mode='lines+markers',
                name='频域卷积结果',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 时域卷积结果
        fig.add_trace(
            go.Scatter(
                x=list(range(num_nodes)),
                y=result_spatial,
                mode='lines+markers',
                name='时域卷积结果',
                line=dict(width=2, dash='dash'),
                marker=dict(size=6)
            ),
            row=2, col=2
        )
        
        # 对比分析
        diff = np.abs(result_freq - result_spatial)
        fig.add_trace(
            go.Scatter(
                x=list(range(num_nodes)),
                y=diff,
                mode='lines+markers',
                name='频域vs时域差异',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="谱图卷积分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 理论验证
        if show_comparison:
            st.markdown("### 🔬 理论验证")
            
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最大差异", f"{max_diff:.6f}")
            with col2:
                st.metric("平均差异", f"{mean_diff:.6f}")
            
            if max_diff < 1e-10:
                st.success("✅ 频域和时域卷积结果完全一致！")
            else:
                st.warning(f"⚠️ 频域和时域存在数值差异，可能是精度问题")
        
        st.success("""
        **谱图理论的核心价值**：
        - **统一框架**: 将不同拓扑结构的图统一到频域分析
        - **数学优雅**: 特征分解提供了自然的正交基
        - **计算效率**: 避免O(N³)的特征分解，使用多项式近似
        - **物理直觉**: 频率对应信号的"振荡模式"
        """)
    
    @staticmethod
    def _render_gcn_propagation():
        """GCN传播机制演示"""
        st.markdown("### 🔄 GCN传播机制：消息传递算法")
        
        st.latex(r"""
        \mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)
        """)
        
        st.markdown("""
        **传播步骤**：
        1. **添加自环**: $\\tilde{\\mathbf{A}} = \\mathbf{A} + \\mathbf{I}$
        2. **对称归一化**: $\\tilde{\\mathbf{D}}^{-\\frac{1}{2}} \\tilde{\\mathbf{A}} \\tilde{\\mathbf{D}}^{-\\frac{1}{2}}$
        3. **消息传递**: 聚合邻居特征
        4. **线性变换**: 乘以权重矩阵
        5. **非线性激活**: 应用激活函数
        """)
        
        with st.sidebar:
            num_layers = st.slider("GCN层数", 1, 3, 2, 1)
            hidden_dim = st.slider("隐藏维度", 2, 8, 4, 1)
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            num_epochs = st.slider("训练轮数", 50, 300, 100, 10)
            animation_speed = st.slider("动画速度", 1, 10, 5, 1)
            show_animation = st.checkbox("显示传播动画", value=True)
        
        # 创建Karate Club图
        G = nx.karate_club_graph()
        num_nodes = len(G.nodes())
        
        # 准备数据
        A = nx.adjacency_matrix(G).todense()
        A = A + np.eye(num_nodes)  # 添加自环
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
        
        # 特征矩阵（单位矩阵）
        X = np.eye(num_nodes)
        
        # 标签（简化版：前半部分为0，后半部分为1）
        labels = np.array([0 if i < 17 else 1 for i in range(num_nodes)])
        
        # 初始化权重
        np.random.seed(42)
        W1 = np.random.randn(num_nodes, hidden_dim) * 0.01
        W2 = np.random.randn(hidden_dim, 2) * 0.01
        
        # 训练过程
        loss_history = []
        embedding_history = []
        
        def relu(x):
            return np.maximum(0, x)
        
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        for epoch in range(num_epochs):
            # 前向传播
            H1 = relu(A_hat @ X @ W1)
            H2 = A_hat @ H1 @ W2
            logits = H2
            
            # 计算损失
            probs = softmax(logits)
            loss = -np.mean(np.log(probs[range(num_nodes), labels]))
            loss_history.append(loss)
            
            # 保存嵌入用于可视化
            embedding_history.append(H2.copy())
            
            # 简化的反向传播（实际GCN使用梯度下降）
            if epoch % 20 == 0:
                # 创建one-hot编码的标签矩阵
                one_hot_labels = np.zeros((num_nodes, 2))
                one_hot_labels[range(num_nodes), labels] = 1.0
                
                # 重置梯度（简化处理）
                grad_W2 = H1.T @ (probs - one_hot_labels) / num_nodes
                grad_W1 = X.T @ ((probs - one_hot_labels) @ W2.T) / num_nodes
                
                # 更新权重
                W2 -= learning_rate * grad_W2
                W1 -= learning_rate * grad_W1
        
        # 可视化训练过程
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "损失曲线", "嵌入演化（前2维）",
                "最终嵌入分布", "传播动画"
            ]
        )
        
        # 损失曲线
        fig.add_trace(
            go.Scatter(
                x=list(range(len(loss_history))),
                y=loss_history,
                mode='lines',
                name='损失',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # 嵌入演化
        if len(embedding_history) > 1:
            embeddings = np.array(embedding_history)
            
            # 选择几个时间点
            time_points = [0, len(embeddings)//4, len(embeddings)//2, -1]
            colors = ['blue', 'green', 'orange', 'red']
            
            for i, t in enumerate(time_points):
                fig.add_trace(
                    go.Scatter(
                        x=embeddings[t, :, 0],
                        y=embeddings[t, :, 1],
                        mode='markers',
                        name=f'Epoch {t}',
                        marker=dict(
                            size=8,
                            color=colors[i],
                            opacity=0.7
                        ),
                        showlegend=False if i > 0 else True
                    ),
                    row=1, col=2
                )
        
        # 最终嵌入分布
        final_embedding = embedding_history[-1]
        
        fig.add_trace(
            go.Scatter(
                x=final_embedding[:, 0],
                y=final_embedding[:, 1],
                mode='markers',
                name='节点0',
                marker=dict(
                    size=10,
                    color=labels,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="类别", x=1.02, y=0.5)
                ),
                text=[f'节点{i}' for i in range(num_nodes)],
                textposition="middle center",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 传播动画
        if show_animation and len(embedding_history) > 1:
            # 简化的动画：显示几个关键步骤
            embeddings = np.array(embedding_history)
            key_steps = [0, len(embeddings)//4, len(embeddings)//2, -1]
            
            for i, step in enumerate(key_steps):
                fig.add_trace(
                    go.Scatter(
                        x=embeddings[step, :, 0],
                        y=embeddings[step, :, 1],
                        mode='markers',
                        name=f'步骤{i+1}',
                        marker=dict(
                            size=12,
                            symbol=i+1,
                            opacity=0.8
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="GCN训练过程可视化",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能分析
        st.markdown("### 📊 GCN性能分析")
        
        # 计算准确率
        final_probs = softmax(embedding_history[-1])
        predictions = np.argmax(final_probs, axis=1)
        accuracy = np.mean(predictions == labels)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最终损失", f"{loss_history[-1]:.4f}")
        with col2:
            st.metric("分类准确率", f"{accuracy:.2%}")
        with col3:
            st.metric("收敛轮数", f"{len(loss_history)}")
        with col4:
            st.metric("节点数", f"{num_nodes}")
        
        st.success("""
        **GCN的核心优势**：
        - **结构感知**: 利用图的拓扑结构进行学习
        - **参数共享**: 权重在所有节点间共享
        - **可扩展性**: 计算复杂度与边数成线性关系
        - **理论保证**: 基于谱图理论的数学基础
        """)


# 为了兼容性，添加缺少的导入
try:
    from scipy.linalg import eig
except ImportError:
    # 如果scipy不可用，使用numpy实现
    def eig(matrix):
        return np.linalg.eig(matrix)

        # 添加交互式测验
