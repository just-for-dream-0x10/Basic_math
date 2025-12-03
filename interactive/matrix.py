"""
交互式矩阵变换可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveMatrix:
    """交互式矩阵变换可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.title("🔢 矩阵论：数据的几何与变换")
        
        # 添加标签页
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📐 线性变换", 
            "🎯 SVD与LoRA", 
            "🔺 XOR升维",
            "📊 PCA降维",
            "🌋 特征值谱",
            "⛰️ 海森与鞍点"
        ])
        
        with tab1:
            InteractiveMatrix._render_linear_transform()
        
        with tab2:
            InteractiveMatrix._render_svd_lora()
        
        with tab3:
            InteractiveMatrix._render_xor_lifting()
        
        with tab4:
            InteractiveMatrix._render_pca()
        
        with tab5:
            InteractiveMatrix._render_eigenspectrum()
        
        with tab6:
            InteractiveMatrix._render_hessian()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("matrix")
        quizzes = QuizTemplates.get_matrix_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_linear_transform():
        """线性变换可视化"""
        st.subheader("📐 线性变换的几何意义")
        st.markdown("""
        **线性变换的本质**: 矩阵是线性变换的表示
        
        对于变换 $T: \\mathbb{R}^n \\to \\mathbb{R}^m$，其矩阵表示为：
        
        $$T(\\mathbf{x}) = A\\mathbf{x}$$
        
        其中 $A \\in \\mathbb{R}^{m \\times n}$ 的第 $j$ 列是 $T(\\mathbf{e}_j)$ (第 $j$ 个基向量的像)
        
        **关键性质**:
        - 行列式 $\\det(A)$: 面积/体积的缩放因子
        - 特征值/特征向量: $A\\mathbf{v} = \\lambda\\mathbf{v}$ (不变方向)
        - 迹 $\\text{tr}(A) = \\sum \\lambda_i$: 特征值之和
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 变换类型")
            transform_type = st.selectbox("选择变换", 
                ["自定义矩阵", "旋转", "缩放", "剪切", "反射", "投影"])
            
            st.markdown("### 📐 可视化设置")
            show_grid = st.checkbox("显示网格", value=True)
            show_eigen = st.checkbox("显示特征向量", value=True)
        
        # 获取变换矩阵
        if transform_type == "自定义矩阵":
            st.markdown("#### 编辑矩阵 (2×2)")
            col1, col2 = st.columns(2)
            with col1:
                a11 = st.number_input("a₁₁", -5.0, 5.0, 1.0, 0.1, key="m11")
                a21 = st.number_input("a₂₁", -5.0, 5.0, 0.0, 0.1, key="m21")
            with col2:
                a12 = st.number_input("a₁₂", -5.0, 5.0, 0.0, 0.1, key="m12")
                a22 = st.number_input("a₂₂", -5.0, 5.0, 1.0, 0.1, key="m22")
            matrix = np.array([[a11, a12], [a21, a22]])
        
        elif transform_type == "旋转":
            st.markdown(r"""
            **旋转矩阵**: 
            $$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
            
            性质: $R^T R = I$ (正交矩阵), $\det(R) = 1$, 保持距离和角度
            """)
            angle = st.slider("旋转角度 (度)", 0, 360, 45, 5)
            theta = np.radians(angle)
            matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
        
        elif transform_type == "缩放":
            st.markdown(r"""
            **缩放矩阵**: 
            $$S(s_x, s_y) = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$
            
            - $\det(S) = s_x \cdot s_y$ (面积缩放因子)
            - 特征值: $\lambda_1 = s_x, \lambda_2 = s_y$
            - 沿坐标轴方向拉伸/压缩
            """)
            scale_x = st.slider("X轴缩放", 0.1, 3.0, 1.0, 0.1)
            scale_y = st.slider("Y轴缩放", 0.1, 3.0, 1.0, 0.1)
            matrix = np.array([[scale_x, 0], [0, scale_y]])
        
        elif transform_type == "剪切":
            st.markdown(r"""
            **剪切矩阵**: 
            $$\text{Shear} = \begin{bmatrix} 1 & k_x \\ k_y & 1 \end{bmatrix}$$
            
            - $\det = 1$ (保持面积)
            - 使正方形变成平行四边形
            - 应用: 斜体字、透视变换
            """)
            shear_x = st.slider("X方向剪切", -2.0, 2.0, 0.5, 0.1)
            shear_y = st.slider("Y方向剪切", -2.0, 2.0, 0.0, 0.1)
            matrix = np.array([[1, shear_x], [shear_y, 1]])
        
        elif transform_type == "反射":
            axis = st.radio("反射轴", ["X轴", "Y轴", "y=x", "y=-x"])
            if axis == "X轴":
                matrix = np.array([[1, 0], [0, -1]])
            elif axis == "Y轴":
                matrix = np.array([[-1, 0], [0, 1]])
            elif axis == "y=x":
                matrix = np.array([[0, 1], [1, 0]])
            else:  # y=-x
                matrix = np.array([[0, -1], [-1, 0]])
        
        elif transform_type == "投影":
            axis = st.radio("投影到", ["X轴", "Y轴", "y=x"])
            if axis == "X轴":
                matrix = np.array([[1, 0], [0, 0]])
            elif axis == "Y轴":
                matrix = np.array([[0, 0], [0, 1]])
            else:  # y=x
                matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        # 显示矩阵
        st.markdown("### 🔢 变换矩阵")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.latex(r"\begin{bmatrix} " + 
                    f"{matrix[0,0]:.2f} & {matrix[0,1]:.2f} \\\\ " +
                    f"{matrix[1,0]:.2f} & {matrix[1,1]:.2f}" + 
                    r" \end{bmatrix}")
            
            # 矩阵属性
            det = np.linalg.det(matrix)
            st.metric("行列式", f"{det:.3f}")
            
            try:
                eigenvalues = np.linalg.eigvals(matrix)
                st.write("**特征值:**")
                for i, ev in enumerate(eigenvalues):
                    if np.isreal(ev):
                        st.write(f"λ{i+1} = {ev.real:.3f}")
                    else:
                        st.write(f"λ{i+1} = {ev.real:.3f} + {ev.imag:.3f}i")
            except:
                st.write("**特征值:** 无法计算")
        
        with col2:
            # 可视化变换
            fig = InteractiveMatrix._visualize_transformation(
                matrix, show_grid, show_eigen
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 向量变换演示
        st.markdown("### 📍 向量变换")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 选择向量")
            v_x = st.slider("向量 x", -3.0, 3.0, 1.0, 0.1, key="vx")
            v_y = st.slider("向量 y", -3.0, 3.0, 1.0, 0.1, key="vy")
        
        v = np.array([v_x, v_y])
        v_transformed = matrix @ v
        
        with col2:
            st.markdown("#### 原始向量")
            st.latex(r"\mathbf{v} = \begin{bmatrix} " + 
                    f"{v[0]:.2f} \\\\ {v[1]:.2f}" + 
                    r" \end{bmatrix}")
            st.write(f"长度: {np.linalg.norm(v):.3f}")
        
        with col3:
            st.markdown("#### 变换后")
            st.latex(r"A\mathbf{v} = \begin{bmatrix} " + 
                    f"{v_transformed[0]:.2f} \\\\ {v_transformed[1]:.2f}" + 
                    r" \end{bmatrix}")
            st.write(f"长度: {np.linalg.norm(v_transformed):.3f}")
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _visualize_transformation(matrix, show_grid, show_eigen):
        """可视化矩阵变换"""
        fig = go.Figure()
        
        # 原始单位正方形
        square = np.array([[0, 1, 1, 0, 0],
                          [0, 0, 1, 1, 0]])
        
        # 变换后的正方形
        transformed = matrix @ square
        
        # 绘制原始形状
        fig.add_trace(go.Scatter(
            x=square[0], y=square[1],
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name='原始',
            showlegend=True
        ))
        
        # 绘制变换后的形状
        fig.add_trace(go.Scatter(
            x=transformed[0], y=transformed[1],
            mode='lines',
            line=dict(color='red', width=3),
            name='变换后',
            showlegend=True
        ))
        
        # 绘制网格
        if show_grid:
            grid_range = 3
            for i in range(-grid_range, grid_range + 1):
                # 垂直线
                line = np.array([[i, i], [-grid_range, grid_range]])
                trans_line = matrix @ line
                fig.add_trace(go.Scatter(
                    x=trans_line[0], y=trans_line[1],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # 水平线
                line = np.array([[-grid_range, grid_range], [i, i]])
                trans_line = matrix @ line
                fig.add_trace(go.Scatter(
                    x=trans_line[0], y=trans_line[1],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # 绘制坐标轴
        fig.add_trace(go.Scatter(
            x=[-5, 5], y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[-5, 5],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 绘制特征向量
        if show_eigen:
            try:
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                    if np.isreal(val):
                        vec_real = np.real(vec)
                        vec_real = vec_real / np.linalg.norm(vec_real) * 2
                        fig.add_trace(go.Scatter(
                            x=[0, vec_real[0]], y=[0, vec_real[1]],
                            mode='lines+markers',
                            line=dict(color='green', width=2),
                            marker=dict(size=8),
                            name=f'特征向量 {i+1}',
                            showlegend=True
                        ))
            except:
                pass
        
        fig.update_layout(
            title="矩阵变换可视化",
            xaxis=dict(range=[-5, 5], constrain='domain', scaleanchor='y'),
            yaxis=dict(range=[-5, 5], constrain='domain'),
            height=500,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def _render_svd_lora():
        """SVD分解与LoRA原理"""
        st.subheader("🎯 SVD分解与LoRA (Low-Rank Adaptation)")
        
        st.markdown("""
        **奇异值分解 (SVD)**: 任意矩阵 $W \\in \\mathbb{R}^{m \\times n}$ 可分解为：
        
        $$W = U \\Sigma V^T$$
        
        其中：
        - $U \\in \\mathbb{R}^{m \\times m}$: 左奇异向量（输出空间的旋转）
        - $\\Sigma \\in \\mathbb{R}^{m \\times n}$: 奇异值对角矩阵（缩放）
        - $V \\in \\mathbb{R}^{n \\times n}$: 右奇异向量（输入空间的旋转）
        
        **LoRA原理**: 在大模型微调中，权重更新 $\\Delta W$ 具有低秩性质：
        
        $$\\Delta W \\approx B \\cdot A$$
        
        其中 $B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times d}$，且 $r \\ll d$
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 参数设置")
            matrix_size = st.slider("矩阵大小", 10, 100, 50, 10)
            rank = st.slider("保留秩 (Rank)", 1, min(20, matrix_size), 5, 1)
            
        # 生成随机矩阵
        np.random.seed(42)
        W = np.random.randn(matrix_size, matrix_size) * 0.1
        
        # SVD分解
        U, S, Vt = np.linalg.svd(W)
        
        # 低秩近似
        W_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        
        # 计算参数量
        params_full = matrix_size * matrix_size
        params_lora = matrix_size * rank + rank * matrix_size
        compression_ratio = params_lora / params_full
        
        # 计算重构误差
        reconstruction_error = np.linalg.norm(W - W_approx, 'fro') / np.linalg.norm(W, 'fro')
        
        with col2:
            st.markdown("### 📊 压缩效果")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("原始参数量", f"{params_full:,}")
            with col_b:
                st.metric("LoRA参数量", f"{params_lora:,}")
            with col_c:
                st.metric("压缩比", f"{compression_ratio*100:.1f}%")
            
            st.metric("重构误差", f"{reconstruction_error*100:.2f}%")
        
        # 可视化奇异值
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(S)+1)),
            y=S,
            mode='lines+markers',
            name='奇异值',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # 标记保留的奇异值
        fig.add_trace(go.Scatter(
            x=list(range(1, rank+1)),
            y=S[:rank],
            mode='markers',
            name=f'保留前{rank}个',
            marker=dict(size=12, color='red')
        ))
        
        fig.update_layout(
            title="奇异值谱",
            xaxis_title="奇异值索引",
            yaxis_title="奇异值大小",
            yaxis_type="log",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **关键观察**:
        - 奇异值快速衰减说明矩阵具有低秩结构
        - 保留前 $r$ 个奇异值可以用很少的参数近似原矩阵
        - LoRA利用这个性质实现参数高效的模型微调
        """)
    
    @staticmethod
    def _render_xor_lifting():
        """XOR问题的升维解决"""
        st.subheader("🔺 Cover定理：升维解决XOR问题")
        
        st.markdown("""
        **Cover定理**: "将复杂的非线性分类问题投射到高维空间，它更有可能变得线性可分。"
        
        **XOR问题**: 在2D空间中线性不可分
        - 类别0: (0,0), (1,1) 
        - 类别1: (0,1), (1,0)
        
        **升维映射**: $\\phi([x_1, x_2]) = [x_1, x_2, x_1 \\cdot x_2]$
        
        在3D空间中，这些点变得线性可分！
        """)
        
        # 生成XOR数据
        X_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR标签
        
        # 升维到3D
        X_3d = np.column_stack([X_2d, X_2d[:, 0] * X_2d[:, 1]])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 2D空间（不可分）")
            fig_2d = go.Figure()
            
            colors = ['blue' if label == 0 else 'red' for label in y]
            fig_2d.add_trace(go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode='markers+text',
                marker=dict(size=20, color=colors),
                text=['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
                textposition='top center',
                showlegend=False
            ))
            
            # 尝试绘制分隔线（无法完美分开）
            x_line = np.array([-0.2, 1.2])
            fig_2d.add_trace(go.Scatter(
                x=x_line, y=0.5*np.ones_like(x_line),
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='无法分开',
                showlegend=False
            ))
            
            fig_2d.update_layout(
                xaxis_title="x₁",
                yaxis_title="x₂",
                xaxis=dict(range=[-0.3, 1.3]),
                yaxis=dict(range=[-0.3, 1.3]),
                height=400
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 3D空间（可分）")
            fig_3d = go.Figure()
            
            colors_3d = ['blue' if label == 0 else 'red' for label in y]
            fig_3d.add_trace(go.Scatter3d(
                x=X_3d[:, 0],
                y=X_3d[:, 1],
                z=X_3d[:, 2],
                mode='markers+text',
                marker=dict(size=10, color=colors_3d),
                text=['(0,0,0)', '(0,1,0)', '(1,0,0)', '(1,1,1)'],
                textposition='top center',
                showlegend=False
            ))
            
            # 绘制分隔平面 z = 0.5
            xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 10),
                                np.linspace(-0.2, 1.2, 10))
            zz = 0.5 * np.ones_like(xx)
            
            fig_3d.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='分隔平面'
            ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="x₁",
                    yaxis_title="x₂",
                    zaxis_title="x₁·x₂",
                ),
                height=400
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.success("✅ 通过升维到3D空间，XOR问题变得线性可分！平面 z=0.5 可以完美分开两类。")
        
        st.markdown("""
        **深度学习中的应用**:
        - **Embedding层**: 将离散ID映射到高维连续空间
        - **Transformer FFN**: 先升维4倍再降维（解开数据纠缠）
        - **核技巧**: 隐式地在无限维空间进行计算
        """)
    
    @staticmethod
    def _render_pca():
        """PCA降维演示"""
        st.subheader("📊 PCA主成分分析")
        
        st.markdown("""
        **PCA目标**: 找到数据方差最大的方向
        
        **与SVD的关系**:
        - 协方差矩阵的特征向量 = 数据矩阵的右奇异向量
        - 方差 = 奇异值的平方 / (n-1)
        
        $$Cov(X) = \\frac{1}{n-1}X^TX = V\\frac{\\Sigma^2}{n-1}V^T$$
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 参数设置")
            n_samples = st.slider("样本数", 50, 500, 200, 50)
            correlation = st.slider("相关性", 0.0, 0.95, 0.8, 0.05)
            noise = st.slider("噪声", 0.0, 1.0, 0.3, 0.1)
        
        # 生成相关数据
        np.random.seed(42)
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        data = np.random.multivariate_normal(mean, cov, n_samples)
        data += np.random.randn(n_samples, 2) * noise
        
        # 中心化
        data_centered = data - data.mean(axis=0)
        
        # PCA (使用SVD)
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        principal_components = Vt
        
        # 解释方差比例
        explained_variance = (S ** 2) / (n_samples - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()
        
        with col2:
            st.markdown("### 📊 解释方差")
            for i, ratio in enumerate(explained_variance_ratio):
                st.metric(f"PC{i+1} 解释方差", f"{ratio*100:.1f}%")
        
        # 可视化
        fig = go.Figure()
        
        # 原始数据点
        fig.add_trace(go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode='markers',
            marker=dict(size=5, color='lightblue'),
            name='数据点'
        ))
        
        # 主成分方向
        scale = 3
        for i, (pc, var_ratio) in enumerate(zip(principal_components, explained_variance_ratio)):
            fig.add_trace(go.Scatter(
                x=[0, pc[0]*scale*np.sqrt(explained_variance[i])],
                y=[0, pc[1]*scale*np.sqrt(explained_variance[i])],
                mode='lines+markers',
                line=dict(width=3, color='red' if i==0 else 'orange'),
                marker=dict(size=10),
                name=f'PC{i+1} ({var_ratio*100:.1f}%)'
            ))
        
        fig.update_layout(
            title="PCA主成分分析",
            xaxis_title="特征1",
            yaxis_title="特征2",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x', scaleratio=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **观察**:
        - PC1（红色）指向方差最大的方向
        - PC2（橙色）与PC1正交，指向次大方差方向
        - 箭头长度与该方向的方差成正比
        """)
    
    @staticmethod
    def _render_eigenspectrum():
        """特征值谱与训练稳定性"""
        st.subheader("🌋 特征值谱与梯度传播")
        
        st.markdown("""
        **特征值谱的重要性**:
        
        在深度网络中，梯度反向传播涉及权重矩阵的连乘 $W^L$：
        
        - **梯度爆炸**: $\\rho(W) = \\max|\\lambda_i| > 1$ → 梯度指数增长
        - **梯度消失**: $\\rho(W) < 1$ → 梯度指数衰减
        - **理想状态**: $\\rho(W) \\approx 1$ → 梯度稳定传播
        
        其中 $\\rho(W)$ 是谱半径（最大特征值的模）
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 矩阵类型")
            matrix_type = st.radio(
                "选择矩阵",
                ["梯度消失 (ρ<1)", "稳定传播 (ρ≈1)", "梯度爆炸 (ρ>1)", "自定义"]
            )
            
            if matrix_type == "自定义":
                scale = st.slider("谱半径", 0.1, 3.0, 1.0, 0.1)
            
            n_layers = st.slider("网络层数", 5, 50, 20, 5)
        
        # 生成矩阵
        np.random.seed(42)
        if matrix_type == "梯度消失 (ρ<1)":
            W = np.random.randn(10, 10) * 0.5
        elif matrix_type == "稳定传播 (ρ≈1)":
            W = np.random.randn(10, 10)
            W = W / np.linalg.norm(W, 2)  # 谱归一化
        elif matrix_type == "梯度爆炸 (ρ>1)":
            W = np.random.randn(10, 10) * 1.5
        else:
            W = np.random.randn(10, 10)
            W = W / np.linalg.norm(W, 2) * scale
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(W)
        spectral_radius = np.max(np.abs(eigenvalues))
        
        with col2:
            st.metric("谱半径 ρ(W)", f"{spectral_radius:.3f}")
            
            # 预测梯度变化
            gradient_scale = spectral_radius ** n_layers
            if gradient_scale < 1e-10:
                st.error(f"⚠️ {n_layers}层后梯度缩放: ~0 (完全消失)")
            elif gradient_scale > 1e10:
                st.error(f"⚠️ {n_layers}层后梯度缩放: ~∞ (完全爆炸)")
            elif gradient_scale < 0.01:
                st.warning(f"⚠️ {n_layers}层后梯度缩放: {gradient_scale:.2e} (严重消失)")
            elif gradient_scale > 100:
                st.warning(f"⚠️ {n_layers}层后梯度缩放: {gradient_scale:.2e} (严重爆炸)")
            else:
                st.success(f"✅ {n_layers}层后梯度缩放: {gradient_scale:.2f} (相对稳定)")
        
        # 可视化特征值谱
        fig = go.Figure()
        
        # 单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='单位圆',
            showlegend=True
        ))
        
        # 特征值
        fig.add_trace(go.Scatter(
            x=eigenvalues.real,
            y=eigenvalues.imag,
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='特征值',
            showlegend=True
        ))
        
        fig.update_layout(
            title="特征值谱分布",
            xaxis_title="实部",
            yaxis_title="虚部",
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x', scaleratio=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **解决方案**:
        - **谱归一化**: $W_{SN} = W / \\sigma_{max}(W)$
        - **正交初始化**: 使特征值接近单位圆
        - **残差连接**: 跳过连接避免连乘
        - **BatchNorm/LayerNorm**: 稳定激活分布
        """)
    
    @staticmethod
    def _render_hessian():
        """海森矩阵与鞍点"""
        st.subheader("⛰️ 海森矩阵：鞍点 vs 极值点")
        
        st.markdown("""
        **海森矩阵 (Hessian)**: 二阶导数矩阵 $H_{ij} = \\frac{\\partial^2 f}{\\partial x_i \\partial x_j}$
        
        **判定准则**:
        - **正定** (所有特征值 > 0): 局部极小值 🟢
        - **负定** (所有特征值 < 0): 局部极大值 🔴
        - **不定** (特征值有正有负): 鞍点 ⚠️
        
        在高维优化中，**鞍点比局部极小值更常见**！
        """)
        
        function_type = st.radio(
            "选择函数",
            ["局部极小值: f(x,y) = x² + y²",
             "鞍点: f(x,y) = x² - y²",
             "Rosenbrock函数 (复杂地形)"]
        )
        
        # 创建网格
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        if function_type == "局部极小值: f(x,y) = x² + y²":
            Z = X**2 + Y**2
            H = np.array([[2, 0], [0, 2]])
            point_type = "极小值"
            eigenvalues = np.linalg.eigvals(H)
        elif function_type == "鞍点: f(x,y) = x² - y²":
            Z = X**2 - Y**2
            H = np.array([[2, 0], [0, -2]])
            point_type = "鞍点"
            eigenvalues = np.linalg.eigvals(H)
        else:  # Rosenbrock
            a, b = 1, 100
            Z = (a - X)**2 + b * (Y - X**2)**2
            # Hessian at (1, 1) for Rosenbrock
            H = np.array([[802, -400], [-400, 200]])
            point_type = "复杂（接近鞍点）"
            eigenvalues = np.linalg.eigvals(H)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 3D表面图
            fig_3d = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
                )
            )])
            
            fig_3d.update_layout(
                title="函数地形",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="f(x,y)"
                ),
                height=400
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            # 等高线图
            fig_contour = go.Figure(data=go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                )
            ))
            
            # 标记原点
            fig_contour.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name='临界点 (0,0)'
            ))
            
            fig_contour.update_layout(
                title="等高线图",
                xaxis_title="x",
                yaxis_title="y",
                height=400
            )
            
            st.plotly_chart(fig_contour, use_container_width=True)
        
        # 显示海森矩阵分析
        st.markdown("### 📊 海森矩阵分析")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("**海森矩阵 H:**")
            st.latex(r"H = \begin{bmatrix} " + 
                    f"{H[0,0]:.0f} & {H[0,1]:.0f} \\\\ " +
                    f"{H[1,0]:.0f} & {H[1,1]:.0f}" + 
                    r" \end{bmatrix}")
        
        with col_b:
            st.markdown("**特征值:**")
            for i, ev in enumerate(eigenvalues):
                st.write(f"λ{i+1} = {ev:.2f}")
        
        with col_c:
            st.markdown("**判定:**")
            if np.all(eigenvalues > 0):
                st.success(f"🟢 {point_type} (正定)")
            elif np.all(eigenvalues < 0):
                st.error(f"🔴 {point_type} (负定)")
            else:
                st.warning(f"⚠️ {point_type} (不定)")
        
        st.markdown("""
        **为什么鞍点很危险？**
        - 一阶梯度 = 0，优化器会停止
        - 但并非真正的极值点
        - 在某些方向上损失仍可下降
        
        **如何逃离鞍点？**
        - 加入动量（Momentum）
        - 使用二阶方法（牛顿法）
        - 随机梯度的噪声帮助逃离
        """)
