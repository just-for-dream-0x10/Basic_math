"""
交互式L1/L2正则化可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveRegularization:
    """交互式L1/L2正则化可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎯 交互式正则化对比")
        st.markdown("""
        **正则化**: 通过约束模型复杂度来防止过拟合
        
        **优化目标**:
        $$\\min_w \\mathcal{L}(w) + \\lambda R(w)$$
        
        其中:
        - $\\mathcal{L}(w)$: 经验损失 (训练误差)
        - $R(w)$: 正则化项
        - $\\lambda$: 正则化强度
        
        **常见正则化**:
        - **L1 (Lasso)**: $R(w) = \\|w\\|_1 = \\sum_i |w_i|$ (产生稀疏解)
        - **L2 (Ridge)**: $R(w) = \\|w\\|_2^2 = \\sum_i w_i^2$ (权重衰减)
        - **Elastic Net**: $R(w) = \\alpha\\|w\\|_1 + (1-\\alpha)\\|w\\|_2^2$ (混合)
        
        **几何解释**: L1约束为菱形, L2约束为圆形
        """)
        
        with st.sidebar:
            st.markdown("### 📊 正则化设置")
            reg_type = st.selectbox("正则化类型", 
                ["L1 (Lasso)", "L2 (Ridge)", "L1+L2 (Elastic Net)", "无正则化"])
            
            lambda_val = st.slider("λ (正则化强度)", 0.0, 5.0, 1.0, 0.1,
                                  help="控制正则化项的权重")
            
            if reg_type == "L1+L2 (Elastic Net)":
                alpha = st.slider("α (L1/L2混合比例)", 0.0, 1.0, 0.5, 0.05,
                                help="0=纯L2, 1=纯L1")
            
            st.markdown("### 🎲 数据设置")
            n_features = st.slider("特征数量", 5, 50, 20, 5)
            noise_level = st.slider("噪声水平", 0.0, 2.0, 0.5, 0.1)
            n_samples = st.slider("样本数量", 50, 500, 100, 50)
        
        # 生成数据
        X, y, true_weights = InteractiveRegularization._generate_regression_data(
            n_samples, n_features, noise_level
        )
        
        # 训练模型
        if reg_type == "L1 (Lasso)":
            weights = InteractiveRegularization._train_lasso(X, y, lambda_val)
            constraint_shape = "diamond"
        elif reg_type == "L2 (Ridge)":
            weights = InteractiveRegularization._train_ridge(X, y, lambda_val)
            constraint_shape = "circle"
        elif reg_type == "L1+L2 (Elastic Net)":
            alpha_val = alpha if 'alpha' in locals() else 0.5
            weights = InteractiveRegularization._train_elastic_net(X, y, lambda_val, alpha_val)
            constraint_shape = "mixed"
        else:  # 无正则化
            weights = InteractiveRegularization._train_ols(X, y)
            constraint_shape = "none"
        
        # 可视化
        st.markdown("### 📈 权重分布对比")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_weights = InteractiveRegularization._plot_weights_comparison(
                true_weights, weights, reg_type
            )
            st.plotly_chart(fig_weights, use_container_width=True)
        
        with col2:
            st.markdown("#### 统计信息")
            st.metric("非零权重", f"{np.sum(np.abs(weights) > 0.01)}/{len(weights)}")
            st.metric("权重L1范数", f"{np.sum(np.abs(weights)):.3f}")
            st.metric("权重L2范数", f"{np.sqrt(np.sum(weights**2)):.3f}")
            
            # 预测性能
            y_pred = X @ weights
            mse = np.mean((y - y_pred)**2)
            st.metric("训练MSE", f"{mse:.3f}")
        
        # 2D约束可视化（仅选择前2个权重）
        if n_features >= 2:
            st.markdown("### 🎯 约束空间可视化 (前两个权重)")
            fig_constraint = InteractiveRegularization._plot_constraint_space(
                X, y, lambda_val, reg_type, constraint_shape
            )
            st.plotly_chart(fig_constraint, use_container_width=True)
        
        # 正则化路径
        st.markdown("### 📉 正则化路径 (λ变化)")
        fig_path = InteractiveRegularization._plot_regularization_path(
            X, y, reg_type
        )
        st.plotly_chart(fig_path, use_container_width=True)
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("regularization")
        quizzes = QuizTemplates.get_regularization_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _generate_regression_data(n_samples, n_features, noise):
        """生成回归数据"""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # 真实权重：只有少数非零（稀疏）
        true_weights = np.zeros(n_features)
        n_nonzero = max(3, n_features // 5)
        nonzero_idx = np.random.choice(n_features, n_nonzero, replace=False)
        true_weights[nonzero_idx] = np.random.randn(n_nonzero) * 3
        
        y = X @ true_weights + np.random.randn(n_samples) * noise
        
        return X, y, true_weights
    
    @staticmethod
    def _train_ols(X, y):
        """普通最小二乘"""
        return np.linalg.lstsq(X, y, rcond=None)[0]
    
    @staticmethod
    def _train_ridge(X, y, lambda_val):
        """Ridge回归 (L2)"""
        n_features = X.shape[1]
        I = np.eye(n_features)
        return np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y
    
    @staticmethod
    def _train_lasso(X, y, lambda_val):
        """Lasso回归 (L1) - 使用坐标下降"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        
        # 简单的坐标下降算法
        for _ in range(100):  # 迭代次数
            for j in range(n_features):
                residual = y - X @ weights + X[:, j] * weights[j]
                rho = X[:, j] @ residual
                
                # Soft thresholding
                if rho < -lambda_val / 2:
                    weights[j] = (rho + lambda_val / 2) / (X[:, j] @ X[:, j])
                elif rho > lambda_val / 2:
                    weights[j] = (rho - lambda_val / 2) / (X[:, j] @ X[:, j])
                else:
                    weights[j] = 0
        
        return weights
    
    @staticmethod
    def _train_elastic_net(X, y, lambda_val, alpha):
        """Elastic Net (L1 + L2)"""
        l1_weight = alpha * lambda_val
        l2_weight = (1 - alpha) * lambda_val
        
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        
        for _ in range(100):
            for j in range(n_features):
                residual = y - X @ weights + X[:, j] * weights[j]
                rho = X[:, j] @ residual
                z = X[:, j] @ X[:, j] + l2_weight
                
                # Soft thresholding with L2
                if rho < -l1_weight / 2:
                    weights[j] = (rho + l1_weight / 2) / z
                elif rho > l1_weight / 2:
                    weights[j] = (rho - l1_weight / 2) / z
                else:
                    weights[j] = 0
        
        return weights
    
    @staticmethod
    def _plot_weights_comparison(true_weights, learned_weights, reg_type):
        """绘制权重对比图"""
        fig = go.Figure()
        
        indices = np.arange(len(true_weights))
        
        fig.add_trace(go.Bar(
            x=indices,
            y=true_weights,
            name='真实权重',
            marker_color='blue',
            opacity=0.6
        ))
        
        fig.add_trace(go.Bar(
            x=indices,
            y=learned_weights,
            name=f'学习权重 ({reg_type})',
            marker_color='red',
            opacity=0.6
        ))
        
        fig.update_layout(
            title="权重对比",
            xaxis_title="特征索引",
            yaxis_title="权重值",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    @staticmethod
    def _plot_constraint_space(X, y, lambda_val, reg_type, constraint_shape):
        """绘制约束空间（2D）"""
        # 只使用前两个特征
        X_2d = X[:, :2]
        
        # 计算损失函数等高线
        w1_range = np.linspace(-3, 3, 100)
        w2_range = np.linspace(-3, 3, 100)
        W1, W2 = np.meshgrid(w1_range, w2_range)
        
        Loss = np.zeros_like(W1)
        for i in range(len(w1_range)):
            for j in range(len(w2_range)):
                w = np.array([W1[j, i], W2[j, i]])
                y_pred = X_2d @ w
                Loss[j, i] = np.mean((y - y_pred)**2)
        
        fig = go.Figure()
        
        # 损失函数等高线
        fig.add_trace(go.Contour(
            x=w1_range, y=w2_range, z=Loss,
            colorscale='Blues',
            showscale=False,
            contours=dict(start=Loss.min(), end=Loss.min() + 10, size=0.5),
            opacity=0.6,
            name='损失函数'
        ))
        
        # 约束区域
        if constraint_shape == "diamond":  # L1
            # |w1| + |w2| <= lambda
            t = np.linspace(0, 2*np.pi, 100)
            r = lambda_val
            constraint_x = r * np.sign(np.cos(t)) * np.abs(np.cos(t))
            constraint_y = r * np.sign(np.sin(t)) * np.abs(np.sin(t))
            
            fig.add_trace(go.Scatter(
                x=constraint_x, y=constraint_y,
                mode='lines',
                line=dict(color='red', width=3),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                name='L1约束区域'
            ))
        
        elif constraint_shape == "circle":  # L2
            # w1^2 + w2^2 <= lambda^2
            theta = np.linspace(0, 2*np.pi, 100)
            constraint_x = lambda_val * np.cos(theta)
            constraint_y = lambda_val * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=constraint_x, y=constraint_y,
                mode='lines',
                line=dict(color='green', width=3),
                fill='toself',
                fillcolor='rgba(0,255,0,0.2)',
                name='L2约束区域'
            ))
        
        fig.update_layout(
            title="约束空间与损失函数等高线",
            xaxis_title="w₁",
            yaxis_title="w₂",
            height=500,
            xaxis=dict(range=[-3, 3]),
            yaxis=dict(range=[-3, 3], scaleanchor="x")
        )
        
        return fig
    
    @staticmethod
    def _plot_regularization_path(X, y, reg_type):
        """绘制正则化路径"""
        lambdas = np.logspace(-2, 1, 50)
        n_features = min(X.shape[1], 10)  # 只显示前10个特征
        
        weights_path = np.zeros((len(lambdas), n_features))
        
        for i, lam in enumerate(lambdas):
            if reg_type == "L1 (Lasso)":
                w = InteractiveRegularization._train_lasso(X, y, lam)
            elif reg_type == "L2 (Ridge)":
                w = InteractiveRegularization._train_ridge(X, y, lam)
            else:
                w = InteractiveRegularization._train_ols(X, y)
            
            weights_path[i, :] = w[:n_features]
        
        fig = go.Figure()
        
        for j in range(n_features):
            fig.add_trace(go.Scatter(
                x=lambdas,
                y=weights_path[:, j],
                mode='lines',
                name=f'w{j}'
            ))
        
        fig.update_layout(
            title="正则化路径 (权重随λ变化)",
            xaxis_title="λ (正则化强度)",
            yaxis_title="权重值",
            xaxis_type="log",
            height=400
        )
        
        return fig
