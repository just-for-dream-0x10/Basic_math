"""
分类模型的本质优化逻辑 - 交互式可视化
基于 8.TheEssentialOptimizationLogicOfClassificationModels.md

核心内容：
1. 三种优化思路的统一框架
2. 最小二乘法 (Least Squares)
3. 最大似然估计 (Maximum Likelihood)
4. SVM 间隔最大化
5. 损失函数对比
6. 决策边界演化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_blobs


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveClassificationOptimization:
    """交互式分类优化逻辑可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎯 分类模型的本质优化逻辑")
        
        st.markdown(r"""
        **核心问题**: 如何让模型输出 $G(X)$ 看齐真实标签 $T(X)$？
        
        **三种经典思路**:
        1. **最小二乘法** - 数值拟合视角
        2. **最大似然估计** - 概率统计视角
        3. **SVM** - 几何间隔视角
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "三种方法统一对比",
                    "最小二乘法 (LSE)",
                    "最大似然估计 (MLE)",
                    "SVM间隔最大化",
                    "损失函数对比",
                    "决策边界演化",
                    "实战案例"
                ]
            )
        
        if demo_type == "三种方法统一对比":
            InteractiveClassificationOptimization._render_unified_comparison()
        elif demo_type == "最小二乘法 (LSE)":
            InteractiveClassificationOptimization._render_least_squares()
        elif demo_type == "最大似然估计 (MLE)":
            InteractiveClassificationOptimization._render_mle()
        elif demo_type == "SVM间隔最大化":
            InteractiveClassificationOptimization._render_svm()
        elif demo_type == "损失函数对比":
            InteractiveClassificationOptimization._render_loss_comparison()
        elif demo_type == "决策边界演化":
            InteractiveClassificationOptimization._render_boundary_evolution()
        elif demo_type == "实战案例":
            InteractiveClassificationOptimization._render_practical_case()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("classification_optimization")
        quizzes = QuizTemplates.get_classification_optimization_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _generate_binary_data(n_samples=100, noise=0.1, random_state=42):
        """生成二分类数据"""
        np.random.seed(random_state)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=random_state
        )
        # 转换标签为 {-1, 1}
        y = 2 * y - 1
        return X, y
    
    @staticmethod
    def _render_unified_comparison():
        """三种方法的统一对比"""
        st.markdown("### 🔄 三种方法的统一框架")
        
        st.markdown(r"""
        **相同的目标，不同的视角**：
        
        | 方面 | 最小二乘法 | 最大似然估计 | SVM |
        |------|------------|--------------|-----|
        | **度量意义** | 数值拟合 | 概率解释 | 几何距离 |
        | **损失函数** | MSE | Cross-Entropy | Hinge Loss |
        | **关注点** | 所有数据点 | 所有数据点 | 边界附近的支持向量 |
        | **生活比喻** | 🎯 扔飞镖 | 🔍 福尔摩斯破案 | 🛣️ 修最宽马路 |
        """)
        
        # 生成数据
        with st.sidebar:
            st.markdown("#### 数据设置")
            n_samples = st.slider("样本数量", 50, 200, 100, 10)
            noise = st.slider("噪声水平", 0.0, 0.3, 0.1, 0.05)
            random_state = st.slider("随机种子", 0, 100, 42, 1)
        
        X, y = InteractiveClassificationOptimization._generate_binary_data(
            n_samples, noise, random_state
        )
        
        # 训练三种模型
        # 1. 最小二乘法 (用线性回归)
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # 2. 最大似然估计 (逻辑回归)
        log_model = LogisticRegression()
        log_model.fit(X, y)
        
        # 3. SVM
        svm_model = SVC(kernel='linear', C=1.0)
        svm_model.fit(X, y)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "最小二乘法 (MSE)",
                "最大似然估计 (Cross-Entropy)",
                "SVM (Hinge Loss)"
            )
        )
        
        # 创建网格用于绘制决策边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 绘制每个模型的决策边界
        models = [
            (lr_model, 1, "LSE"),
            (log_model, 2, "MLE"),
            (svm_model, 3, "SVM")
        ]
        
        for model, col, name in models:
            # 预测
            if name == "LSE":
                Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
            elif name == "MLE":
                Z = log_model.predict(np.c_[xx.ravel(), yy.ravel()])
            else:  # SVM
                Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
            
            Z = Z.reshape(xx.shape)
            
            # 决策边界
            fig.add_trace(
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=Z,
                    colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
                    showscale=False,
                    opacity=0.3,
                    contours=dict(start=-1, end=1, size=2),
                    hoverinfo='skip'
                ),
                row=1, col=col
            )
            
            # 数据点
            for label in [-1, 1]:
                mask = y == label
                fig.add_trace(
                    go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        name=f'Class {label}' if col == 1 else None,
                        marker=dict(
                            size=8,
                            color='blue' if label == -1 else 'red',
                            line=dict(width=1, color='white')
                        ),
                        showlegend=(col == 1)
                    ),
                    row=1, col=col
                )
            
            # 添加决策边界线
            if name == "SVM":
                # 对于SVM，突出显示支持向量
                sv_mask = np.zeros(len(X), dtype=bool)
                sv_mask[svm_model.support_] = True
                fig.add_trace(
                    go.Scatter(
                        x=X[sv_mask, 0],
                        y=X[sv_mask, 1],
                        mode='markers',
                        name='Support Vectors' if col == 3 else None,
                        marker=dict(
                            size=12,
                            color='yellow',
                            symbol='circle-open',
                            line=dict(width=3, color='black')
                        ),
                        showlegend=(col == 3)
                    ),
                    row=1, col=col
                )
        
        fig.update_xaxes(title_text="Feature 1")
        fig.update_yaxes(title_text="Feature 2")
        fig.update_layout(height=500, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能对比
        st.markdown("### 📊 性能对比")
        
        # 计算准确率
        lr_acc = np.mean((lr_model.predict(X) > 0) == (y > 0))
        log_acc = log_model.score(X, y)
        svm_acc = svm_model.score(X, y)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最小二乘法", f"{lr_acc:.1%}")
        with col2:
            st.metric("最大似然估计", f"{log_acc:.1%}")
        with col3:
            st.metric("SVM", f"{svm_acc:.1%}")
        
        st.success(r"""
        **关键观察**:
        
        1. **决策边界差异**:
           - LSE: 直线，但可能受离群点影响大
           - MLE: 平滑的概率边界
           - SVM: 最大化间隔的边界，只关心支持向量
        
        2. **鲁棒性**:
           - LSE对离群点敏感
           - MLE相对平衡
           - SVM最鲁棒（忽略远离边界的点）
        
        3. **适用场景**:
           - LSE: 简单快速，但不推荐用于分类
           - MLE: 现代深度学习的标准选择
           - SVM: 小样本、高维数据的经典方法
        """)

    
    @staticmethod
    def _render_least_squares():
        """最小二乘法演示"""
        st.markdown("### 🎯 最小二乘法：扔飞镖游戏")
        
        st.markdown(r"""
        **核心思想**: 把模型输出看作连续数值，直接拟合真实标签
        
        **损失函数**:
        """)
        
        st.latex(r"""
        \mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^n (f(x_i) - y_i)^2
        """)
        
        st.markdown(r"""
        **生活比喻**: 🎯 扔飞镖
        - 真实标签是靶心
        - 模型输出是飞镖落点
        - 计算每个飞镖离靶心的**平方距离**
        - 目标：让总平方距离最小
        """)
        
        # 生成数据
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_samples = st.slider("样本数量", 50, 200, 100, 10)
            show_outlier = st.checkbox("添加离群点", value=False)
            show_confident = st.checkbox("展示'太好'预测的惩罚", value=False)
        
        X, y = InteractiveClassificationOptimization._generate_binary_data(n_samples)
        
        # 添加离群点
        if show_outlier:
            outlier_x = np.array([[X[:, 0].max() - 0.5, X[:, 1].max() - 0.5]])
            outlier_y = np.array([-1])  # 与周围点相反
            X = np.vstack([X, outlier_x])
            y = np.hstack([y, outlier_y])
        
        # 训练模型
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # 预测
        y_pred = lr_model.predict(X)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "决策边界与数据点",
                "平方误差分布"
            )
        )
        
        # 左图：决策边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale='RdBu',
                showscale=True,
                opacity=0.5,
                colorbar=dict(x=0.45)
            ),
            row=1, col=1
        )
        
        # 数据点
        for label in [-1, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=f'y={label}',
                    marker=dict(
                        size=10,
                        color='blue' if label == -1 else 'red',
                        line=dict(width=1, color='white')
                    )
                ),
                row=1, col=1
            )
        
        # 如果有离群点，特别标注
        if show_outlier:
            fig.add_trace(
                go.Scatter(
                    x=[X[-1, 0]],
                    y=[X[-1, 1]],
                    mode='markers',
                    name='离群点',
                    marker=dict(
                        size=15,
                        color='yellow',
                        symbol='star',
                        line=dict(width=2, color='black')
                    )
                ),
                row=1, col=1
            )
        
        # 右图：平方误差
        squared_errors = (y_pred - y) ** 2
        colors = ['lightgreen' if e < 1 else 'orange' if e < 4 else 'red' 
                  for e in squared_errors]
        
        fig.add_trace(
            go.Bar(
                y=squared_errors,
                marker=dict(color=colors),
                showlegend=False,
                text=[f'{e:.2f}' for e in squared_errors],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Feature 1", row=1, col=1)
        fig.update_yaxes(title_text="Feature 2", row=1, col=1)
        fig.update_xaxes(title_text="样本索引", row=1, col=2)
        fig.update_yaxes(title_text="平方误差", row=1, col=2)
        fig.update_layout(height=500, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 误差分析")
        
        mse = np.mean(squared_errors)
        max_error_idx = np.argmax(squared_errors)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均平方误差", f"{mse:.3f}")
        with col2:
            st.metric("最大误差", f"{squared_errors[max_error_idx]:.3f}")
        with col3:
            st.metric("准确率", f"{np.mean((y_pred > 0) == (y > 0)):.1%}")
        
        # 展示"太好"预测的问题
        if show_confident:
            st.warning(r"""
            **问题：惩罚"太好"的预测**
            
            假设真实标签 $y = 1$：
            - 预测 $f(x) = 1.1$: 误差 = $(1.1 - 1)^2 = 0.01$ ✅ 很好
            - 预测 $f(x) = 5.0$: 误差 = $(5.0 - 1)^2 = 16.0$ ❌ 非常差！
            
            但在分类任务中，$f(x) = 5.0$ 表示模型**非常确信**这是正类，
            这应该被奖励，而不是惩罚！
            
            这就是为什么**MSE不适合分类**的根本原因。
            """)
        
        st.info(r"""
        **最小二乘法的三大问题**:
        
        1. **逻辑尴尬** 💭
           - 把 $\{-1, +1\}$ 这样的类别标签当作连续数值拟合
           - 输出可能是 $-5$ 或 $10$，但标签只能是 $-1$ 或 $1$
        
        2. **离群点敏感** 🎯
           - 一个离群点产生巨大的平方误差
           - 可能把整个模型"带偏"
           - 试试添加离群点看效果！
        
        3. **惩罚"太好"的预测** ⚠️
           - 模型很有信心的正确预测反而被重罚
           - 违背分类任务的直觉
        
        **结论**: MSE适合回归，不适合分类！
        """)

    
    @staticmethod
    def _render_mle():
        """最大似然估计演示"""
        st.markdown("### 🔍 最大似然估计：福尔摩斯破案")
        
        st.markdown(r"""
        **核心思想**: 把模型输出转换为概率，最大化观测数据的似然
        
        **从输出到概率**:
        """)
        
        st.latex(r"""
        p(y=1|x) = \sigma(f(x)) = \frac{1}{1 + e^{-f(x)}}
        """)
        
        st.markdown(r"""
        **损失函数 (交叉熵)**:
        """)
        
        st.latex(r"""
        \mathcal{L}_{CE} = -\sum_{i=1}^n [y_i \log p_i + (1-y_i) \log(1-p_i)]
        """)
        
        st.markdown(r"""
        **生活比喻**: 🔍 福尔摩斯破案
        - 训练数据是一串脚印（线索）
        - 模型参数 $\theta$ 是不同的嫌疑人
        - **目标**: 找那个最有可能留下这串脚印的嫌疑人
        """)
        
        # Sigmoid函数演示
        with st.sidebar:
            st.markdown("#### 参数设置")
            show_sigmoid = st.checkbox("显示Sigmoid转换", value=True)
            n_samples = st.slider("样本数量", 50, 200, 100, 10)
        
        X, y = InteractiveClassificationOptimization._generate_binary_data(n_samples)
        
        # 转换y到{0, 1}用于逻辑回归
        y_binary = (y + 1) // 2
        
        # 训练模型
        log_model = LogisticRegression()
        log_model.fit(X, y_binary)
        
        # 预测概率
        y_proba = log_model.predict_proba(X)[:, 1]
        
        # 可视化
        if show_sigmoid:
            # 显示Sigmoid函数
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "Sigmoid函数：从输出到概率",
                    "概率分布"
                )
            )
            
            # 左图：Sigmoid函数
            z = np.linspace(-6, 6, 100)
            sigmoid = 1 / (1 + np.exp(-z))
            
            fig.add_trace(
                go.Scatter(
                    x=z,
                    y=sigmoid,
                    mode='lines',
                    name='σ(z)',
                    line=dict(color='purple', width=3)
                ),
                row=1, col=1
            )
            
            # 标注关键点
            key_points = [(-2, 1/(1+np.exp(2))), (0, 0.5), (2, 1/(1+np.exp(-2)))]
            for z_val, sig_val in key_points:
                fig.add_trace(
                    go.Scatter(
                        x=[z_val],
                        y=[sig_val],
                        mode='markers+text',
                        marker=dict(size=10, color='red'),
                        text=[f'({z_val:.0f}, {sig_val:.2f})'],
                        textposition='top center',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
            
            # 右图：概率分布
            fig.add_trace(
                go.Histogram(
                    x=y_proba[y_binary == 0],
                    name='y=0 (负类)',
                    marker_color='blue',
                    opacity=0.6,
                    nbinsx=20
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=y_proba[y_binary == 1],
                    name='y=1 (正类)',
                    marker_color='red',
                    opacity=0.6,
                    nbinsx=20
                ),
                row=1, col=2
            )
            
            fig.add_vline(x=0.5, line_dash="dash", line_color="green", 
                         annotation_text="决策阈值", row=1, col=2)
            
            fig.update_xaxes(title_text="f(x) (模型输出)", row=1, col=1)
            fig.update_yaxes(title_text="σ(f(x)) (概率)", row=1, col=1)
            fig.update_xaxes(title_text="预测概率", row=1, col=2)
            fig.update_yaxes(title_text="样本数量", row=1, col=2)
            fig.update_layout(height=500, showlegend=True, barmode='overlay')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 决策边界和置信度
        st.markdown("### 🎨 决策边界与置信度")
        
        fig2 = go.Figure()
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 预测概率
        Z = log_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # 绘制概率热图
        fig2.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title="P(y=1)"),
                contours=dict(
                    start=0,
                    end=1,
                    size=0.1
                )
            )
        )
        
        # 数据点，大小表示置信度
        for label in [0, 1]:
            mask = y_binary == label
            confidences = y_proba[mask] if label == 1 else (1 - y_proba[mask])
            
            fig2.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=f'y={label}',
                    marker=dict(
                        size=10 + 20 * confidences,  # 大小表示置信度
                        color='white' if label == 0 else 'red',
                        line=dict(width=2, color='black')
                    ),
                    text=[f'P={p:.2f}' for p in (y_proba[mask] if label == 1 else 1-y_proba[mask])],
                    hovertemplate='%{text}<extra></extra>'
                )
            )
        
        fig2.update_layout(
            title="决策边界与预测置信度（圆圈大小表示置信度）",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 性能指标
        st.markdown("### 📊 模型性能")
        
        y_pred = (y_proba > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_binary)
        avg_confidence = np.mean(np.maximum(y_proba, 1 - y_proba))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("准确率", f"{accuracy:.1%}")
        with col2:
            st.metric("平均置信度", f"{avg_confidence:.1%}")
        with col3:
            # 计算交叉熵
            epsilon = 1e-15
            ce = -np.mean(y_binary * np.log(y_proba + epsilon) + 
                         (1 - y_binary) * np.log(1 - y_proba + epsilon))
            st.metric("交叉熵", f"{ce:.3f}")
        
        st.success(r"""
        **最大似然估计的优势**:
        
        1. **概率解释** 🎲
           - 输出不是硬分类，而是概率
           - 告诉我们模型的"确信程度"
           - 可以设置不同的决策阈值
        
        2. **合理的损失函数** ✅
           - 奖励正确且有信心的预测
           - 不惩罚"太好"的预测
           - 对所有点都有梯度信号
        
        3. **现代标准** 🌟
           - 神经网络分类的标准损失函数
           - Softmax + Cross-Entropy
           - 从Logistic Regression到深度学习
        
        **为什么交叉熵更好？**
        - 分类正确且概率→1时，loss→0
        - 分类错误且概率→0时，loss→∞
        - 提供持续的优化动力
        """)

    
    @staticmethod
    def _render_svm():
        """SVM间隔最大化演示"""
        st.markdown("### 🛣️ SVM：修最宽的马路")
        
        st.markdown(r"""
        **核心思想**: 在类别之间修一条最宽的"双黄线"（最大化间隔）
        
        **优化目标**:
        """)
        
        st.latex(r"""
        \begin{cases}
        \min \frac{1}{2} \|w\|^2  & \text{(让马路尽可能宽)} \\
        \text{s.t. } y_i(w^T x_i + b) \ge 1 & \text{(所有人都在路两边)}
        \end{cases}
        """)
        
        st.markdown(r"""
        **Hinge Loss**:
        """)
        
        st.latex(r"""
        \mathcal{L}_{Hinge}(y, f(x)) = \max(0, 1 - y \cdot f(x))
        """)
        
        st.markdown(r"""
        **生活比喻**: 🛣️ 修马路
        - 要把两类人完全隔开
        - 在中间修一条最宽的双黄线
        - 死死顶在路边缘的人是"支持向量"
        """)
        
        # 生成数据
        with st.sidebar:
            st.markdown("#### 参数设置")
            C = st.slider("C (软间隔惩罚)", 0.1, 10.0, 1.0, 0.1)
            n_samples = st.slider("样本数量", 50, 200, 100, 10)
            show_margin = st.checkbox("显示间隔带", value=True)
        
        X, y = InteractiveClassificationOptimization._generate_binary_data(n_samples)
        
        # 训练SVM
        svm_model = SVC(kernel='linear', C=C)
        svm_model.fit(X, y)
        
        # 获取参数
        w = svm_model.coef_[0]
        b = svm_model.intercept_[0]
        
        # 支持向量
        support_vectors = X[svm_model.support_]
        
        # 可视化
        fig = go.Figure()
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 预测
        Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和间隔
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Decision Function"),
                contours=dict(
                    start=-3,
                    end=3,
                    size=0.5
                )
            )
        )
        
        # 决策边界 (f(x) = 0)
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                showscale=False,
                contours=dict(
                    start=0,
                    end=0,
                    coloring='lines'
                ),
                line=dict(color='black', width=3),
                name='决策边界'
            )
        )
        
        # 间隔边界
        if show_margin:
            for margin_val, color, name in [(1, 'green', '正间隔'), (-1, 'blue', '负间隔')]:
                fig.add_trace(
                    go.Contour(
                        x=xx[0],
                        y=yy[:, 0],
                        z=Z,
                        showscale=False,
                        contours=dict(
                            start=margin_val,
                            end=margin_val,
                            coloring='lines'
                        ),
                        line=dict(color=color, width=2, dash='dash'),
                        name=name
                    )
                )
        
        # 数据点
        for label in [-1, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(
                        size=8,
                        color='blue' if label == -1 else 'red',
                        line=dict(width=1, color='white')
                    )
                )
            )
        
        # 支持向量
        fig.add_trace(
            go.Scatter(
                x=support_vectors[:, 0],
                y=support_vectors[:, 1],
                mode='markers',
                name='支持向量',
                marker=dict(
                    size=15,
                    color='yellow',
                    symbol='circle-open',
                    line=dict(width=3, color='black')
                )
            )
        )
        
        # 计算间隔宽度
        margin_width = 2 / np.linalg.norm(w)
        
        fig.update_layout(
            title=f"SVM决策边界与间隔 (C={C}, 间隔宽度={margin_width:.3f})",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 SVM分析")
        
        n_support = len(svm_model.support_)
        accuracy = svm_model.score(X, y)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("支持向量数量", f"{n_support}")
        with col2:
            st.metric("间隔宽度", f"{margin_width:.3f}")
        with col3:
            st.metric("准确率", f"{accuracy:.1%}")
        
        # Hinge Loss演示
        st.markdown("### 📉 Hinge Loss的特性")
        
        # 计算每个点的Hinge Loss
        decision_values = svm_model.decision_function(X)
        hinge_losses = np.maximum(0, 1 - y * decision_values)
        
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Hinge Loss vs 样本",
                "Hinge Loss函数"
            )
        )
        
        # 左图：每个样本的loss
        colors = ['green' if loss == 0 else 'orange' if loss < 1 else 'red' 
                  for loss in hinge_losses]
        
        fig2.add_trace(
            go.Bar(
                y=hinge_losses,
                marker=dict(color=colors),
                showlegend=False,
                text=[f'{loss:.2f}' for loss in hinge_losses],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 右图：Hinge Loss函数
        margin = np.linspace(-2, 3, 100)
        hinge = np.maximum(0, 1 - margin)
        
        fig2.add_trace(
            go.Scatter(
                x=margin,
                y=hinge,
                mode='lines',
                name='Hinge Loss',
                line=dict(color='purple', width=3)
            ),
            row=1, col=2
        )
        
        # 标注关键区域
        fig2.add_vrect(x0=-2, x1=1, fillcolor="red", opacity=0.1, 
                      annotation_text="Loss > 0", row=1, col=2)
        fig2.add_vrect(x0=1, x1=3, fillcolor="green", opacity=0.1,
                      annotation_text="Loss = 0", row=1, col=2)
        fig2.add_vline(x=1, line_dash="dash", line_color="black", row=1, col=2)
        
        fig2.update_xaxes(title_text="样本索引", row=1, col=1)
        fig2.update_yaxes(title_text="Hinge Loss", row=1, col=1)
        fig2.update_xaxes(title_text="y·f(x) (函数间隔)", row=1, col=2)
        fig2.update_yaxes(title_text="Loss", row=1, col=2)
        fig2.update_layout(height=400)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.success(r"""
        **SVM的独特之处**:
        
        1. **只关心边界** 🎯
           - 只有支持向量（黄色圈）影响决策边界
           - 远离边界的点Loss=0，不参与优化
           - 这使得SVM对远离边界的噪声不敏感
        
        2. **最大化间隔** 📏
           - 间隔宽度 = $\frac{2}{\|w\|}$
           - 最小化 $\|w\|^2$ ⟺ 最大化间隔
           - 大间隔 → 更好的泛化能力（低VC维）
        
        3. **Hinge Loss的智慧** 💡
           - 当 $y \cdot f(x) \ge 1$: Loss = 0（已经很好了，不管它）
           - 当 $y \cdot f(x) < 1$: Loss = $1 - y \cdot f(x)$（需要改进）
           - 既不惩罚"太好"，也不过度关注"已经够好"的点
        
        4. **C参数的作用** ⚖️
           - C小：允许更多误分类，追求更大间隔（软间隔）
           - C大：减少误分类，可能间隔变小
           - 调整C观察支持向量和间隔的变化！
        """)

    
    @staticmethod
    def _render_loss_comparison():
        """损失函数对比"""
        st.markdown("### 📉 三种损失函数的对比")
        
        st.markdown(r"""
        **核心问题**: 对于相同的预测误差，三种损失函数如何反应？
        
        设定：真实标签 $y = 1$，预测值 $f(x)$ 从 -3 到 3
        """)
        
        # 生成数据
        f_x = np.linspace(-3, 3, 200)
        y = 1  # 假设真实标签为1
        
        # 三种损失函数
        # 1. MSE
        mse_loss = (f_x - y) ** 2
        
        # 2. Cross-Entropy (通过sigmoid转换)
        sigmoid = 1 / (1 + np.exp(-f_x))
        ce_loss = -np.log(sigmoid + 1e-15)
        
        # 3. Hinge Loss
        hinge_loss = np.maximum(0, 1 - y * f_x)
        
        # 可视化
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=f_x,
                y=mse_loss,
                mode='lines',
                name='MSE',
                line=dict(color='blue', width=3)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=f_x,
                y=ce_loss,
                mode='lines',
                name='Cross-Entropy',
                line=dict(color='red', width=3)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=f_x,
                y=hinge_loss,
                mode='lines',
                name='Hinge Loss',
                line=dict(color='green', width=3)
            )
        )
        
        # 标注关键区域
        fig.add_vline(x=0, line_dash="dash", line_color="gray", 
                     annotation_text="决策边界")
        fig.add_vline(x=1, line_dash="dash", line_color="orange",
                     annotation_text="SVM间隔")
        
        # 标注区域
        fig.add_vrect(x0=-3, x1=0, fillcolor="red", opacity=0.1,
                     annotation_text="分类错误", annotation_position="top left")
        fig.add_vrect(x0=0, x1=1, fillcolor="yellow", opacity=0.1,
                     annotation_text="正确但不够自信", annotation_position="top left")
        fig.add_vrect(x0=1, x1=3, fillcolor="green", opacity=0.1,
                     annotation_text="正确且自信", annotation_position="top left")
        
        fig.update_layout(
            title="三种损失函数对比 (y=1)",
            xaxis_title="f(x) (模型输出)",
            yaxis_title="Loss",
            yaxis_range=[0, 10],
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 🔍 关键观察")
        
        # 创建对比表
        import pandas as pd
        
        scenarios = [
            {"f(x)": -2, "情况": "严重错误"},
            {"f(x)": -0.5, "情况": "轻微错误"},
            {"f(x)": 0.5, "情况": "正确但不自信"},
            {"f(x)": 1.5, "情况": "正确且自信"},
            {"f(x)": 3.0, "情况": "非常自信"}
        ]
        
        for scenario in scenarios:
            fx = scenario["f(x)"]
            scenario["MSE"] = f"{(fx - y)**2:.2f}"
            scenario["Cross-Entropy"] = f"{-np.log(1/(1+np.exp(-fx)) + 1e-15):.2f}"
            scenario["Hinge"] = f"{max(0, 1 - fx):.2f}"
        
        df = pd.DataFrame(scenarios)
        st.dataframe(df, use_container_width=True)
        
        # 三列对比
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 MSE")
            st.warning("""
            **问题**:
            - ❌ 惩罚"太自信"的正确预测
            - f(x)=3时，Loss=4（很大！）
            - 不符合分类任务的直觉
            - 梯度在远处很大，可能不稳定
            """)
        
        with col2:
            st.markdown("#### 🔍 Cross-Entropy")
            st.success("""
            **优势**:
            - ✅ 正确且自信时Loss→0
            - ✅ 错误时Loss→∞
            - ✅ 处处有梯度
            - ✅ 现代深度学习标准
            """)
        
        with col3:
            st.markdown("#### 🛣️ Hinge Loss")
            st.info("""
            **特点**:
            - ✅ f(x)>1时Loss=0
            - ✅ 不关心"已经够好"的点
            - ⚠️ 不可微（在f(x)=1处）
            - 🎯 SVM的选择
            """)
        
        # 梯度对比
        st.markdown("### 📊 梯度对比")
        
        # 计算梯度
        dx = f_x[1] - f_x[0]
        mse_grad = np.gradient(mse_loss, dx)
        ce_grad = np.gradient(ce_loss, dx)
        hinge_grad = np.gradient(hinge_loss, dx)
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(x=f_x, y=mse_grad, mode='lines',
                                  name='MSE梯度', line=dict(color='blue', width=2)))
        fig2.add_trace(go.Scatter(x=f_x, y=ce_grad, mode='lines',
                                  name='CE梯度', line=dict(color='red', width=2)))
        fig2.add_trace(go.Scatter(x=f_x, y=hinge_grad, mode='lines',
                                  name='Hinge梯度', line=dict(color='green', width=2)))
        
        fig2.update_layout(
            title="损失函数梯度对比",
            xaxis_title="f(x)",
            yaxis_title="∂Loss/∂f(x)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info(r"""
        **梯度的含义**:
        
        1. **MSE梯度** 📈
           - 线性增长：$\frac{\partial}{\partial f}[(f-y)^2] = 2(f-y)$
           - 远离标签时梯度很大
           - 可能导致训练不稳定
        
        2. **Cross-Entropy梯度** 🎯
           - 自适应：错误时梯度大，正确时梯度小
           - $\frac{\partial CE}{\partial f} = \sigma(f) - y$
           - 提供持续但合理的优化信号
        
        3. **Hinge梯度** ⚡
           - 阶跃函数：要么-1要么0
           - f(x)>1时梯度=0（不再优化）
           - 节省计算，但可能错过进一步优化
        """)

    
    @staticmethod
    def _render_boundary_evolution():
        """决策边界演化"""
        st.markdown("### 🎬 决策边界的训练过程")
        
        st.markdown(r"""
        **观察**: 三种方法如何从随机初始化逐步学习到决策边界
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_samples = st.slider("样本数量", 50, 200, 100, 10)
            animation_steps = st.slider("显示的训练步数", 5, 20, 10, 1)
        
        X, y = InteractiveClassificationOptimization._generate_binary_data(n_samples)
        y_binary = (y + 1) // 2
        
        # 为了演示演化，我们记录训练过程中的参数
        # 这里简化处理：展示最终结果的不同阶段
        st.info("💡 本演示展示三种方法的最终决策边界对比")
        
        # 训练三种模型
        lr_model = LinearRegression().fit(X, y)
        log_model = LogisticRegression(max_iter=1000).fit(X, y_binary)
        svm_model = SVC(kernel='linear', C=1.0).fit(X, y)
        
        # 创建对比图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "数据分布",
                "最小二乘法",
                "最大似然估计",
                "SVM"
            )
        )
        
        # 准备网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 子图1: 原始数据
        for label in [-1, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(
                        size=8,
                        color='blue' if label == -1 else 'red'
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 子图2-4: 三种方法的决策边界
        models_info = [
            (lr_model, 1, 2, "LSE"),
            (log_model, 2, 1, "MLE"),
            (svm_model, 2, 2, "SVM")
        ]
        
        for model, row, col, name in models_info:
            # 预测
            if name == "LSE":
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            elif name == "MLE":
                Z = 2 * model.predict(np.c_[xx.ravel(), yy.ravel()]) - 1
            else:  # SVM
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            
            Z = Z.reshape(xx.shape)
            
            # 决策边界背景
            fig.add_trace(
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=Z,
                    colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
                    showscale=False,
                    opacity=0.3,
                    contours=dict(start=-1, end=1, size=2),
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # 数据点
            for label in [-1, 1]:
                mask = y == label
                fig.add_trace(
                    go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='blue' if label == -1 else 'red'
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # SVM的支持向量
            if name == "SVM":
                sv_mask = np.zeros(len(X), dtype=bool)
                sv_mask[model.support_] = True
                fig.add_trace(
                    go.Scatter(
                        x=X[sv_mask, 0],
                        y=X[sv_mask, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='yellow',
                            symbol='circle-open',
                            line=dict(width=3, color='black')
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 收敛特性对比
        st.markdown("### 📈 收敛特性对比")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 最小二乘法")
            st.info("""
            **收敛速度**: ⚡ 最快
            - 闭式解：$(X^TX)^{-1}X^Ty$
            - 一步到位
            - 但结果可能不理想
            """)
        
        with col2:
            st.markdown("#### 最大似然估计")
            st.info("""
            **收敛速度**: 🐢 较慢
            - 需要迭代优化
            - 梯度下降/Newton法
            - 通常几十到几百次迭代
            """)
        
        with col3:
            st.markdown("#### SVM")
            st.info("""
            **收敛速度**: 🚀 中等
            - 二次规划问题
            - SMO算法高效
            - 只需优化支持向量
            """)
        
        st.success("""
        **训练效率对比**:
        
        - **LSE**: 计算最快，但分类效果差
        - **MLE**: 平衡性能与效率，最常用
        - **SVM**: 在中小规模数据上很高效，但大规模数据较慢
        """)
    
    @staticmethod
    def _render_practical_case():
        """实战案例"""
        st.markdown("### 🎮 实战案例：交互式实验")
        
        st.markdown("""
        **探索空间**: 调整数据分布，观察三种方法的表现
        """)
        
        with st.sidebar:
            st.markdown("#### 数据生成")
            n_samples = st.slider("样本数量", 50, 300, 100, 10)
            noise_level = st.slider("噪声水平", 0.0, 0.5, 0.1, 0.05)
            n_outliers = st.slider("离群点数量", 0, 20, 0, 1)
            separability = st.slider("可分性", 0.5, 3.0, 1.5, 0.1)
            
            st.markdown("#### 模型选择")
            show_lse = st.checkbox("最小二乘法", value=True)
            show_mle = st.checkbox("最大似然估计", value=True)
            show_svm = st.checkbox("SVM", value=True)
            
            if show_svm:
                svm_c = st.slider("SVM的C参数", 0.1, 10.0, 1.0, 0.1)
        
        # 生成数据
        np.random.seed(42)
        
        # 使用make_blobs生成更可控的数据
        from sklearn.datasets import make_blobs
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[[-separability, -separability], [separability, separability]],
            cluster_std=1.0 + noise_level * 2,
            random_state=42
        )
        y = 2 * y - 1  # 转换为{-1, 1}
        
        # 添加离群点
        if n_outliers > 0:
            outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
            y[outlier_indices] = -y[outlier_indices]
        
        y_binary = (y + 1) // 2
        
        # 训练模型
        models = {}
        if show_lse:
            models['LSE'] = LinearRegression().fit(X, y)
        if show_mle:
            models['MLE'] = LogisticRegression().fit(X, y_binary)
        if show_svm:
            models['SVM'] = SVC(kernel='linear', C=svm_c).fit(X, y)
        
        # 可视化
        fig = go.Figure()
        
        # 网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 绘制每个模型的决策边界
        colors_map = {'LSE': 'blue', 'MLE': 'green', 'SVM': 'purple'}
        
        for name, model in models.items():
            if name == "LSE":
                decision = model.predict(np.c_[xx.ravel(), yy.ravel()])
            elif name == "MLE":
                decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:  # SVM
                decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            
            decision = decision.reshape(xx.shape)
            
            # 只画决策边界线
            fig.add_trace(
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=decision,
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=0,
                        coloring='lines'
                    ),
                    line=dict(color=colors_map[name], width=3),
                    name=name
                )
            )
        
        # 数据点
        for label in [-1, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(
                        size=8,
                        color='lightblue' if label == -1 else 'lightcoral',
                        line=dict(width=1, color='darkblue' if label == -1 else 'darkred')
                    )
                )
            )
        
        fig.update_layout(
            title="三种方法的决策边界对比",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能对比
        st.markdown("### 📊 性能对比")
        
        results = []
        for name, model in models.items():
            if name == "LSE":
                y_pred = (model.predict(X) > 0).astype(int)
                acc = np.mean((y_pred == 1) == (y == 1))
            elif name == "MLE":
                acc = model.score(X, y_binary)
            else:  # SVM
                acc = model.score(X, y)
            
            results.append({'方法': name, '准确率': f"{acc:.1%}"})
        
        import pandas as pd
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        
        st.success("""
        **实验建议**:
        
        1. **调整可分性**: 观察数据越难分时，三种方法的差异
        2. **添加离群点**: LSE受影响最大，SVM最鲁棒
        3. **增加噪声**: 观察哪种方法更稳定
        4. **调整SVM的C**: 大C追求准确，小C追求大间隔
        
        **结论**: 
        - 简单问题：三种方法都可以
        - 有噪声/离群点：SVM > MLE > LSE
        - 需要概率输出：MLE最佳
        - 追求速度：LSE最快（但不推荐分类）
        """)


# 注册到__all__
__all__ = ['InteractiveClassificationOptimization']

