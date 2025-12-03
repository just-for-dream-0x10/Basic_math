"""
交互式拉格朗日乘子法可视化
严格按照 4.Lagrange_Multiplier.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveLagrange:
    """交互式拉格朗日乘子法可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎯 交互式拉格朗日乘子法")
        st.markdown("""
        **核心思想**: 在约束条件下寻找目标函数的极值
        
        拉格朗日函数: $\\mathcal{L}(x, y, \\lambda) = f(x, y) - \\lambda g(x, y)$
        
        最优条件: $\\nabla f = \\lambda \\nabla g$ (梯度平行)
        """)
        
        with st.sidebar:
            st.markdown("### 📊 问题设置")
            problem_type = st.selectbox("选择问题类型", 
                ["圆形约束-线性目标", "椭圆约束-二次目标", 
                 "SVM对偶问题", "KKT条件演示"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if problem_type == "圆形约束-线性目标":
            InteractiveLagrange._render_circle_linear()
        elif problem_type == "椭圆约束-二次目标":
            InteractiveLagrange._render_ellipse_quadratic()
        elif problem_type == "SVM对偶问题":
            InteractiveLagrange._render_svm_dual()
        elif problem_type == "KKT条件演示":
            InteractiveLagrange._render_kkt()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("lagrange")
        quizzes = QuizTemplates.get_lagrange_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_circle_linear():
        """
        问题1: 圆形约束下的线性目标函数
        
        目标函数: f(x, y) = x + y (最大化)
        约束条件: g(x, y) = x² + y² - 1 = 0 (单位圆)
        
        拉格朗日函数: L(x, y, λ) = x + y - λ(x² + y² - 1)
        """
        st.markdown("### 📐 问题1: 在单位圆上最大化 f(x,y) = x + y")
        
        st.latex(r"""
        \begin{aligned}
        \text{maximize:} \quad & f(x, y) = x + y \\
        \text{subject to:} \quad & g(x, y) = x^2 + y^2 - 1 = 0
        \end{aligned}
        """)
        
        with st.sidebar:
            show_gradient = st.checkbox("显示梯度向量", value=True)
            show_contour = st.checkbox("显示等高线", value=True)
        
        # 创建网格
        x = np.linspace(-1.5, 1.5, 300)
        y = np.linspace(-1.5, 1.5, 300)
        X, Y = np.meshgrid(x, y)
        
        # 目标函数值
        F = X + Y
        
        # 约束条件 (圆)
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        # 解析解: ∇f = λ∇g
        # ∇f = (1, 1)
        # ∇g = (2x, 2y)
        # => 1 = 2λx, 1 = 2λy
        # => x = y = 1/(2λ)
        # 代入约束: 2/(4λ²) = 1 => λ = 1/√2
        # => x* = y* = 1/√2
        
        x_opt = 1/np.sqrt(2)
        y_opt = 1/np.sqrt(2)
        lambda_opt = 1/np.sqrt(2)
        
        fig = go.Figure()
        
        # 等高线
        if show_contour:
            fig.add_trace(go.Contour(
                x=x, y=y, z=F,
                colorscale='Viridis',
                showscale=True,
                contours=dict(
                    start=-2, end=2, size=0.2,
                    showlabels=True
                ),
                opacity=0.6,
                name='目标函数等高线'
            ))
        
        # 约束圆
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color='red', width=3),
            name='约束: x² + y² = 1'
        ))
        
        # 最优点
        fig.add_trace(go.Scatter(
            x=[x_opt], y=[y_opt],
            mode='markers',
            marker=dict(size=15, color='yellow', 
                       line=dict(color='black', width=2),
                       symbol='star'),
            name=f'最优解: ({x_opt:.3f}, {y_opt:.3f})'
        ))
        
        # 梯度向量
        if show_gradient:
            # ∇f at optimal point
            grad_f_scale = 0.3
            fig.add_trace(go.Scatter(
                x=[x_opt, x_opt + grad_f_scale],
                y=[y_opt, y_opt + grad_f_scale],
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=8, symbol='arrow', angleref='previous'),
                name='∇f = (1, 1)'
            ))
            
            # ∇g at optimal point (perpendicular to circle)
            grad_g_x = 2 * x_opt
            grad_g_y = 2 * y_opt
            grad_g_scale = 0.3 / np.sqrt(grad_g_x**2 + grad_g_y**2)
            
            fig.add_trace(go.Scatter(
                x=[x_opt, x_opt + grad_g_x * grad_g_scale],
                y=[y_opt, y_opt + grad_g_y * grad_g_scale],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=8, symbol='arrow', angleref='previous'),
                name='∇g = (2x, 2y)'
            ))
        
        fig.update_layout(
            title="圆形约束下的线性目标函数优化",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            xaxis=dict(range=[-1.5, 1.5], constrain='domain', scaleanchor='y'),
            yaxis=dict(range=[-1.5, 1.5], constrain='domain'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示解析解
        st.markdown("### 📊 解析解")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最优 x*", f"{x_opt:.4f}")
        with col2:
            st.metric("最优 y*", f"{y_opt:.4f}")
        with col3:
            st.metric("拉格朗日乘子 λ*", f"{lambda_opt:.4f}")
        
        st.markdown(f"""
        **最优目标函数值**: $f(x^*, y^*) = {x_opt + y_opt:.4f}$
        
        **验证梯度平行条件**:
        - $\\nabla f = (1, 1)$
        - $\\nabla g|_{{(x^*, y^*)}} = (2x^*, 2y^*) = ({2*x_opt:.3f}, {2*y_opt:.3f})$
        - $\\lambda^* \\nabla g = {lambda_opt:.3f} \\times ({2*x_opt:.3f}, {2*y_opt:.3f}) = ({lambda_opt*2*x_opt:.3f}, {lambda_opt*2*y_opt:.3f}) \\approx (1, 1)$ ✓
        """)
    
    @staticmethod
    def _render_ellipse_quadratic():
        """
        问题2: 椭圆约束下的二次目标函数
        
        目标函数: f(x, y) = x² + y² (最小化)
        约束条件: g(x, y) = x²/4 + y²/1 - 1 = 0 (椭圆)
        """
        st.markdown("### 📐 问题2: 在椭圆上最小化 f(x,y) = x² + y²")
        
        st.latex(r"""
        \begin{aligned}
        \text{minimize:} \quad & f(x, y) = x^2 + y^2 \\
        \text{subject to:} \quad & g(x, y) = \frac{x^2}{4} + y^2 - 1 = 0
        \end{aligned}
        """)
        
        with st.sidebar:
            a = st.slider("椭圆长轴 a", 1.0, 5.0, 2.0, 0.1)
            b = st.slider("椭圆短轴 b", 0.5, 3.0, 1.0, 0.1)
        
        # 创建网格
        x = np.linspace(-a*1.5, a*1.5, 300)
        y = np.linspace(-b*1.5, b*1.5, 300)
        X, Y = np.meshgrid(x, y)
        
        # 目标函数
        F = X**2 + Y**2
        
        # 椭圆约束
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse_x = a * np.cos(theta)
        ellipse_y = b * np.sin(theta)
        
        # 解析解: 在椭圆上找离原点最近的点
        # ∇f = (2x, 2y)
        # ∇g = (x/2, 2y) for a=2, b=1
        # 2x = λ(x/2) => λ = 4
        # 2y = λ(2y) => λ = 1
        # 矛盾！所以需要数值求解
        
        # 使用拉格朗日乘子法数值求解
        from scipy.optimize import minimize
        
        def objective(vars):
            x, y = vars
            return x**2 + y**2
        
        def constraint(vars):
            x, y = vars
            return x**2/a**2 + y**2/b**2 - 1
        
        # 初始点
        x0 = [0.5, 0.5]
        
        # 约束条件
        cons = {'type': 'eq', 'fun': constraint}
        
        # 优化
        result = minimize(objective, x0, constraints=cons, method='SLSQP')
        
        x_opt, y_opt = result.x
        f_opt = result.fun
        
        fig = go.Figure()
        
        # 等高线
        fig.add_trace(go.Contour(
            x=x, y=y, z=F,
            colorscale='Reds',
            showscale=True,
            contours=dict(
                start=0, end=10, size=0.5,
                showlabels=True
            ),
            opacity=0.5,
            name='目标函数 x² + y²'
        ))
        
        # 椭圆约束
        fig.add_trace(go.Scatter(
            x=ellipse_x, y=ellipse_y,
            mode='lines',
            line=dict(color='blue', width=3),
            name=f'椭圆: x²/{a}² + y²/{b}² = 1'
        ))
        
        # 最优点
        fig.add_trace(go.Scatter(
            x=[x_opt], y=[y_opt],
            mode='markers',
            marker=dict(size=15, color='yellow',
                       line=dict(color='black', width=2),
                       symbol='star'),
            name=f'最优解: ({x_opt:.3f}, {y_opt:.3f})'
        ))
        
        # 从原点到最优点的线段
        fig.add_trace(go.Scatter(
            x=[0, x_opt], y=[0, y_opt],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='距离原点'
        ))
        
        fig.update_layout(
            title="椭圆约束下的二次目标函数优化",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            xaxis=dict(constrain='domain', scaleanchor='y'),
            yaxis=dict(constrain='domain'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示结果
        st.markdown("### 📊 数值解")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最优 x*", f"{x_opt:.4f}")
        with col2:
            st.metric("最优 y*", f"{y_opt:.4f}")
        with col3:
            st.metric("最小距离²", f"{f_opt:.4f}")
        
        st.info(f"从原点到椭圆的最短距离: {np.sqrt(f_opt):.4f}")
    
    @staticmethod
    def _render_svm_dual():
        """
        SVM对偶问题可视化
        
        原问题: min 1/2||w||² s.t. y_i(w·x_i + b) >= 1
        对偶问题: max Σα_i - 1/2ΣΣα_iα_jy_iy_j(x_i·x_j)
        """
        st.markdown("### 🎯 SVM对偶问题")
        
        st.latex(r"""
        \begin{aligned}
        \text{原问题:} \quad & \min_{w,b} \frac{1}{2}\|w\|^2 \\
        & \text{s.t. } y_i(w \cdot x_i + b) \geq 1, \forall i \\
        \\
        \text{对偶问题:} \quad & \max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_iy_j (x_i \cdot x_j) \\
        & \text{s.t. } \alpha_i \geq 0, \sum_i \alpha_i y_i = 0
        \end{aligned}
        """)
        
        with st.sidebar:
            n_samples = st.slider("样本数量", 10, 50, 20, 5)
            margin = st.slider("类别分离度", 0.5, 3.0, 1.5, 0.1)
        
        # 生成线性可分数据
        np.random.seed(42)
        X_pos = np.random.randn(n_samples//2, 2) + [margin, margin]
        X_neg = np.random.randn(n_samples//2, 2) - [margin, margin]
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
        
        # 使用sklearn求解SVM
        from sklearn.svm import SVC
        
        clf = SVC(kernel='linear', C=1000)  # 大C近似硬间隔
        clf.fit(X, y)
        
        # 获取支持向量
        support_vectors = clf.support_vectors_
        alpha = np.zeros(len(X))
        alpha[clf.support_] = np.abs(clf.dual_coef_[0])
        
        # 绘制决策边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        
        # 决策边界等高线
        fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale='RdBu',
            showscale=False,
            contours=dict(
                start=-2, end=2, size=0.5,
                showlabels=True
            ),
            opacity=0.3,
            name='决策函数'
        ))
        
        # 数据点
        fig.add_trace(go.Scatter(
            x=X[y==1, 0], y=X[y==1, 1],
            mode='markers',
            marker=dict(size=10, color='red', 
                       line=dict(color='black', width=1)),
            name='正类 (y=+1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=X[y==-1, 0], y=X[y==-1, 1],
            mode='markers',
            marker=dict(size=10, color='blue',
                       line=dict(color='black', width=1)),
            name='负类 (y=-1)'
        ))
        
        # 支持向量
        fig.add_trace(go.Scatter(
            x=support_vectors[:, 0], y=support_vectors[:, 1],
            mode='markers',
            marker=dict(size=15, color='yellow',
                       line=dict(color='black', width=2),
                       symbol='circle-open'),
            name='支持向量'
        ))
        
        fig.update_layout(
            title="SVM对偶问题：最大间隔分类器",
            xaxis_title="特征 x₁",
            yaxis_title="特征 x₂",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示对偶变量
        st.markdown("### 📊 对偶变量 α")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 绘制α分布
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Bar(
                x=list(range(len(alpha))),
                y=alpha,
                marker_color=['red' if a > 1e-5 else 'lightgray' for a in alpha],
                name='α_i'
            ))
            fig_alpha.update_layout(
                title="对偶变量分布（红色为支持向量）",
                xaxis_title="样本索引",
                yaxis_title="α_i",
                height=300
            )
            st.plotly_chart(fig_alpha, use_container_width=True)
        
        with col2:
            st.markdown("#### 统计信息")
            st.metric("支持向量数", len(support_vectors))
            st.metric("非零α数量", np.sum(alpha > 1e-5))
            st.metric("||w||²", f"{np.sum(clf.coef_**2):.4f}")
            st.metric("间隔", f"{2/np.sqrt(np.sum(clf.coef_**2)):.4f}")
        
        st.markdown("""
        **对偶性质验证**:
        - ✓ 只有支持向量的 $\\alpha_i > 0$
        - ✓ $\\sum_i \\alpha_i y_i = 0$ (对偶约束)
        - ✓ $w = \\sum_i \\alpha_i y_i x_i$ (权重由支持向量表示)
        """)
    
    @staticmethod
    def _render_kkt():
        """KKT条件演示"""
        st.markdown("### 📐 KKT条件 (Karush-Kuhn-Tucker)")
        
        st.latex(r"""
        \begin{aligned}
        \text{原问题:} \quad & \min f(x) \\
        & \text{s.t. } g_i(x) \leq 0, \quad h_j(x) = 0 \\
        \\
        \text{KKT条件:} \quad & \nabla f(x^*) + \sum_i \mu_i \nabla g_i(x^*) + \sum_j \lambda_j \nabla h_j(x^*) = 0 \\
        & g_i(x^*) \leq 0, \quad h_j(x^*) = 0 \\
        & \mu_i \geq 0, \quad \mu_i g_i(x^*) = 0 \quad \text{(互补松弛)}
        \end{aligned}
        """)
        
        st.markdown("""
        #### 🔑 互补松弛条件 (Complementary Slackness)
        
        $\\mu_i g_i(x^*) = 0$ 意味着:
        - 如果约束不活跃 ($g_i(x^*) < 0$), 则 $\\mu_i = 0$
        - 如果约束活跃 ($g_i(x^*) = 0$), 则 $\\mu_i \\geq 0$
        
        **在SVM中的体现**:
        - 非支持向量: $y_i(w \\cdot x_i + b) > 1 \\Rightarrow \\alpha_i = 0$
        - 支持向量: $y_i(w \\cdot x_i + b) = 1 \\Rightarrow \\alpha_i > 0$
        """)
        
        # 示例：带不等式约束的优化
        st.markdown("#### 📊 示例: 最小化 $f(x,y) = x^2 + y^2$ 在 $x + y \\geq 1$ 约束下")
        
        with st.sidebar:
            constraint_value = st.slider("约束值 c", 0.5, 3.0, 1.0, 0.1)
        
        # 目标函数
        x = np.linspace(-1, 3, 300)
        y = np.linspace(-1, 3, 300)
        X_grid, Y_grid = np.meshgrid(x, y)
        F = X_grid**2 + Y_grid**2
        
        # 约束线 x + y = c
        x_line = np.linspace(-0.5, constraint_value+0.5, 100)
        y_line = constraint_value - x_line
        
        # 解析解: x* = y* = c/2 (在约束边界上)
        x_opt = constraint_value / 2
        y_opt = constraint_value / 2
        
        fig = go.Figure()
        
        # 等高线
        fig.add_trace(go.Contour(
            x=x, y=y, z=F,
            colorscale='Viridis',
            showscale=True,
            contours=dict(showlabels=True),
            opacity=0.5
        ))
        
        # 约束线
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'约束: x + y = {constraint_value}'
        ))
        
        # 可行域填充
        fig.add_trace(go.Scatter(
            x=[constraint_value, 3, 3, constraint_value],
            y=[0, 0, 3, constraint_value],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 最优点
        fig.add_trace(go.Scatter(
            x=[x_opt], y=[y_opt],
            mode='markers',
            marker=dict(size=15, color='yellow',
                       line=dict(color='black', width=2),
                       symbol='star'),
            name=f'最优解: ({x_opt:.2f}, {y_opt:.2f})'
        ))
        
        fig.update_layout(
            title=f"不等式约束优化 (约束值 c = {constraint_value})",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            xaxis=dict(range=[-0.5, 3]),
            yaxis=dict(range=[-0.5, 3], scaleanchor='x'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # KKT条件验证
        st.markdown("### ✅ KKT条件验证")
        
        # 约束值
        g_value = constraint_value - (x_opt + y_opt)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**原始条件**:")
            st.write(f"- 约束: $x + y \\geq {constraint_value}$")
            st.write(f"- 约束值: $g(x^*, y^*) = {g_value:.6f}$ (应该=0)")
            
        with col2:
            st.markdown("**KKT乘子**:")
            mu = 2 * x_opt  # ∇f = (2x, 2y), ∇g = (1, 1)
            st.write(f"- $\\mu = {mu:.4f}$ (应该>0)")
            st.write(f"- 互补松弛: $\\mu \\cdot g = {mu * g_value:.6f}$ (应该=0)")
