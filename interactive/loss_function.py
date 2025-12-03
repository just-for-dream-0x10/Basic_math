"""
损失函数交互式可视化
展示不同损失函数的原理、特性和应用场景
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render, safe_compute
from common.quiz_system import QuizSystem, QuizTemplates
from common.performance import cache_data, PerformanceMonitor


class InteractiveLossFunction:
    """交互式损失函数可视化"""
    
    @staticmethod
    @safe_render
    def render():
        st.title("📉 损失函数：优化的指南针")
        
        # 添加标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 损失函数对比",
            "🎯 回归损失",
            "🔢 分类损失", 
            "🌋 损失地形",
            "💡 鲁棒性分析"
        ])
        
        with tab1:
            InteractiveLossFunction._render_loss_comparison()
        
        with tab2:
            InteractiveLossFunction._render_regression_losses()
        
        with tab3:
            InteractiveLossFunction._render_classification_losses()
        
        with tab4:
            InteractiveLossFunction._render_loss_landscape()
        
        with tab5:
            InteractiveLossFunction._render_robustness()
    

        # 添加交互式测验
        quiz_system = QuizSystem("loss_function")
        quizzes = QuizTemplates.get_loss_function_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_loss_comparison():
        """损失函数全局对比"""
        st.subheader("📊 常见损失函数对比")
        
        st.markdown("""
        **损失函数的作用**:
        - 量化模型预测与真实值的差距
        - 为优化器提供梯度方向
        - 不同任务需要不同的损失函数
        
        **核心分类**:
        1. **回归损失**: MSE, MAE, Huber
        2. **分类损失**: CrossEntropy, Hinge, Focal
        3. **排序损失**: Ranking Loss, Triplet Loss
        """)
        
        # 创建对比表
        loss_table = {
            "损失函数": ["MSE (L2)", "MAE (L1)", "Huber", "Cross Entropy", "Hinge (SVM)", "Focal Loss"],
            "应用场景": ["回归", "回归", "回归", "分类", "分类", "分类"],
            "对异常值": ["敏感", "鲁棒", "鲁棒", "中等", "鲁棒", "鲁棒"],
            "梯度特性": ["线性增长", "常数", "分段", "指数", "分段", "自适应"],
            "优点": [
                "数学简单，凸函数",
                "对异常值鲁棒",
                "平衡MSE和MAE",
                "概率解释清晰",
                "最大间隔",
                "处理类别不平衡"
            ]
        }
        
        import pandas as pd
        df = pd.DataFrame(loss_table)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def _render_regression_losses():
        """回归损失函数详解"""
        st.subheader("🎯 回归损失函数")
        
        st.markdown("""
        **回归任务**: 预测连续值 $\\hat{y} = f(x)$，真实值为 $y$
        
        **三种经典损失**:
        
        1. **均方误差 (MSE)**: $L = \\frac{1}{2}(y - \\hat{y})^2$
           - 梯度: $\\frac{\\partial L}{\\partial \\hat{y}} = -(y - \\hat{y})$
           - 特点: 对大误差惩罚重（平方关系）
        
        2. **平均绝对误差 (MAE)**: $L = |y - \\hat{y}|$
           - 梯度: $\\frac{\\partial L}{\\partial \\hat{y}} = -\\text{sign}(y - \\hat{y})$
           - 特点: 对所有误差惩罚相同（鲁棒）
        
        3. **Huber损失**:
        """)
        
        st.latex(r"""
        L_{\delta}(y, \hat{y}) = \begin{cases}
        \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
        \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{if } |y - \hat{y}| > \delta
        \end{cases}
        """)
        
        st.markdown("""
           - 特点: 小误差用MSE，大误差用MAE
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 参数设置")
            delta = st.slider("Huber δ参数", 0.5, 3.0, 1.0, 0.1)
            show_gradient = st.checkbox("显示梯度", value=True)
        
        # 生成数据
        errors = np.linspace(-5, 5, 200)
        
        # 计算三种损失
        mse_loss = 0.5 * errors**2
        mae_loss = np.abs(errors)
        
        huber_loss = np.where(
            np.abs(errors) <= delta,
            0.5 * errors**2,
            delta * (np.abs(errors) - 0.5 * delta)
        )
        
        with col2:
            # 创建子图
            if show_gradient:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("损失函数", "梯度")
                )
                
                # 损失函数
                fig.add_trace(go.Scatter(x=errors, y=mse_loss, name='MSE (L2)', 
                                        line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=errors, y=mae_loss, name='MAE (L1)', 
                                        line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=errors, y=huber_loss, name=f'Huber (δ={delta})', 
                                        line=dict(color='green')), row=1, col=1)
                
                # 梯度
                mse_grad = -errors
                mae_grad = -np.sign(errors)
                huber_grad = np.where(
                    np.abs(errors) <= delta,
                    -errors,
                    -delta * np.sign(errors)
                )
                
                fig.add_trace(go.Scatter(x=errors, y=mse_grad, name='MSE梯度', 
                                        line=dict(color='blue', dash='dash'), 
                                        showlegend=False), row=1, col=2)
                fig.add_trace(go.Scatter(x=errors, y=mae_grad, name='MAE梯度', 
                                        line=dict(color='red', dash='dash'), 
                                        showlegend=False), row=1, col=2)
                fig.add_trace(go.Scatter(x=errors, y=huber_grad, name='Huber梯度', 
                                        line=dict(color='green', dash='dash'), 
                                        showlegend=False), row=1, col=2)
                
                fig.update_xaxes(title_text="误差 (y - ŷ)", row=1, col=1)
                fig.update_xaxes(title_text="误差 (y - ŷ)", row=1, col=2)
                fig.update_yaxes(title_text="损失值", row=1, col=1)
                fig.update_yaxes(title_text="梯度值", row=1, col=2)
                
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=errors, y=mse_loss, name='MSE (L2)', 
                                        line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=errors, y=mae_loss, name='MAE (L1)', 
                                        line=dict(color='red', width=2)))
                fig.add_trace(go.Scatter(x=errors, y=huber_loss, name=f'Huber (δ={delta})', 
                                        line=dict(color='green', width=2)))
                
                fig.update_xaxes(title_text="误差 (y - ŷ)")
                fig.update_yaxes(title_text="损失值")
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **关键观察**:
        - 📈 **MSE**: 误差越大，惩罚呈平方增长 → 对异常值敏感
        - 📏 **MAE**: 线性惩罚 → 对异常值鲁棒，但梯度不连续
        - ⚖️ **Huber**: 最佳平衡 → 小误差用MSE（快速收敛），大误差用MAE（鲁棒性）
        """)
    
    @staticmethod
    def _render_classification_losses():
        """分类损失函数详解"""
        st.subheader("🔢 分类损失函数")
        
        st.markdown("""
        **二分类问题**: 预测 $\\hat{y} \\in [0,1]$，真实标签 $y \\in \\{0, 1\\}$
        
        **交叉熵损失 (Cross Entropy)**:
        $$L = -[y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y})]$$
        
        **与信息论的联系**:
        - 交叉熵 = 负对数似然
        - 度量真实分布 $p$ 和预测分布 $q$ 的差异
        - $H(p,q) = -\\sum p(x)\\log q(x)$
        
        **Logits vs Probabilities**:
        - Logits: $z \\in \\mathbb{R}$ (未归一化)
        - Sigmoid: $\\sigma(z) = \\frac{1}{1+e^{-z}} \\in (0,1)$
        - Softmax (多分类): $\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 可视化选项")
            vis_type = st.radio(
                "选择视角",
                ["交叉熵曲线", "Sigmoid函数", "Logits空间"]
            )
        
        with col2:
            if vis_type == "交叉熵曲线":
                # 交叉熵损失
                y_pred = np.linspace(0.001, 0.999, 200)
                
                fig = go.Figure()
                
                # y=1时的损失
                loss_y1 = -np.log(y_pred)
                fig.add_trace(go.Scatter(
                    x=y_pred, y=loss_y1,
                    name='y=1 (正类)',
                    line=dict(color='blue', width=2)
                ))
                
                # y=0时的损失
                loss_y0 = -np.log(1 - y_pred)
                fig.add_trace(go.Scatter(
                    x=y_pred, y=loss_y0,
                    name='y=0 (负类)',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title="二元交叉熵损失",
                    xaxis_title="预测概率 ŷ",
                    yaxis_title="损失值",
                    yaxis_type="log",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **观察**: 
                - 当真实标签y=1时，预测ŷ→0会导致损失→∞
                - 当真实标签y=0时，预测ŷ→1会导致损失→∞
                - 预测越自信且正确，损失越小
                """)
            
            elif vis_type == "Sigmoid函数":
                # Sigmoid函数
                z = np.linspace(-10, 10, 200)
                sigmoid = 1 / (1 + np.exp(-z))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=z, y=sigmoid,
                    name='σ(z)',
                    line=dict(color='purple', width=3)
                ))
                
                # 添加参考线
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                             annotation_text="决策边界")
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="Sigmoid激活函数",
                    xaxis_title="Logit z",
                    yaxis_title="概率 σ(z)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Sigmoid性质**:
                - $\\sigma(0) = 0.5$ (决策边界)
                - $\\sigma(z) + \\sigma(-z) = 1$ (对称性)
                - 梯度: $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$
                - 梯度消失: $|z|$很大时梯度→0
                """)
            
            else:  # Logits空间
                # 创建2D logits网格
                z1 = np.linspace(-3, 3, 50)
                z2 = np.linspace(-3, 3, 50)
                Z1, Z2 = np.meshgrid(z1, z2)
                
                # Softmax
                exp_z1 = np.exp(Z1)
                exp_z2 = np.exp(Z2)
                prob_class1 = exp_z1 / (exp_z1 + exp_z2)
                
                fig = go.Figure(data=go.Contour(
                    x=z1, y=z2, z=prob_class1,
                    colorscale='RdBu',
                    contours=dict(
                        start=0, end=1, size=0.1,
                        showlabels=True
                    ),
                    colorbar=dict(title="P(类别1)")
                ))
                
                # 添加决策边界
                fig.add_trace(go.Scatter(
                    x=[-3, 3], y=[-3, 3],
                    mode='lines',
                    line=dict(color='yellow', width=3, dash='dash'),
                    name='决策边界 z₁=z₂'
                ))
                
                fig.update_layout(
                    title="Logits空间的Softmax概率",
                    xaxis_title="Logit z₁ (类别1)",
                    yaxis_title="Logit z₂ (类别2)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("决策边界在 z₁=z₂ 处，此时两类概率均为0.5")
    
    @staticmethod
    def _render_loss_landscape():
        """损失地形可视化"""
        st.subheader("🌋 损失地形 (Loss Landscape)")
        
        st.markdown("""
        **损失地形**: 参数空间中的损失函数值分布
        
        理解损失地形对优化至关重要：
        - **凸函数**: 单一全局最小值，容易优化
        - **非凸函数**: 多个局部最小值、鞍点，难优化
        - **平坦vs陡峭**: 影响学习率选择和收敛速度
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 选择地形")
            landscape_type = st.selectbox(
                "地形类型",
                ["简单凸函数", "多峰函数", "Rosenbrock峡谷", "鞍点函数"]
            )
            
            show_path = st.checkbox("显示优化路径", value=True)
        
        # 生成网格
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        if landscape_type == "简单凸函数":
            Z = X**2 + Y**2
            title = "碗状地形 (凸函数)"
            optimal = (0, 0)
        elif landscape_type == "多峰函数":
            Z = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)
            title = "多峰地形 (多个局部最小值)"
            optimal = (0, 0)
        elif landscape_type == "Rosenbrock峡谷":
            a, b = 1, 10
            Z = (a - X)**2 + b * (Y - X**2)**2
            title = "Rosenbrock峡谷 (细长峡谷)"
            optimal = (1, 1)
        else:  # 鞍点
            Z = X**2 - Y**2
            title = "鞍点地形"
            optimal = (0, 0)
        
        with col2:
            fig = go.Figure()
            
            # 添加等高线
            fig.add_trace(go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                contours=dict(
                    coloring='heatmap',
                    showlabels=True
                ),
                colorbar=dict(title="损失值")
            ))
            
            # 标记最优点
            fig.add_trace(go.Scatter(
                x=[optimal[0]], y=[optimal[1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='最优点'
            ))
            
            # 添加优化路径示例
            if show_path and landscape_type == "简单凸函数":
                # 模拟梯度下降路径
                path_x = [2.5]
                path_y = [2.5]
                lr = 0.1
                for _ in range(20):
                    grad_x = 2 * path_x[-1]
                    grad_y = 2 * path_y[-1]
                    path_x.append(path_x[-1] - lr * grad_x)
                    path_y.append(path_y[-1] - lr * grad_y)
                
                fig.add_trace(go.Scatter(
                    x=path_x, y=path_y,
                    mode='lines+markers',
                    line=dict(color='white', width=2),
                    marker=dict(size=5, color='white'),
                    name='GD路径'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="参数 w₁",
                yaxis_title="参数 w₂",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **地形特征解读**:
        - **简单凸函数**: 最理想，任意点梯度下降都能找到全局最优
        - **多峰函数**: 容易陷入局部最优，需要全局优化方法
        - **Rosenbrock峡谷**: 峡谷导致"之字形"路径，需要动量
        - **鞍点**: 一阶导数为0但不是极值，二阶方法可判断
        """)
    
    @staticmethod
    def _render_robustness():
        """损失函数的鲁棒性分析"""
        st.subheader("💡 鲁棒性分析：异常值的影响")
        
        st.markdown("""
        **鲁棒性**: 损失函数对异常值/噪声数据的敏感程度
        
        **实验设置**: 
        - 生成正常数据点和异常值
        - 对比不同损失函数的表现
        - 观察损失值和梯度的差异
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 数据设置")
            n_normal = st.slider("正常点数量", 10, 100, 50, 10)
            n_outliers = st.slider("异常值数量", 0, 20, 5, 1)
            outlier_scale = st.slider("异常值偏离程度", 1.0, 5.0, 3.0, 0.5)
        
        # 生成数据
        np.random.seed(42)
        
        # 正常数据
        X_normal = np.linspace(0, 10, n_normal)
        y_normal = 2 * X_normal + 1 + np.random.randn(n_normal) * 0.5
        
        # 异常值
        X_outliers = np.random.uniform(0, 10, n_outliers)
        y_outliers = 2 * X_outliers + 1 + np.random.randn(n_outliers) * outlier_scale * 3
        
        # 合并数据
        X_all = np.concatenate([X_normal, X_outliers])
        y_all = np.concatenate([y_normal, y_outliers])
        
        # 拟合三种损失函数
        # 简单线性回归 w*x + b
        def fit_model(X, y, loss_type='mse'):
            # 使用梯度下降
            w, b = 1.0, 0.0
            lr = 0.01
            
            for _ in range(1000):
                y_pred = w * X + b
                errors = y - y_pred
                
                if loss_type == 'mse':
                    grad_w = -np.mean(errors * X)
                    grad_b = -np.mean(errors)
                elif loss_type == 'mae':
                    grad_w = -np.mean(np.sign(errors) * X)
                    grad_b = -np.mean(np.sign(errors))
                else:  # huber
                    delta = 1.0
                    grad_w = -np.mean(np.where(
                        np.abs(errors) <= delta,
                        errors * X,
                        delta * np.sign(errors) * X
                    ))
                    grad_b = -np.mean(np.where(
                        np.abs(errors) <= delta,
                        errors,
                        delta * np.sign(errors)
                    ))
                
                w -= lr * grad_w
                b -= lr * grad_b
            
            return w, b
        
        # 拟合三个模型
        w_mse, b_mse = fit_model(X_all, y_all, 'mse')
        w_mae, b_mae = fit_model(X_all, y_all, 'mae')
        w_huber, b_huber = fit_model(X_all, y_all, 'huber')
        
        with col2:
            fig = go.Figure()
            
            # 绘制数据点
            fig.add_trace(go.Scatter(
                x=X_normal, y=y_normal,
                mode='markers',
                marker=dict(size=8, color='lightblue'),
                name='正常数据'
            ))
            
            if n_outliers > 0:
                fig.add_trace(go.Scatter(
                    x=X_outliers, y=y_outliers,
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='x'),
                    name='异常值'
                ))
            
            # 绘制拟合直线
            X_line = np.array([0, 10])
            
            fig.add_trace(go.Scatter(
                x=X_line, y=w_mse * X_line + b_mse,
                mode='lines',
                line=dict(color='blue', width=2),
                name=f'MSE (w={w_mse:.2f})'
            ))
            
            fig.add_trace(go.Scatter(
                x=X_line, y=w_mae * X_line + b_mae,
                mode='lines',
                line=dict(color='red', width=2),
                name=f'MAE (w={w_mae:.2f})'
            ))
            
            fig.add_trace(go.Scatter(
                x=X_line, y=w_huber * X_line + b_huber,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name=f'Huber (w={w_huber:.2f})'
            ))
            
            # 真实直线
            fig.add_trace(go.Scatter(
                x=X_line, y=2 * X_line + 1,
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                name='真实关系 (w=2.0)'
            ))
            
            fig.update_layout(
                title="不同损失函数的鲁棒性对比",
                xaxis_title="X",
                yaxis_title="y",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 显示拟合结果
        st.markdown("### 📊 拟合结果分析")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("真实斜率", "2.00")
        with col_b:
            error_mse = abs(w_mse - 2.0)
            st.metric("MSE斜率", f"{w_mse:.2f}", f"{error_mse:.2f}")
        with col_c:
            error_mae = abs(w_mae - 2.0)
            st.metric("MAE斜率", f"{w_mae:.2f}", f"{error_mae:.2f}")
        with col_d:
            error_huber = abs(w_huber - 2.0)
            st.metric("Huber斜率", f"{w_huber:.2f}", f"{error_huber:.2f}")
        
        st.markdown("""
        **结论**:
        - 🔵 **MSE**: 对异常值敏感，拟合直线被异常值拉偏
        - 🔴 **MAE**: 对异常值鲁棒，更接近真实关系
        - 🟢 **Huber**: 平衡两者，既快速收敛又保持鲁棒性
        
        **选择建议**:
        - 数据干净 → MSE (收敛快)
        - 有异常值 → MAE或Huber (鲁棒)
        - 兼顾两者 → Huber (推荐)
        """)
        
        # 添加交互式测验
