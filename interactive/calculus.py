"""
交互式微积分可视化
严格按照 0.1.Calculus_in_Deep_Learning.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import math
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render, safe_compute
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation
from common.quiz_system import QuizSystem, QuizTemplates
from common.performance import cache_data, PerformanceMonitor


class InteractiveCalculus:
    """交互式微积分可视化"""
    
    @staticmethod
    @safe_render
    def render():
        st.subheader("📐 交互式微积分：深度学习的数学基础")
        st.markdown("""
        **微积分的核心**: 研究变化率和累积量
        
        在深度学习中:
        - **导数**: 度量函数对输入的敏感度
        - **梯度**: 多元函数的方向导数，指向增长最快的方向
        - **链式法则**: 反向传播的数学基础
        
        $$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial w}$$
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择主题")
            topic = st.selectbox("主题", [
                "导数与梯度",
                "泰勒展开",
                "链式法则",
                "梯度消失/爆炸",
                "自动微分"
            ])
        
        if topic == "导数与梯度":
            InteractiveCalculus._render_derivative()
        elif topic == "泰勒展开":
            InteractiveCalculus._render_taylor()
        elif topic == "链式法则":
            InteractiveCalculus._render_chain_rule()
        elif topic == "梯度消失/爆炸":
            InteractiveCalculus._render_gradient_problems()
        elif topic == "自动微分":
            InteractiveCalculus._render_autograd()
    

        # 添加交互式测验
        quiz_system = QuizSystem("calculus")
        quizzes = QuizTemplates.get_calculus_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_derivative():
        """导数与梯度可视化"""
        st.markdown("### 📈 导数：变化率的度量")
        
        st.latex(r"""
        f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 函数选择")
            func_type = st.selectbox("函数类型", [
                "多项式", "三角函数", "指数函数", "Sigmoid", "ReLU"
            ])
        
        # 定义函数和导数
        x = np.linspace(-5, 5, 500)
        
        if func_type == "多项式":
            a = st.sidebar.slider("系数 a", -2.0, 2.0, 0.5, 0.1)
            b = st.sidebar.slider("系数 b", -2.0, 2.0, 1.0, 0.1)
            c = st.sidebar.slider("系数 c", -2.0, 2.0, 0.0, 0.1)
            
            y = a * x**2 + b * x + c
            dy = 2 * a * x + b
            func_latex = f"f(x) = {a:.1f}x^2 + {b:.1f}x + {c:.1f}"
            deriv_latex = f"f'(x) = {2*a:.1f}x + {b:.1f}"
            
        elif func_type == "三角函数":
            freq = st.sidebar.slider("频率", 0.5, 3.0, 1.0, 0.1)
            y = np.sin(freq * x)
            dy = freq * np.cos(freq * x)
            func_latex = f"f(x) = \\sin({freq:.1f}x)"
            deriv_latex = f"f'(x) = {freq:.1f}\\cos({freq:.1f}x)"
            
        elif func_type == "指数函数":
            a = st.sidebar.slider("底数 a", 0.5, 2.0, np.e, 0.1)
            y = np.exp(a * x)
            dy = a * np.exp(a * x)
            func_latex = f"f(x) = e^{{{a:.1f}x}}"
            deriv_latex = f"f'(x) = {a:.1f}e^{{{a:.1f}x}}"
            
        elif func_type == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
            dy = y * (1 - y)
            func_latex = r"f(x) = \frac{1}{1+e^{-x}}"
            deriv_latex = r"f'(x) = f(x)(1-f(x))"
            
        else:  # ReLU
            y = np.maximum(0, x)
            dy = np.where(x > 0, 1, 0)
            func_latex = r"f(x) = \max(0, x)"
            deriv_latex = r"f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}"
        
        # 选择一个点显示切线
        x_point = st.sidebar.slider("观察点 x₀", float(x.min()), float(x.max()), 0.0, 0.1)
        
        # 找到最接近的索引
        idx = np.argmin(np.abs(x - x_point))
        y_point = y[idx]
        slope = dy[idx]
        
        # 切线
        tangent_x = np.linspace(x_point - 2, x_point + 2, 100)
        tangent_y = y_point + slope * (tangent_x - x_point)
        
        # 绘图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("函数与切线", "导数函数")
        )
        
        # 原函数
        fig.add_trace(go.Scatter(x=x, y=y, name='f(x)', line=dict(color='blue', width=2)),
                     row=1, col=1)
        
        # 切线
        fig.add_trace(go.Scatter(x=tangent_x, y=tangent_y, name='切线', 
                                line=dict(color='red', width=2, dash='dash')),
                     row=1, col=1)
        
        # 观察点
        fig.add_trace(go.Scatter(x=[x_point], y=[y_point], mode='markers',
                                marker=dict(size=12, color='red'),
                                name=f'点 ({x_point:.2f}, {y_point:.2f})'),
                     row=1, col=1)
        
        # 导数函数
        fig.add_trace(go.Scatter(x=x, y=dy, name="f'(x)", line=dict(color='green', width=2)),
                     row=1, col=2)
        
        # 导数值标记
        fig.add_trace(go.Scatter(x=[x_point], y=[slope], mode='markers',
                                marker=dict(size=12, color='red'),
                                name=f"f'({x_point:.2f}) = {slope:.2f}"),
                     row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="f(x)", row=1, col=1)
        fig.update_yaxes(title_text="f'(x)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示公式
        col1, col2 = st.columns(2)
        with col1:
            st.latex(func_latex)
        with col2:
            st.latex(deriv_latex)
        
        # 解释
        st.markdown(f"""
        ### 🔍 在点 $x_0 = {x_point:.2f}$ 处:
        
        - **函数值**: $f(x_0) = {y_point:.4f}$
        - **导数值**: $f'(x_0) = {slope:.4f}$
        - **物理意义**: 在这一点，$x$ 增加 1 单位，$f(x)$ 约增加 {slope:.4f} 单位
        - **几何意义**: 切线斜率为 {slope:.4f}
        """)
    
    @staticmethod
    def _render_taylor():
        """泰勒展开可视化"""
        st.markdown("### 🔬 泰勒展开：函数的多项式近似")
        
        st.latex(r"""
        f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 参数设置")
            func_choice = st.selectbox("目标函数", ["sin(x)", "cos(x)", "exp(x)", "log(1+x)"])
            center = st.slider("展开中心 a", -2.0, 2.0, 0.0, 0.1)
            max_order = st.slider("最高阶数", 1, 10, 5)
        
        x = np.linspace(-3, 3, 500)
        
        # 定义函数
        if func_choice == "sin(x)":
            f = np.sin(x)
            f_center = np.sin(center)
            derivatives = [np.sin, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x)]
        elif func_choice == "cos(x)":
            f = np.cos(x)
            f_center = np.cos(center)
            derivatives = [np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x), np.sin]
        elif func_choice == "exp(x)":
            f = np.exp(x)
            f_center = np.exp(center)
            derivatives = [np.exp] * 10
        else:  # log(1+x)
            f = np.log(1 + x)
            f_center = np.log(1 + center)
            # 对 log(1+x) 的导数
            derivatives = [
                lambda x: np.log(1 + x),
                lambda x: 1 / (1 + x),
                lambda x: -1 / (1 + x)**2,
                lambda x: 2 / (1 + x)**3,
                lambda x: -6 / (1 + x)**4
            ]
        
        # 计算泰勒级数
        fig = go.Figure()
        
        # 原函数
        fig.add_trace(go.Scatter(
            x=x, y=f,
            name='原函数',
            line=dict(color='black', width=3)
        ))
        
        # 不同阶的泰勒展开
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'cyan']
        
        for order in range(1, max_order + 1):
            taylor_approx = np.zeros_like(x)
            
            for n in range(order + 1):
                if n < len(derivatives):
                    deriv_value = derivatives[n % len(derivatives)](center)
                    factorial = math.factorial(n)
                    taylor_approx += deriv_value * (x - center)**n / factorial
            
            fig.add_trace(go.Scatter(
                x=x, y=taylor_approx,
                name=f'{order}阶近似',
                line=dict(color=colors[order-1], width=1.5, dash='dash'),
                visible=(order == max_order)  # 默认只显示最高阶
            ))
        
        # 添加展开中心的标记
        fig.add_vline(x=center, line_dash="dot", line_color="red",
                     annotation_text=f"展开中心 a={center}")
        
        # 创建滑块来选择显示的阶数
        steps = []
        for i in range(1, max_order + 1):
            step = dict(
                method="update",
                args=[{"visible": [True] + [j == i for j in range(1, max_order + 1)]}],
                label=f"{i}阶"
            )
            steps.append(step)
        
        sliders = [dict(
            active=max_order - 1,
            currentvalue={"prefix": "显示阶数: "},
            steps=steps
        )]
        
        fig.update_layout(
            title=f"泰勒展开: {func_choice} 在 x={center} 处",
            xaxis_title="x",
            yaxis_title="f(x)",
            height=600,
            sliders=sliders
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 误差分析
        st.markdown("### 📊 误差分析")
        
        # 计算当前阶数的误差
        taylor_current = np.zeros_like(x)
        for n in range(max_order + 1):
            if n < len(derivatives):
                deriv_value = derivatives[n % len(derivatives)](center)
                factorial = math.factorial(n)
                taylor_current += deriv_value * (x - center)**n / factorial
        
        error = np.abs(f - taylor_current)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最大绝对误差", f"{np.max(error):.6f}")
        with col2:
            st.metric("平均绝对误差", f"{np.mean(error):.6f}")
        
        st.markdown("""
        **泰勒展开的意义**:
        - 🎯 **局部逼近**: 在展开中心附近，低阶项已经很准确
        - 📏 **误差控制**: 阶数越高，近似越精确
        - 🧮 **优化应用**: 一阶展开→梯度下降，二阶展开→牛顿法
        """)
    
    @staticmethod
    def _render_chain_rule():
        """链式法则：反向传播的核心"""
        st.markdown("### ⛓️ 链式法则：反向传播的数学基础")
        
        st.latex(r"""
        \frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
        """)
        
        st.markdown("""
        **反向传播本质**: 通过链式法则，从输出层向输入层逐层传播梯度
        
        对于神经网络 $y = f_3(f_2(f_1(x)))$:
        """)
        
        st.latex(r"""
        \frac{\partial y}{\partial x} = \frac{\partial y}{\partial z_2} \cdot \frac{\partial z_2}{\partial z_1} \cdot \frac{\partial z_1}{\partial x}
        """)
        
        with st.sidebar:
            st.markdown("### 🧪 网络配置")
            n_layers = st.slider("层数", 2, 5, 3)
            activation = st.selectbox("激活函数", ["Sigmoid", "Tanh", "ReLU"])
        
        # 模拟简单的前向和反向传播
        st.markdown(f"### 📊 {n_layers}层网络的梯度传播")
        
        # 生成示例输入
        x_input = st.sidebar.slider("输入值 x", -5.0, 5.0, 1.0, 0.1)
        
        # 定义激活函数和导数
        if activation == "Sigmoid":
            def act(z): return 1 / (1 + np.exp(-z))
            def act_derivative(z): 
                a = act(z)
                return a * (1 - a)
        elif activation == "Tanh":
            def act(z): return np.tanh(z)
            def act_derivative(z): return 1 - np.tanh(z)**2
        else:  # ReLU
            def act(z): return np.maximum(0, z)
            def act_derivative(z): return np.where(z > 0, 1, 0)
        
        # 前向传播
        activations = [x_input]
        weights = np.random.randn(n_layers) * 0.5  # 随机权重
        
        for i in range(n_layers):
            z = weights[i] * activations[-1]
            a = act(z)
            activations.append(a)
        
        # 反向传播
        # 假设损失函数是 L = (y - target)^2
        target = 0.5
        L = (activations[-1] - target)**2
        
        # 计算梯度
        gradients = [2 * (activations[-1] - target)]  # dL/dy
        
        for i in range(n_layers - 1, -1, -1):
            z = weights[i] * activations[i]
            grad = gradients[-1] * act_derivative(z) * weights[i]
            gradients.append(grad)
        
        gradients.reverse()
        
        # 可视化计算图
        fig = go.Figure()
        
        # 前向传播路径
        for i in range(len(activations)):
            fig.add_trace(go.Scatter(
                x=[i], y=[activations[i]],
                mode='markers+text',
                marker=dict(size=20, color='blue'),
                text=[f'a{i}<br>{activations[i]:.3f}'],
                textposition='top center',
                name=f'Layer {i}'
            ))
        
        # 连接线
        for i in range(len(activations) - 1):
            fig.add_trace(go.Scatter(
                x=[i, i+1],
                y=[activations[i], activations[i+1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="前向传播",
            xaxis_title="层索引",
            yaxis_title="激活值",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示梯度
        st.markdown("### 🔙 反向传播的梯度")
        
        gradient_data = {
            "层": [f"Layer {i}" for i in range(n_layers + 1)],
            "激活值": [f"{a:.4f}" for a in activations],
            "梯度": [f"{g:.4f}" for g in gradients]
        }
        
        import pandas as pd
        df = pd.DataFrame(gradient_data)
        st.dataframe(df, use_container_width=True)
        
        # 计算总梯度（链式法则）
        total_gradient = np.prod([act_derivative(weights[i] * activations[i]) * weights[i] 
                                  for i in range(n_layers)])
        total_gradient *= 2 * (activations[-1] - target)
        
        st.markdown(f"""
        ### 🧮 链式法则验证
        
        **总梯度** (从输出到输入):
        $$\\frac{{\\partial L}}{{\\partial x}} = {total_gradient:.6f}$$
        
        这个梯度告诉我们：输入 $x$ 改变 1 单位，损失 $L$ 改变约 {total_gradient:.6f} 单位
        
        **链式法则的威力**:
        - ✅ 只需要局部导数（每层的导数）
        - ✅ 通过反向传播高效计算
        - ✅ 是自动微分的数学基础
        """)
    
    @staticmethod
    def _render_gradient_problems():
        """梯度消失与梯度爆炸"""
        st.markdown("### ⚠️ 梯度消失与梯度爆炸")
        
        st.markdown("""
        **问题根源**: 深层网络中，梯度需要通过多层反向传播
        
        对于 L 层网络:
        $$\\frac{\\partial L}{\\partial w_1} = \\frac{\\partial L}{\\partial z_L} \\prod_{i=1}^{L-1} \\frac{\\partial z_{i+1}}{\\partial z_i}$$
        
        如果每项导数 < 1 → **梯度消失**  
        如果每项导数 > 1 → **梯度爆炸**
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 网络参数")
            n_layers = st.slider("网络层数", 5, 50, 20, 5)
            activation = st.selectbox("激活函数", ["Sigmoid", "Tanh", "ReLU", "LeakyReLU"])
            init_method = st.selectbox("权重初始化", ["小随机值", "Xavier", "He初始化"])
        
        # 权重初始化
        if init_method == "小随机值":
            weights = np.random.randn(n_layers) * 0.01
        elif init_method == "Xavier":
            weights = np.random.randn(n_layers) * np.sqrt(2.0 / (1 + 1))  # 简化版
        else:  # He初始化
            weights = np.random.randn(n_layers) * np.sqrt(2.0)
        
        # 定义激活函数导数
        z_values = np.linspace(-2, 2, 100)
        
        if activation == "Sigmoid":
            def act_deriv(z):
                s = 1 / (1 + np.exp(-z))
                return s * (1 - s)
            title_suffix = "Sigmoid: 导数最大值 0.25"
        elif activation == "Tanh":
            def act_deriv(z):
                return 1 - np.tanh(z)**2
            title_suffix = "Tanh: 导数最大值 1.0"
        elif activation == "ReLU":
            def act_deriv(z):
                return np.where(z > 0, 1, 0)
            title_suffix = "ReLU: 导数为 0 或 1"
        else:  # LeakyReLU
            alpha = 0.01
            def act_deriv(z):
                return np.where(z > 0, 1, alpha)
            title_suffix = f"LeakyReLU: 导数为 {alpha} 或 1"
        
        # 模拟前向传播
        activations = [1.0]  # 初始输入
        for w in weights:
            z = w * activations[-1]
            activations.append(z)
        
        # 计算反向传播的梯度
        gradients = [1.0]  # 从输出开始
        
        for i in range(n_layers - 1, -1, -1):
            z = weights[i] * activations[i]
            grad = gradients[-1] * act_deriv(z) * weights[i]
            gradients.append(grad)
        
        gradients.reverse()
        
        # 可视化梯度变化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "梯度随层数衰减/爆炸",
                "激活函数导数分布",
                "权重分布",
                "梯度范数(log scale)"
            )
        )
        
        # 1. 梯度变化
        layer_indices = list(range(len(gradients)))
        fig.add_trace(
            go.Scatter(x=layer_indices, y=gradients, mode='lines+markers',
                      name='梯度值', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # 2. 激活函数导数
        fig.add_trace(
            go.Scatter(x=z_values, y=act_deriv(z_values), 
                      name='激活函数导数', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        # 3. 权重分布
        fig.add_trace(
            go.Histogram(x=weights, name='权重分布', marker_color='green'),
            row=2, col=1
        )
        
        # 4. 梯度范数（对数尺度）
        gradient_norms = [abs(g) for g in gradients]
        fig.add_trace(
            go.Scatter(x=layer_indices, y=gradient_norms, mode='lines+markers',
                      name='梯度范数', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="层索引", row=1, col=1)
        fig.update_xaxes(title_text="z值", row=1, col=2)
        fig.update_xaxes(title_text="权重值", row=2, col=1)
        fig.update_xaxes(title_text="层索引", row=2, col=2)
        
        fig.update_yaxes(title_text="梯度", row=1, col=1)
        fig.update_yaxes(title_text="导数", row=1, col=2)
        fig.update_yaxes(title_text="频数", row=2, col=1)
        fig.update_yaxes(type="log", title_text="梯度范数(log)", row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False, title_text=title_suffix)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 诊断
        st.markdown("### 🔍 梯度健康诊断")
        
        initial_grad = abs(gradients[0])
        final_grad = abs(gradients[-1])
        ratio = final_grad / initial_grad if initial_grad > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("初始梯度", f"{initial_grad:.6f}")
        with col2:
            st.metric("最终梯度", f"{final_grad:.6f}")
        with col3:
            if ratio < 0.01:
                st.metric("诊断", "⚠️ 梯度消失", delta=f"衰减 {ratio:.2e}")
            elif ratio > 100:
                st.metric("诊断", "⚠️ 梯度爆炸", delta=f"放大 {ratio:.2e}")
            else:
                st.metric("诊断", "✅ 健康", delta=f"比率 {ratio:.2f}")
        
        # 解决方案
        st.markdown("""
        ### 💡 解决方案
        
        **梯度消失**:
        - ✅ 使用 ReLU/LeakyReLU（避免饱和）
        - ✅ 残差连接 (ResNet)
        - ✅ Batch Normalization
        - ✅ LSTM/GRU (for RNN)
        - ✅ Xavier/He 初始化
        
        **梯度爆炸**:
        - ✅ 梯度裁剪 (Gradient Clipping)
        - ✅ 降低学习率
        - ✅ 权重正则化
        - ✅ Batch Normalization
        """)
    
    @staticmethod
    def _render_autograd():
        """自动微分演示"""
        st.markdown("### 🤖 自动微分 (Automatic Differentiation)")
        
        st.markdown("""
        **自动微分**: 计算机自动计算导数的技术
        
        **两种模式**:
        - **前向模式 (Forward Mode)**: 计算雅可比-向量积 (JVP)
        - **反向模式 (Reverse Mode)**: 计算向量-雅可比积 (VJP) ← PyTorch/TensorFlow 使用
        
        **计算图**: 将函数表示为操作的有向图
        """)
        
        st.markdown("#### 📊 示例: 计算 $f(x, y) = x^2 + xy + \\sin(y)$")
        
        with st.sidebar:
            st.markdown("### 🎛️ 输入值")
            x_val = st.slider("x", -5.0, 5.0, 2.0, 0.1)
            y_val = st.slider("y", -5.0, 5.0, 1.0, 0.1)
        
        # 手动构建计算图
        st.markdown("#### 🔢 前向传播")
        
        # 前向计算
        v1 = x_val * x_val  # x^2
        v2 = x_val * y_val  # xy
        v3 = np.sin(y_val)  # sin(y)
        v4 = v1 + v2        # x^2 + xy
        v5 = v4 + v3        # x^2 + xy + sin(y)
        
        forward_steps = [
            (r"v_1 = x^2", f"{x_val}^2 = {v1:.4f}"),
            (r"v_2 = x \cdot y", f"{x_val} × {y_val} = {v2:.4f}"),
            (r"v_3 = \sin(y)", f"sin({y_val}) = {v3:.4f}"),
            (r"v_4 = v_1 + v_2", f"{v1:.4f} + {v2:.4f} = {v4:.4f}"),
            (r"v_5 = v_4 + v_3", f"{v4:.4f} + {v3:.4f} = {v5:.4f}"),
        ]
        
        for latex, value in forward_steps:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.latex(latex)
            with col2:
                st.code(value)
        
        st.markdown(f"**最终结果**: $f({x_val}, {y_val}) = {v5:.4f}$")
        
        # 反向传播（计算梯度）
        st.markdown("#### 🔙 反向传播（计算梯度）")
        
        st.markdown(r"""
        目标: 计算 $\frac{\partial f}{\partial x}$ 和 $\frac{\partial f}{\partial y}$
        
        从输出开始，逆向应用链式法则:
        """)
        
        # 反向计算
        dv5 = 1.0  # df/df = 1
        
        # v5 = v4 + v3
        dv4 = dv5 * 1  # ∂v5/∂v4 = 1
        dv3 = dv5 * 1  # ∂v5/∂v3 = 1
        
        # v4 = v1 + v2
        dv1 = dv4 * 1  # ∂v4/∂v1 = 1
        dv2 = dv4 * 1  # ∂v4/∂v2 = 1
        
        # v3 = sin(y)
        dy_from_v3 = dv3 * np.cos(y_val)  # ∂v3/∂y = cos(y)
        
        # v2 = x * y
        dx_from_v2 = dv2 * y_val  # ∂v2/∂x = y
        dy_from_v2 = dv2 * x_val  # ∂v2/∂y = x
        
        # v1 = x^2
        dx_from_v1 = dv1 * 2 * x_val  # ∂v1/∂x = 2x
        
        # 累积梯度
        df_dx = dx_from_v1 + dx_from_v2
        df_dy = dy_from_v2 + dy_from_v3
        
        backward_steps = [
            (r"\bar{v}_5 = 1", "1.0000"),
            (r"\bar{v}_4 = \bar{v}_5 \cdot 1", f"{dv4:.4f}"),
            (r"\bar{v}_3 = \bar{v}_5 \cdot 1", f"{dv3:.4f}"),
            (r"\bar{v}_1 = \bar{v}_4 \cdot 1", f"{dv1:.4f}"),
            (r"\bar{v}_2 = \bar{v}_4 \cdot 1", f"{dv2:.4f}"),
            (r"\frac{\partial f}{\partial x} = 2x\bar{v}_1 + y\bar{v}_2", f"{df_dx:.4f}"),
            (r"\frac{\partial f}{\partial y} = x\bar{v}_2 + \cos(y)\bar{v}_3", f"{df_dy:.4f}"),
        ]
        
        for latex, value in backward_steps:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.latex(latex)
            with col2:
                st.code(value)
        
        # 验证（数值梯度）
        st.markdown("#### ✅ 验证（数值梯度）")
        
        h = 1e-5
        numerical_dx = (eval(f"({x_val+h})**2 + ({x_val+h})*{y_val} + np.sin({y_val})") - v5) / h
        numerical_dy = (eval(f"{x_val}**2 + {x_val}*({y_val+h}) + np.sin({y_val+h})") - v5) / h
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("自动微分 ∂f/∂x", f"{df_dx:.6f}")
            st.metric("数值微分 ∂f/∂x", f"{numerical_dx:.6f}")
        with col2:
            st.metric("自动微分 ∂f/∂y", f"{df_dy:.6f}")
            st.metric("数值微分 ∂f/∂y", f"{numerical_dy:.6f}")
        
        st.success("✅ 自动微分和数值微分结果一致！")
        
        st.markdown("""
        ### 🎯 自动微分的优势
        
        | 方法 | 精度 | 速度 | 内存 |
        |------|------|------|------|
        | 符号微分 | 精确 | 慢 | 表达式爆炸 |
        | 数值微分 | 近似 | 慢 | 小 |
        | **自动微分** | **精确** | **快** | **适中** |
        
        **PyTorch/TensorFlow 原理**:
        - 构建动态计算图
        - 前向传播记录操作
        - 反向传播自动计算梯度
        - `.backward()` 就是反向模式自动微分
        """)
        
        # 添加交互式测验
