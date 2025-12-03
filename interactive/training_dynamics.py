"""
训练动力学交互式可视化
严格按照 18.Training_Dynamics.md 中的理论实现

核心内容：
1. 初始化的物理学 - 信号传播理论
2. 归一化的几何本质 - 平滑损失景观
3. SGD的随机过程 - SDE视角
4. 神经正切核 (NTK)
5. 工程实践 - 可视化信号传播
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation

class InteractiveTrainingDynamics:
    """交互式训练动力学可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🔬 训练动力学：从炼丹到化学")
        
        st.markdown("""
        **核心认知**: 超参数不是静态配置，而是动态控制系统
        
        **三大支柱**:
        1. **初始化**: 决定信号是否"活着"传播
        2. **归一化**: 平滑损失地形，控制Lipschitz常数
        3. **学习率与Batch Size**: 控制"噪声温度"，影响泛化
        
        **从炼丹到化学**: 理解超参数背后的数学物理原理
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "信号传播与初始化",
                    "归一化的几何效果",
                    "SGD的噪声温度",
                    "线性缩放定律",
                    "神经正切核(NTK)",
                    "超参数诊断表"
                ]
            )
        
        if demo_type == "信号传播与初始化":
            InteractiveTrainingDynamics._render_initialization()
        elif demo_type == "归一化的几何效果":
            InteractiveTrainingDynamics._render_normalization()
        elif demo_type == "SGD的噪声温度":
            InteractiveTrainingDynamics._render_noise_temperature()
        elif demo_type == "线性缩放定律":
            InteractiveTrainingDynamics._render_linear_scaling()
        elif demo_type == "神经正切核(NTK)":
            InteractiveTrainingDynamics._render_ntk()
        elif demo_type == "超参数诊断表":
            InteractiveTrainingDynamics._render_diagnosis()
    

        # 添加交互式测验
        quiz_system = QuizSystem("training_dynamics")
        quizzes = QuizTemplates.get_training_dynamics_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_initialization():
        """初始化与信号传播可视化"""
        st.markdown("### 🌊 信号传播：权重初始化的生与死")
        
        st.markdown(r"""
        **核心问题**: 为什么不能用标准高斯 $\mathcal{N}(0,1)$ 初始化？
        
        **方差传播公式**:
        """)
        
        st.latex(r"""
        \text{Var}(y) = \text{Var}\left(\sum_{i=1}^{n_{in}} w_i x_i\right) 
        \approx n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)
        """)
        
        st.markdown("""
        **三种命运**:
        - $n_{in} \\cdot \\text{Var}(w) > 1$ → 信号**爆炸** 💥
        - $n_{in} \\cdot \\text{Var}(w) < 1$ → 信号**消失** 💀
        - $n_{in} \\cdot \\text{Var}(w) = 1$ → 信号**稳定** ✅
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            init_method = st.selectbox(
                "初始化方法",
                ["标准高斯(错误)", "Xavier初始化", "Kaiming初始化"]
            )
            activation = st.selectbox(
                "激活函数",
                ["ReLU", "Tanh", "Sigmoid"]
            )
            n_layers = st.slider("网络深度", 10, 100, 50, 10)
            layer_width = st.slider("层宽度", 64, 1024, 512, 64)
        
        # 模拟信号传播
        activations_std = []
        activations_mean = []
        gradient_norms = []
        
        x_var = 1.0  # 输入方差
        
        for layer in range(n_layers):
            # 根据初始化方法设置权重方差
            if init_method == "标准高斯(错误)":
                w_var = 1.0  # 错误：方差太大
            elif init_method == "Xavier初始化":
                w_var = 1.0 / layer_width
            else:  # Kaiming
                if activation == "ReLU":
                    w_var = 2.0 / layer_width
                else:
                    w_var = 1.0 / layer_width
            
            # 计算激活值方差
            pre_activation_var = layer_width * w_var * x_var
            
            # 激活函数的影响
            if activation == "ReLU":
                # ReLU杀死一半神经元
                x_var = pre_activation_var * 0.5
            elif activation == "Tanh":
                # Tanh在0附近近似线性，方差保持
                x_var = pre_activation_var * 0.5  # 近似
            else:  # Sigmoid
                x_var = pre_activation_var * 0.25  # 更强的压缩
            
            # 记录统计量
            activations_std.append(np.sqrt(x_var))
            activations_mean.append(0.0)
            
            # 梯度范数（简化模型）
            grad_norm = np.sqrt(x_var) * (0.95 ** layer)  # 指数衰减
            gradient_norms.append(grad_norm)
            
            # 检查数值稳定性
            if x_var > 1e10:  # 爆炸
                st.error(f"⚠️ 信号在第 {layer} 层爆炸！")
                break
            elif x_var < 1e-10:  # 消失
                st.warning(f"⚠️ 信号在第 {layer} 层消失！")
                break
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "激活值标准差 (前向传播)",
                "梯度范数 (反向传播)",
                "信号稳定性分析",
                "方差传播图"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        layers = list(range(len(activations_std)))
        
        # 1. 激活值标准差
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=activations_std,
                mode='lines+markers',
                name='Activation Std',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # 理想值参考线
        fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                     annotation_text="理想值 (σ=1)",
                     row=1, col=1)
        
        # 2. 梯度范数
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=gradient_norms,
                mode='lines+markers',
                name='Gradient Norm',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                     annotation_text="梯度消失阈值",
                     row=1, col=2)
        
        # 3. 稳定性分析（方差比）
        var_ratios = [std**2 for std in activations_std]
        
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=var_ratios,
                mode='lines',
                name='Variance',
                fill='tozeroy',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # 稳定区域标注
        fig.add_hrect(y0=0.5, y1=2.0, 
                     fillcolor="green", opacity=0.1,
                     annotation_text="稳定区",
                     row=2, col=1)
        
        # 4. 方差传播理论曲线
        theoretical_var = []
        for layer in range(n_layers):
            if init_method == "标准高斯(错误)":
                # 错误初始化导致指数增长/衰减
                if layer_width * 1.0 > 1:
                    var = (layer_width * 1.0) ** layer  # 爆炸
                else:
                    var = (layer_width * 1.0) ** layer  # 消失
            else:
                # 正确初始化保持方差
                var = 1.0
            
            theoretical_var.append(var)
            
            if var > 1e10:
                break
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(theoretical_var))),
                y=theoretical_var,
                mode='lines',
                name='理论方差',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="层深度", row=1, col=1)
        fig.update_yaxes(title_text="标准差", type="log", row=1, col=1)
        fig.update_xaxes(title_text="层深度", row=1, col=2)
        fig.update_yaxes(title_text="梯度范数", type="log", row=1, col=2)
        fig.update_xaxes(title_text="层深度", row=2, col=1)
        fig.update_yaxes(title_text="方差", type="log", row=2, col=1)
        fig.update_xaxes(title_text="层深度", row=2, col=2)
        fig.update_yaxes(title_text="方差", type="log", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"信号传播分析 - {init_method} + {activation}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 诊断结果
        st.markdown("### 📊 诊断结果")
        
        final_std = activations_std[-1] if activations_std else 0
        final_grad = gradient_norms[-1] if gradient_norms else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("最终激活标准差", f"{final_std:.4f}")
        
        with col2:
            st.metric("最终梯度范数", f"{final_grad:.4f}")
        
        with col3:
            if 0.5 < final_std < 2.0:
                st.success("✅ 稳定")
            elif final_std > 2.0:
                st.error("💥 爆炸")
            else:
                st.warning("💀 消失")
        
        with col4:
            depth_reached = len(activations_std)
            st.metric("有效深度", f"{depth_reached}/{n_layers}")
        
        # 理论解释
        st.markdown("### 🎓 理论要点")
        
        if init_method == "标准高斯(错误)":
            st.error("""
            **标准高斯初始化的问题**:
            - 权重方差 $\\text{Var}(w) = 1$
            - 输出方差 $\\text{Var}(y) = n_{in} \\cdot 1 \\cdot 1 = n_{in}$
            - 每层放大 $n_{in}$ 倍，指数爆炸！
            - **结果**: 深层网络无法训练
            """)
        
        elif init_method == "Xavier初始化":
            st.success("""
            **Xavier初始化** (适用于Tanh/Sigmoid):
            """)
            st.latex(r"\\text{Var}(w) = \\frac{1}{n_{in}}")
            st.markdown(r"""
            - 确保 $\text{Var}(y) = \text{Var}(x)$
            - 信号在前向和反向传播中都保持稳定
            - **适用**: 对称激活函数 (Tanh, Sigmoid)
            """)
        
        else:  # Kaiming
            st.success("""
            **Kaiming (He) 初始化** (适用于ReLU):
            """)
            st.latex(r"\\text{Var}(w) = \\frac{2}{n_{in}}")
            st.markdown(r"""
            - ReLU杀死一半神经元，需要补偿2倍方差
            - $\frac{1}{2} n_{in} \cdot \frac{2}{n_{in}} = 1$ ✅
            - **适用**: ReLU及其变体 (LeakyReLU, PReLU)
            - **结果**: 可以训练100+层的深度网络
            """)
    
    @staticmethod
    def _render_normalization():
        """归一化的几何效果可视化"""
        st.markdown("### 🏔️ 归一化：平滑损失景观")
        
        st.markdown(r"""
        **核心作用**: 降低损失函数的Lipschitz常数，平滑地形
        
        **Lipschitz约束**:
        """)
        
        st.latex(r"""
        \|\nabla L(x) - \nabla L(y)\| \le K \|x - y\|
        """)
        
        st.markdown("""
        **几何直观**: 
        - **没有归一化**: 悬崖峭壁 🏔️ (梯度剧烈变化)
        - **有归一化**: 平缓土坡 ⛰️ (梯度平滑)
        - **结果**: 可以使用更大的学习率
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            norm_type = st.selectbox(
                "归一化类型",
                ["无归一化", "Batch Normalization", "Layer Normalization"]
            )
            learning_rate = st.slider("学习率", 0.001, 1.0, 0.1, 0.01)
            curvature = st.slider("损失曲率", 1.0, 100.0, 10.0, 1.0)
        
        # 创建损失地形
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # 根据归一化类型调整地形
        if norm_type == "无归一化":
            # 陡峭的地形 - 高曲率
            Z = curvature * (X**2 + Y**2) + 5 * np.sin(X) * np.cos(Y)
            lipschitz_k = curvature
        elif norm_type == "Batch Normalization":
            # 中等平滑
            Z = (curvature / 5) * (X**2 + Y**2) + np.sin(X) * np.cos(Y)
            lipschitz_k = curvature / 5
        else:  # Layer Normalization
            # 最平滑
            Z = (curvature / 10) * (X**2 + Y**2) + 0.5 * np.sin(X) * np.cos(Y)
            lipschitz_k = curvature / 10
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "损失地形",
                "梯度场",
                "优化轨迹",
                "学习率 vs 收敛"
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )
        
        # 1. 损失地形（使用Contour热力图）
        fig.add_trace(
            go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                showscale=False,
                contours=dict(
                    coloring='heatmap',
                    showlabels=True
                )
            ),
            row=1, col=1
        )
        
        # 2. 梯度场 (2D等高线+箭头)
        fig.add_trace(
            go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                showscale=False,
                contours=dict(
                    coloring='lines',
                    showlabels=True
                )
            ),
            row=1, col=2
        )
        
        # 计算梯度
        dZ_dx = np.gradient(Z, axis=1)
        dZ_dy = np.gradient(Z, axis=0)
        
        # 添加梯度箭头（采样）
        step = 10
        for i in range(0, len(x), step):
            for j in range(0, len(y), step):
                # 梯度方向
                dx = -dZ_dx[j, i]
                dy = -dZ_dy[j, i]
                
                # 归一化
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    dx /= norm
                    dy /= norm
                
                fig.add_annotation(
                    x=X[j, i] + dx * 0.2,
                    y=Y[j, i] + dy * 0.2,
                    ax=X[j, i],
                    ay=Y[j, i],
                    xref='x2', yref='y2',
                    axref='x2', ayref='y2',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='white',
                    opacity=0.6
                )
        
        # 3. 优化轨迹模拟
        # 从不同起点开始梯度下降
        start_points = [
            [2.5, 2.5], [-2.5, 2.5], [2.5, -2.5], [-2.5, -2.5],
            [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]
        ]
        
        for start in start_points:
            trajectory = [start]
            pos = np.array(start)
            
            for _ in range(100):
                # 计算当前位置的梯度（插值）
                idx_x = np.argmin(np.abs(x - pos[0]))
                idx_y = np.argmin(np.abs(y - pos[1]))
                
                grad = np.array([dZ_dx[idx_y, idx_x], dZ_dy[idx_y, idx_x]])
                
                # 梯度下降更新
                pos = pos - learning_rate * grad
                
                trajectory.append(pos.tolist())
                
                # 边界检查
                if np.linalg.norm(pos) > 4:
                    break
                
                # 收敛检查
                if np.linalg.norm(grad) < 0.01:
                    break
            
            trajectory = np.array(trajectory)
            
            fig.add_trace(
                go.Scatter(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode='lines',
                    line=dict(width=2),
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 起点
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[0, 0]],
                    y=[trajectory[0, 1]],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 终点
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[-1, 0]],
                    y=[trajectory[-1, 1]],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='star'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 添加等高线
        fig.add_trace(
            go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                showscale=False,
                opacity=0.3,
                contours=dict(showlabels=False)
            ),
            row=2, col=1
        )
        
        # 4. 学习率 vs 收敛速度
        lr_range = np.logspace(-3, 0, 50)
        convergence_steps = []
        
        for lr in lr_range:
            # 模拟一条轨迹
            pos = np.array([2.5, 2.5])
            steps = 0
            max_steps = 1000
            
            for _ in range(max_steps):
                idx_x = np.argmin(np.abs(x - pos[0]))
                idx_y = np.argmin(np.abs(y - pos[1]))
                grad = np.array([dZ_dx[idx_y, idx_x], dZ_dy[idx_y, idx_x]])
                
                pos = pos - lr * grad
                steps += 1
                
                if np.linalg.norm(pos) > 5:  # 发散
                    steps = max_steps
                    break
                
                if np.linalg.norm(grad) < 0.01:  # 收敛
                    break
            
            convergence_steps.append(steps)
        
        fig.add_trace(
            go.Scatter(
                x=lr_range,
                y=convergence_steps,
                mode='lines+markers',
                name='收敛步数',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # 标注当前学习率
        fig.add_trace(
            go.Scatter(
                x=[learning_rate, learning_rate],
                y=[0, max(convergence_steps)],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'当前LR={learning_rate}',
                showlegend=True
            ),
            row=2, col=2
        )
        
        # 最优学习率区域 - 使用shape代替vrect
        optimal_lr = 1.0 / lipschitz_k
        # 添加背景区域标注
        fig.add_trace(
            go.Scatter(
                x=[optimal_lr * 0.5, optimal_lr * 2.0, optimal_lr * 2.0, optimal_lr * 0.5, optimal_lr * 0.5],
                y=[0, 0, max(convergence_steps), max(convergence_steps), 0],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(width=0),
                name='最优区域',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Y", row=1, col=2)
        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Y", row=2, col=1)
        fig.update_xaxes(title_text="学习率", type="log", row=2, col=2)
        fig.update_yaxes(title_text="收敛步数", row=2, col=2)
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"归一化效果 - {norm_type}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 效果分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lipschitz常数 K", f"{lipschitz_k:.2f}")
        
        with col2:
            optimal_lr = 1.0 / lipschitz_k
            st.metric("理论最优学习率", f"{optimal_lr:.4f}")
        
        with col3:
            if learning_rate > 2.0 / lipschitz_k:
                st.error("❌ 发散风险")
            elif learning_rate > optimal_lr * 0.5:
                st.success("✅ 良好")
            else:
                st.warning("⚠️ 过慢")
        
        with col4:
            speedup = curvature / lipschitz_k
            st.metric("相对加速", f"{speedup:.1f}x")
        
        # 理论解释
        st.markdown("### 🎓 Pre-Norm vs Post-Norm")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Post-Norm** (原始Transformer)")
            st.code("x = Norm(x + Sublayer(x))", language="python")
            st.warning("""
            **问题**:
            - 梯度需要经过Norm层
            - 深层容易梯度消失
            - 需要强Warmup
            """)
        
        with col_b:
            st.markdown("**Pre-Norm** (现代LLM)")
            st.code("x = x + Sublayer(Norm(x))", language="python")
            st.success("""
            **优势**:
            - 恒等映射直通
            - 梯度流畅通无阻
            - 训练极度稳定
            - GPT-3/LLaMA标配
            """)
    
    @staticmethod
    def _render_noise_temperature():
        """SGD噪声温度可视化"""
        st.markdown("### 🌡️ SGD的噪声温度：泛化的秘密")
        
        st.markdown(r"""
        **核心洞察**: SGD不是确定性算法，而是随机微分方程(SDE)
        
        **SDE形式**:
        """)
        
        st.latex(r"""
        d\theta_t = -\nabla L(\theta_t)dt + \sqrt{\frac{\eta}{B} C(\theta_t)} dW_t
        """)
        
        st.markdown(r"""
        **噪声温度**: $T = \frac{\eta}{B}$ (扩散系数)
        
        **物理类比**:
        - **高温**(大$T$): 粒子剧烈运动，只能停留在宽阔的盆地
        - **低温**(小$T$): 粒子安静，容易陷入尖锐的坑
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            learning_rate = st.slider("学习率 η", 0.001, 1.0, 0.1, 0.01)
            batch_size = st.slider("Batch Size B", 8, 512, 64, 8)
            n_iterations = st.slider("迭代次数", 100, 1000, 500, 100)
        
        # 计算噪声温度
        temperature = learning_rate / batch_size
        
        # 创建损失地形（简化的2D）
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # 两个极小值：一个尖锐，一个平坦
        sharp_minima = 10 * ((X + 1.5)**2 + (Y + 1.5)**2)  # 尖锐极小值
        flat_minima = 0.5 * ((X - 1.5)**2 + (Y - 1.5)**2)  # 平坦极小值
        
        # 组合地形
        Z = np.minimum(sharp_minima, flat_minima + 5) + 0.5 * np.sin(3*X) * np.sin(3*Y)
        
        # 模拟SGD轨迹
        np.random.seed(42)
        
        # 多条轨迹从相同起点开始
        n_trajectories = 5
        trajectories = []
        
        for traj_idx in range(n_trajectories):
            trajectory = []
            pos = np.array([0.0, 0.0])  # 从中间开始
            
            for iteration in range(n_iterations):
                # 计算梯度
                idx_x = np.argmin(np.abs(x - pos[0]))
                idx_y = np.argmin(np.abs(y - pos[1]))
                
                dZ_dx = np.gradient(Z, axis=1)
                dZ_dy = np.gradient(Z, axis=0)
                
                grad = np.array([dZ_dx[idx_y, idx_x], dZ_dy[idx_y, idx_x]])
                
                # SGD更新：确定性项 + 随机项
                noise = np.random.randn(2) * np.sqrt(temperature)
                pos = pos - learning_rate * grad + noise
                
                trajectory.append(pos.copy())
                
                # 边界检查
                if np.abs(pos[0]) > 3 or np.abs(pos[1]) > 3:
                    break
            
            trajectories.append(np.array(trajectory))
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "损失地形与SGD轨迹",
                "Sharp vs Flat Minima",
                "温度对收敛的影响",
                "泛化性能分析"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. SGD轨迹
        fig.add_trace(
            go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                showscale=False,
                contours=dict(coloring='lines')
            ),
            row=1, col=1
        )
        
        # 标注两个极小值
        fig.add_trace(
            go.Scatter(
                x=[-1.5], y=[-1.5],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='x'),
                text=['Sharp Minima'],
                textposition='top center',
                name='尖锐极小值',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[1.5], y=[1.5],
                mode='markers+text',
                marker=dict(size=15, color='green', symbol='star'),
                text=['Flat Minima'],
                textposition='bottom center',
                name='平坦极小值',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 绘制所有轨迹
        for idx, traj in enumerate(trajectories):
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode='lines',
                    line=dict(width=2),
                    opacity=0.6,
                    name=f'轨迹 {idx+1}',
                    showlegend=(idx == 0)
                ),
                row=1, col=1
            )
        
        # 2. Sharp vs Flat 的剖面图
        # 沿着两个极小值的剖面
        profile_x = np.linspace(-3, 3, 100)
        sharp_profile = 10 * (profile_x + 1.5)**2
        flat_profile = 0.5 * (profile_x - 1.5)**2 + 5
        
        fig.add_trace(
            go.Scatter(
                x=profile_x,
                y=sharp_profile,
                mode='lines',
                name='Sharp Minima',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=profile_x,
                y=flat_profile,
                mode='lines',
                name='Flat Minima',
                line=dict(color='green', width=3)
            ),
            row=1, col=2
        )
        
        # 标注"逃逸能量"
        escape_energy = temperature * 50  # 粗略估计
        fig.add_hline(y=escape_energy, line_dash="dash", line_color="orange",
                     annotation_text=f"噪声能量 ≈ {escape_energy:.2f}",
                     row=1, col=2)
        
        # 3. 温度对收敛的影响
        temp_range = np.logspace(-4, -1, 50)
        final_positions = []
        
        for temp in temp_range:
            pos = np.array([0.0, 0.0])
            
            # 简化模拟
            for _ in range(100):
                grad = pos  # 简化：假设梯度指向原点
                noise = np.random.randn(2) * np.sqrt(temp)
                pos = pos - 0.1 * grad + noise
            
            final_positions.append(np.linalg.norm(pos))
        
        fig.add_trace(
            go.Scatter(
                x=temp_range,
                y=final_positions,
                mode='lines+markers',
                name='最终位置',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # 标注当前温度
        fig.add_trace(
            go.Scatter(
                x=[temperature, temperature],
                y=[min(final_positions), max(final_positions)],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'当前T={temperature:.4f}',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # 4. 泛化性能 vs 温度
        # 模拟：低温->Sharp->差泛化，高温->Flat->好泛化
        generalization_gap = []
        
        for temp in temp_range:
            # 简化模型：温度越高，泛化越好（到一定程度）
            if temp < 0.001:
                gap = 5.0  # 低温，陷入Sharp，泛化差
            elif temp < 0.01:
                gap = 2.0 - 100 * temp  # 逐渐改善
            else:
                gap = 0.5 + 10 * (temp - 0.01)  # 过高温度，训练不稳定
            
            generalization_gap.append(gap)
        
        fig.add_trace(
            go.Scatter(
                x=temp_range,
                y=generalization_gap,
                mode='lines+markers',
                name='泛化间隙',
                line=dict(color='blue', width=3),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.2)'
            ),
            row=2, col=2
        )
        
        # 最优温度区域
        fig.add_trace(
            go.Scatter(
                x=[0.001, 0.01, 0.01, 0.001, 0.001],
                y=[0, 0, max(generalization_gap), max(generalization_gap), 0],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(width=0),
                name='最优区域',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=2, col=2
        )
        
        # 标注当前温度
        fig.add_trace(
            go.Scatter(
                x=[temperature, temperature],
                y=[0, max(generalization_gap)],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='当前温度',
                showlegend=True
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        fig.update_xaxes(title_text="参数值", row=1, col=2)
        fig.update_yaxes(title_text="损失", row=1, col=2)
        fig.update_xaxes(title_text="温度 T = η/B", type="log", row=2, col=1)
        fig.update_yaxes(title_text="收敛半径", row=2, col=1)
        fig.update_xaxes(title_text="温度 T = η/B", type="log", row=2, col=2)
        fig.update_yaxes(title_text="泛化间隙", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"噪声温度 T = {temperature:.4f} (η={learning_rate}, B={batch_size})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 温度诊断")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("学习率 η", f"{learning_rate:.4f}")
        
        with col2:
            st.metric("Batch Size B", f"{batch_size}")
        
        with col3:
            st.metric("噪声温度 T", f"{temperature:.6f}")
        
        with col4:
            if temperature > 0.01:
                st.error("❌ 温度过高")
            elif temperature > 0.001:
                st.success("✅ 温度适中")
            else:
                st.warning("⚠️ 温度过低")
        
        # 建议
        st.markdown("### 💡 调参建议")
        
        if temperature > 0.01:
            st.warning("""
            **温度过高 (T > 0.01)**:
            - 训练不稳定，损失震荡
            - **建议**: 减小学习率或增大Batch Size
            - 例如: η ← η/2 或 B ← 2B
            """)
        elif temperature < 0.0001:
            st.warning("""
            **温度过低 (T < 0.0001)**:
            - 容易陷入Sharp Minima，泛化差
            - **建议**: 增大学习率或减小Batch Size
            - 例如: η ← 2η 或 B ← B/2
            """)
        else:
            st.success("""
            **温度适中 ✅**:
            - 能够逃离Sharp Minima
            - 收敛到Flat Minima
            - 泛化性能良好
            """)
    
    @staticmethod
    def _render_linear_scaling():
        """线性缩放定律可视化"""
        st.markdown("### 📏 线性缩放定律：分布式训练的铁律")
        
        st.markdown(r"""
        **核心法则**: 扩大Batch Size必须同时扩大学习率
        """)
        
        st.latex(r"""
        \\text{当 } B_{new} = k \\cdot B_{old} \\text{ 时，} \\eta_{new} = k \\cdot \\eta_{old}
        """)
        
        st.markdown(r"""
        **原因**: 保持噪声温度 $T = \frac{\eta}{B}$ 不变
        
        **应用**: 分布式训练ImageNet、GPT等大模型的关键技巧
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            base_lr = st.slider("基准学习率", 0.001, 1.0, 0.1, 0.01)
            base_batch = st.slider("基准Batch Size", 8, 512, 32, 8)
            scale_factor = st.slider("扩展因子 k", 1, 16, 4, 1)
        
        # 计算缩放后的参数
        scaled_batch = base_batch * scale_factor
        scaled_lr_correct = base_lr * scale_factor  # 正确
        scaled_lr_wrong = base_lr  # 错误：不调整LR
        
        # 基准温度
        base_temp = base_lr / base_batch
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "线性缩放法则",
                "训练曲线对比",
                "噪声温度保持",
                "吞吐量 vs 收敛时间"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. 线性缩放关系
        scale_factors = np.arange(1, 17)
        batch_sizes = base_batch * scale_factors
        correct_lrs = base_lr * scale_factors
        wrong_lrs = np.full_like(correct_lrs, base_lr)
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=correct_lrs,
                mode='lines+markers',
                name='正确：线性缩放',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=wrong_lrs,
                mode='lines+markers',
                name='错误：不调整LR',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # 标注当前配置
        fig.add_trace(
            go.Scatter(
                x=[scaled_batch],
                y=[scaled_lr_correct],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name='当前配置',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. 模拟训练曲线
        epochs = np.arange(0, 100)
        
        # 基准训练
        loss_base = 2.0 * np.exp(-0.05 * epochs) + 0.1
        
        # 正确缩放：收敛速度和质量相同
        loss_correct = 2.0 * np.exp(-0.05 * epochs) + 0.1
        
        # 错误缩放：收敛慢或不收敛
        loss_wrong = 2.0 * np.exp(-0.01 * epochs) + 0.5
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_base,
                mode='lines',
                name=f'基准 (B={base_batch}, η={base_lr})',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_correct,
                mode='lines',
                name=f'正确缩放 (B={scaled_batch}, η={scaled_lr_correct:.3f})',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_wrong,
                mode='lines',
                name=f'错误缩放 (B={scaled_batch}, η={base_lr})',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. 温度保持
        temperatures_correct = correct_lrs / batch_sizes
        temperatures_wrong = wrong_lrs / batch_sizes
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=temperatures_correct,
                mode='lines+markers',
                name='正确：温度恒定',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=temperatures_wrong,
                mode='lines+markers',
                name='错误：温度下降',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # 理想温度参考线
        fig.add_hline(y=base_temp, line_dash="dot", line_color="gray",
                     annotation_text=f"目标温度 = {base_temp:.6f}",
                     row=2, col=1)
        
        # 4. 吞吐量 vs 训练时间权衡
        # 吞吐量线性增长（理想情况）
        throughput = batch_sizes / base_batch
        
        # 训练时间（正确缩放：保持不变；错误缩放：增加）
        time_correct = np.full_like(batch_sizes, 100.0, dtype=float)  # 收敛epoch不变
        time_wrong = 100.0 * batch_sizes / base_batch  # 错误：时间线性增长
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=throughput,
                mode='lines+markers',
                name='吞吐量提升',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                yaxis='y4'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=time_correct,
                mode='lines+markers',
                name='正确：时间不变',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=time_wrong,
                mode='lines+markers',
                name='错误：时间增加',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Batch Size", row=1, col=1)
        fig.update_yaxes(title_text="学习率 η", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_xaxes(title_text="Batch Size", row=2, col=1)
        fig.update_yaxes(title_text="温度 T = η/B", type="log", row=2, col=1)
        fig.update_xaxes(title_text="Batch Size", row=2, col=2)
        fig.update_yaxes(title_text="训练Epochs", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="线性缩放定律"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 缩放分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("缩放因子", f"{scale_factor}x")
        
        with col2:
            st.metric("Batch Size", f"{base_batch} → {scaled_batch}")
        
        with col3:
            st.metric("学习率", f"{base_lr} → {scaled_lr_correct:.3f}")
        
        with col4:
            speedup = scale_factor * 0.9  # 实际加速比略低于理论值
            st.metric("实际加速比", f"~{speedup:.1f}x")
        
        # 实践指南
        st.markdown("### 🎯 实践指南")
        
        st.success("""
        **线性缩放定律的应用**:
        
        1. **分布式训练 ImageNet**:
           - 基准: B=256, η=0.1
           - 8卡: B=2048, η=0.8
           - 保持90 epochs收敛
        
        2. **大模型训练 (GPT/LLaMA)**:
           - 扩大到数千/数万Batch Size
           - 同步调整学习率
           - 使用Warmup缓解初期震荡
        
        3. **注意事项**:
           - 极大Batch (>8K) 可能需要微调
           - 需要配合学习率Warmup
           - 监控梯度范数和损失曲线
        """)
        
        st.warning("""
        **常见错误**:
        - ❌ 增大Batch但不调整LR → 温度过低，泛化差
        - ❌ LR增长不成比例 → 训练不稳定
        - ❌ 没有Warmup → 初期爆炸
        """)
    
    @staticmethod
    def _render_ntk():
        """神经正切核(NTK)可视化"""
        st.markdown("### 🧠 神经正切核：无限宽的奇迹")
        
        st.markdown(r"""
        **核心发现**: 当网络宽度 → ∞ 时，神经网络变成核回归
        
        **懒惰训练**:
        """)
        
        st.latex(r"""
        f(x,t) \approx f(x,0) + \nabla f(x,0)^T (w_t - w_0)
        """)
        
        st.markdown("""
        **关键洞察**:
        - 权重几乎不移动，但损失降到0
        - 等价于在高维空间做线性插值
        - 解释了过参数化模型为何容易拟合
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            network_width = st.slider("网络宽度", 10, 1000, 100, 10)
            n_samples = st.slider("训练样本数", 10, 100, 50, 10)
            show_theory = st.checkbox("显示理论预测", value=True)
        
        # 模拟NTK行为
        np.random.seed(42)
        
        # 生成数据
        X_train = np.linspace(-3, 3, n_samples)
        y_train = np.sin(X_train) + 0.1 * np.random.randn(n_samples)
        
        X_test = np.linspace(-3, 3, 200)
        y_test = np.sin(X_test)
        
        # 模拟不同宽度下的训练动力学
        widths = [10, 50, 100, 500, 1000]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "训练动力学：权重变化",
                "不同宽度的拟合效果",
                "NTK vs 标准训练",
                "权重移动距离"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. 权重变化可视化
        iterations = np.arange(0, 100)
        
        for width in [10, 100, 1000]:
            # 权重移动距离（宽度越大，移动越小）
            weight_change = 1.0 / np.sqrt(width) * (1 - np.exp(-0.1 * iterations))
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=weight_change,
                    mode='lines',
                    name=f'宽度={width}',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 标注懒惰训练区域
        fig.add_hrect(
            y0=0, y1=0.01,
            fillcolor="green", opacity=0.1,
            annotation_text="懒惰训练区",
            row=1, col=1
        )
        
        # 2. 不同宽度的拟合效果
        colors = px.colors.sequential.Viridis
        
        for idx, width in enumerate([10, 50, 200, 1000]):
            # 简化模拟：宽度越大，越接近核回归
            # 核回归在训练点处精确拟合
            y_pred = np.interp(X_test, X_train, y_train)
            
            # 添加一些基于宽度的波动
            noise_scale = 1.0 / np.sqrt(width)
            y_pred += noise_scale * np.sin(5 * X_test)
            
            fig.add_trace(
                go.Scatter(
                    x=X_test,
                    y=y_pred,
                    mode='lines',
                    name=f'宽度={width}',
                    line=dict(width=2, color=colors[idx*2]),
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 真实函数
        fig.add_trace(
            go.Scatter(
                x=X_test,
                y=y_test,
                mode='lines',
                name='真实函数',
                line=dict(color='black', width=3, dash='dash'),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 训练数据
        fig.add_trace(
            go.Scatter(
                x=X_train,
                y=y_train,
                mode='markers',
                name='训练数据',
                marker=dict(size=8, color='red'),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 3. NTK vs 标准训练的损失曲线
        epochs = np.arange(0, 200)
        
        # NTK regime: 指数快速收敛
        loss_ntk = 2.0 * np.exp(-0.1 * epochs) + 0.01
        
        # 标准训练: 较慢收敛
        loss_standard = 2.0 * np.exp(-0.02 * epochs) + 0.05
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_ntk,
                mode='lines',
                name='NTK regime (宽网络)',
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_standard,
                mode='lines',
                name='标准训练 (窄网络)',
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        
        # 4. 权重移动距离 vs 宽度
        width_range = np.logspace(1, 3, 50)
        weight_movement = 1.0 / np.sqrt(width_range)
        
        fig.add_trace(
            go.Scatter(
                x=width_range,
                y=weight_movement,
                mode='lines+markers',
                name='权重移动 ∝ 1/√width',
                line=dict(color='purple', width=3),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # 标注当前宽度
        current_movement = 1.0 / np.sqrt(network_width)
        fig.add_trace(
            go.Scatter(
                x=[network_width],
                y=[current_movement],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='当前配置',
                showlegend=True
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="迭代次数", row=1, col=1)
        fig.update_yaxes(title_text="‖w_t - w_0‖", row=1, col=1)
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Y", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", type="log", row=2, col=1)
        fig.update_xaxes(title_text="网络宽度", type="log", row=2, col=2)
        fig.update_yaxes(title_text="权重移动距离", type="log", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"神经正切核 (NTK) - 宽度={network_width}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 NTK分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("网络宽度", f"{network_width}")
        
        with col2:
            movement = 1.0 / np.sqrt(network_width)
            st.metric("权重移动", f"{movement:.4f}")
        
        with col3:
            if network_width > 500:
                regime = "NTK regime"
            elif network_width > 100:
                regime = "过渡区"
            else:
                regime = "标准训练"
            st.metric("训练模式", regime)
        
        with col4:
            overparameterization = network_width / n_samples
            st.metric("过参数化比", f"{overparameterization:.1f}")
        
        # 理论解释
        st.markdown("### 🎓 理论要点")
        
        st.success(r"""
        **NTK的三大发现**:
        
        1. **懒惰训练 (Lazy Training)**:
           - 宽网络的权重几乎不动
           - $\|w_t - w_0\| \propto \frac{1}{\sqrt{width}}$
           - 但损失仍能降到0！
        
        2. **线性化近似**:
           - 网络行为可以用泰勒展开近似
           - $f(x,t) \approx f(x,0) + \nabla f^T \Delta w$
           - 等价于核回归
        
        3. **过参数化的好处**:
           - 解释了为什么大模型容易训练
           - 损失地形变得近乎凸
           - 在高维空间"插值"数据
        """)
        
        st.info("""
        **实践意义**:
        
        - **大模型容易拟合**: 过参数化 → NTK regime → 线性插值
        - **但需要泛化**: 需要正则化、数据增强等技巧
        - **理论与实践的差距**: 实际模型并非无限宽，权重会显著移动
        - **Feature Learning**: 窄网络会学习特征，宽网络只是记忆
        """)
    
    @staticmethod
    def _render_diagnosis():
        """超参数诊断表"""
        st.markdown("### 🔬 超参数诊断表：从症状到处方")
        
        st.markdown("""
        **从炼丹到化学**: 理解现象背后的数学原理
        
        下表总结了常见训练问题的诊断和解决方案：
        """)
        
        # 创建诊断表
        diagnosis_data = {
            "现象 (Symptom)": [
                "Loss不下降",
                "Loss震荡/发散",
                "训练好但泛化差",
                "深层Transformer难训",
                "大Batch训练失效",
                "梯度爆炸 (NaN)",
                "梯度消失",
                "训练初期不稳定"
            ],
            "理论原因 (Diagnosis)": [
                "信号在深层消失\n初始梯度太小",
                "损失地形太陡峭\nLipschitz常数大",
                "掉入尖锐极小值\n(Sharp Minima)",
                "梯度流在反向传播中受阻",
                "噪声温度太低\nT = η/B ≪ 1",
                "权重初始化方差过大\n信号指数放大",
                "权重初始化方差过小\n信号指数衰减",
                "学习率过大\n或Batch Size过小"
            ],
            "解决方案 (Prescription)": [
                "✅ Kaiming Init (ReLU)\n✅ Xavier Init (Tanh)\n✅ 检查激活函数",
                "✅ 添加 BatchNorm/LayerNorm\n✅ 减小学习率\n✅ 梯度裁剪",
                "✅ 增大噪声温度 T=η/B\n✅ 增大学习率 或\n✅ 减小 Batch Size",
                "✅ 使用 Pre-Norm 结构\n✅ 恒等映射路径\n✅ GPT-3/LLaMA标配",
                "✅ 线性缩放定律\n✅ η_new = k·η_old\n✅ 当 B_new = k·B_old",
                "✅ 使用正确初始化\n✅ 梯度裁剪 (Clip Norm)\n✅ 降低学习率",
                "✅ 增大初始化方差\n✅ 使用残差连接\n✅ 添加归一化层",
                "✅ 学习率 Warmup\n✅ 从小LR逐渐增大\n✅ 前5-10% epochs"
            ],
            "相关理论": [
                "信号传播理论\n方差保持",
                "Lipschitz约束\n几何平滑",
                "SDE理论\n噪声温度",
                "梯度流分析\nPre/Post-Norm",
                "线性缩放定律\n温度守恒",
                "信号传播\n方差爆炸",
                "信号传播\n方差消失",
                "优化动力学\n初始化敏感"
            ]
        }
        
        import pandas as pd
        df = pd.DataFrame(diagnosis_data)
        
        # 显示表格（使用streamlit的dataframe）
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # 交互式诊断工具
        st.markdown("### 🩺 交互式诊断工具")
        
        st.markdown("**选择您遇到的问题，获取针对性建议：**")
        
        problem = st.selectbox(
            "您的训练遇到了什么问题？",
            [
                "选择问题类型...",
                "Loss不下降",
                "Loss震荡/发散",
                "训练好但泛化差",
                "深层Transformer难训",
                "大Batch训练失效",
                "梯度爆炸(NaN)",
                "梯度消失",
                "训练初期不稳定"
            ]
        )
        
        if problem != "选择问题类型...":
            idx = df[df["现象 (Symptom)"] == problem].index[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.error(f"**🔴 问题**: {problem}")
                st.warning(f"**🔍 原因**:\n{diagnosis_data['理论原因 (Diagnosis)'][idx]}")
            
            with col2:
                st.success(f"**💊 解决方案**:\n{diagnosis_data['解决方案 (Prescription)'][idx]}")
                st.info(f"**📚 相关理论**:\n{diagnosis_data['相关理论'][idx]}")
        
        # 总结
        st.markdown("### 🎯 一句话总结")
        
        st.success("""
        **训练动力学的本质**:
        
        调参不是玄学，而是在高维空间中：
        - **控制信号的方差** (初始化 + 归一化)
        - **控制优化的温度** (学习率 + Batch Size)
        
        的动力学过程。
        """)
        
        st.info(r"""
        **三个核心公式**:
        
        1. **方差保持**: $\text{Var}(w) = \frac{1}{n_{in}}$ (Xavier) 或 $\frac{2}{n_{in}}$ (Kaiming)
        
        2. **噪声温度**: $T = \frac{\eta}{B}$ (控制Sharp vs Flat)
        
        3. **线性缩放**: $\eta_{new} = k \cdot \eta_{old}$ 当 $B_{new} = k \cdot B_{old}$
        """)

        # 添加交互式测验
