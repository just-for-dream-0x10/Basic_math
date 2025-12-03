"""
交互式希尔伯特空间可视化
严格按照 12.Hilbert_space.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

# 可选导入 scipy
try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft, fftfreq, fftshift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class InteractiveHilbertSpace:
    """交互式希尔伯特空间可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🌐 希尔伯特空间、傅里叶变换与神经网络")
        st.markdown("""
        **核心思想**: 神经网络在希尔伯特空间中学习映射，傅里叶变换提供了优雅的理论视角
        
        关键概念：
        - **希尔伯特空间**: 神经网络的数学宇宙，所有操作都在其中发生
        - **傅里叶变换**: 在这个宇宙中旋转坐标系，让卷积变得简单
        - **卷积定理**: 时域卷积 = 频域相乘
        - **Parseval恒等式**: 能量守恒，范数不变
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["傅里叶变换基础", "卷积定理演示", "CNN频域分析", "图傅里叶变换"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "傅里叶变换基础":
            InteractiveHilbertSpace._render_fourier_basics()
        elif viz_type == "卷积定理演示":
            InteractiveHilbertSpace._render_convolution_theorem()
        elif viz_type == "CNN频域分析":
            InteractiveHilbertSpace._render_cnn_frequency()
        elif viz_type == "图傅里叶变换":
            InteractiveHilbertSpace._render_graph_fourier()
    

        # 添加交互式测验
        quiz_system = QuizSystem("hilbert_space")
        quizzes = QuizTemplates.get_hilbert_space_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_fourier_basics():
        """傅里叶变换基础概念"""
        st.markdown("### 📈 傅里叶变换基础")
        
        st.latex(r"""
        \mathcal{F}[f](\omega) = \hat{f}(\omega) = \int_{\mathbb{R}^d} f(x) e^{-i\langle \omega, x \rangle} \,dx
        """)
        
        with st.sidebar:
            signal_type = st.selectbox("信号类型", ["正弦波", "方波", "高斯脉冲", "复合信号"])
            frequency = st.slider("基频", 1, 20, 5)
            sampling_rate = st.slider("采样率", 50, 500, 200)
            duration = st.slider("持续时间", 1, 5, 2)
        
        # 生成信号
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        if signal_type == "正弦波":
            input_signal = np.sin(2 * np.pi * frequency * t)
            signal_name = f"sin({frequency}Hz)"
        elif signal_type == "方波":
            if SCIPY_AVAILABLE:
                input_signal = scipy_signal.square(2 * np.pi * frequency * t)
            else:
                # 简单的方波实现
                input_signal = np.sign(np.sin(2 * np.pi * frequency * t))
            signal_name = f"square({frequency}Hz)"
        elif signal_type == "高斯脉冲":
            input_signal = np.exp(-((t - duration/2)**2) / (2 * (1/frequency)**2))
            signal_name = "gaussian"
        else:  # 复合信号
            input_signal = (np.sin(2 * np.pi * frequency * t) + 
                           0.5 * np.sin(2 * np.pi * 3 * frequency * t) +
                           0.3 * np.sin(2 * np.pi * 5 * frequency * t))
            signal_name = "composite"
        
        # 计算傅里叶变换
        if SCIPY_AVAILABLE:
            fft_vals = fft(input_signal)
            fft_freq = fftfreq(len(t), 1/sampling_rate)
        else:
            # 使用numpy的FFT
            fft_vals = np.fft.fft(input_signal)
            fft_freq = np.fft.fftfreq(len(t), 1/sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = fft_freq > 0
        positive_freq = fft_freq[positive_freq_idx]
        positive_fft = np.abs(fft_vals[positive_freq_idx])
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["时域信号", "频域幅度谱"],
            vertical_spacing=0.1
        )
        
        # 时域信号
        fig.add_trace(
            go.Scatter(x=t, y=input_signal, mode='lines', name=signal_name),
            row=1, col=1
        )
        
        # 频域幅度谱
        fig.add_trace(
            go.Scatter(x=positive_freq, y=positive_fft, mode='lines', name='幅度谱'),
            row=2, col=1
        )
        
        fig.update_layout(
            title="傅里叶变换：时域与频域的对偶性",
            height=600,

        # 添加交互式测验
            showlegend=False
        )
        fig.update_xaxes(title_text="时间 (s)", row=1, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="幅度", row=1, col=1)
        fig.update_yaxes(title_text="频域幅度", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Parseval 定理验证
        time_energy = np.sum(np.abs(input_signal)**2) / len(input_signal)
        freq_energy = np.sum(np.abs(fft_vals)**2) / len(fft_vals)**2
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("时域能量", f"{time_energy:.4f}")
        with col2:
            st.metric("频域能量", f"{freq_energy:.4f}")
        with col3:
            energy_ratio = freq_energy / time_energy if time_energy > 0 else 0
            st.metric("能量比", f"{energy_ratio:.6f}")
        
        st.info("""
        **Parseval 恒等式验证**：
        - 时域能量应等于频域能量
        - 这证明了傅里叶变换是酉变换，保持内积不变
        - 在神经网络中，这对应于 LayerNorm/BatchNorm 的能量守恒思想
        """)
    
    @staticmethod
    def _render_convolution_theorem():
        """卷积定理演示"""
        st.markdown("### 🔄 卷积定理演示")
        
        st.latex(r"""
        \mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]
        """)
        
        with st.sidebar:
            signal_type = st.selectbox("输入信号", ["脉冲", "阶跃", "正弦", "噪声"])
            kernel_type = st.selectbox("卷积核类型", ["低通", "高通", "带通", "自定义"])
            kernel_size = st.slider("卷积核大小", 3, 31, 11, 2)
        
        # 生成信号
        n_points = 200
        x = np.linspace(0, 10, n_points)
        
        if signal_type == "脉冲":
            input_signal = np.zeros(n_points)
            input_signal[n_points//2] = 1
        elif signal_type == "阶跃":
            input_signal = np.ones(n_points)
            input_signal[:n_points//2] = 0
        elif signal_type == "正弦":
            input_signal = np.sin(2 * np.pi * 2 * x)
        else:  # 噪声
            input_signal = np.random.randn(n_points)
        
        # 生成卷积核
        if kernel_type == "低通":
            if SCIPY_AVAILABLE:
                kernel = scipy_signal.windows.gaussian(kernel_size, std=kernel_size/6)
            else:
                # 简单的高斯核实现
                x = np.arange(kernel_size) - kernel_size//2
                kernel = np.exp(-(x**2) / (2 * (kernel_size/6)**2))
                kernel = kernel / kernel.sum()
        elif kernel_type == "高通":
            if SCIPY_AVAILABLE:
                kernel = scipy_signal.windows.gaussian(kernel_size, std=kernel_size/6)
            else:
                x = np.arange(kernel_size) - kernel_size//2
                kernel = np.exp(-(x**2) / (2 * (kernel_size/6)**2))
                kernel = kernel / kernel.sum()
            kernel = -kernel
            kernel[kernel_size//2] += 1
        elif kernel_type == "带通":
            t_k = np.arange(kernel_size) - kernel_size//2
            carrier = np.sin(2 * np.pi * t_k / (kernel_size/4))
            if SCIPY_AVAILABLE:
                envelope = scipy_signal.windows.gaussian(kernel_size, std=kernel_size/6)
            else:
                envelope = np.exp(-(t_k**2) / (2 * (kernel_size/6)**2))
                envelope = envelope / envelope.sum()
            kernel = carrier * envelope
        else:  # 自定义
            kernel = np.array([1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0])[:kernel_size]
            if len(kernel) < kernel_size:
                kernel = np.pad(kernel, (0, kernel_size - len(kernel)))
        
        # 时域卷积
        convolution_time = np.convolve(input_signal, kernel, mode='same')
        
        # 频域计算
        if SCIPY_AVAILABLE:
            signal_fft = fft(input_signal)
            kernel_fft = fft(kernel, n=n_points)  # 零填充到相同长度
            from scipy.fft import ifft
            convolution_freq = np.real(ifft(signal_fft * kernel_fft))
        else:
            signal_fft = np.fft.fft(input_signal)
            kernel_fft = np.fft.fft(kernel, n=n_points)
            convolution_freq = np.real(np.fft.ifft(signal_fft * kernel_fft))
        
        # 创建可视化
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "输入信号", "输入信号频谱",
                "卷积核", "卷积核频谱", 
                "时域卷积结果", "频域相乘结果"
            ],
            vertical_spacing=0.08
        )
        
        # 输入信号
        fig.add_trace(go.Scatter(x=x, y=input_signal, mode='lines', name='输入'), row=1, col=1)
        if SCIPY_AVAILABLE:
            fft_shifted = fftshift(signal_fft)
        else:
            fft_shifted = np.fft.fftshift(signal_fft)
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=np.abs(fft_shifted), 
                               mode='lines', name='频谱'), row=1, col=2)
        
        # 卷积核
        fig.add_trace(go.Scatter(x=np.arange(kernel_size), y=kernel, mode='lines+markers', 
                               name='核'), row=2, col=1)
        if SCIPY_AVAILABLE:
            kernel_fft_padded = fftshift(kernel_fft)
        else:
            kernel_fft_padded = np.fft.fftshift(kernel_fft)
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=np.abs(kernel_fft_padded), 
                               mode='lines', name='核频谱'), row=2, col=2)
        
        # 卷积结果
        fig.add_trace(go.Scatter(x=x, y=convolution_time, mode='lines', name='时域卷积'), 
                     row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=convolution_freq, mode='lines', name='频域相乘', 
                               line=dict(dash='dash')), row=3, col=2)
        
        fig.update_layout(
            title="卷积定理：时域卷积 vs 频域相乘",
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 误差分析
        error = np.mean(np.abs(convolution_time - convolution_freq))
        st.metric("时域与频域结果误差", f"{error:.2e}")
        
        st.success("""
        **卷积定理验证**：
        - 时域卷积结果应与频域相乘结果完全一致
        - 误差应接近机器精度（~1e-15）
        - 这是CNN频域理解的数学基础
        """)
    
    @staticmethod
    def _render_cnn_frequency():
        """CNN频域分析"""
        st.markdown("### 🧠 CNN频域分析")
        
        st.markdown("""
        **关键洞察**：
        - 训练后的CNN滤波器通常呈现**低通特性**
        - 自然图像的大部分能量集中在低频
        - 深层网络学习更精细的频率结构
        """)
        
        with st.sidebar:
            layer_depth = st.selectbox("网络层深度", ["浅层", "中层", "深层"])
            filter_size = st.slider("滤波器大小", 3, 7, 3)
            num_filters = st.slider("滤波器数量", 4, 16, 8)
        
        # 模拟不同深度的CNN滤波器
        np.random.seed(42)
        
        if layer_depth == "浅层":
            # 浅层：简单的边缘检测器
            filters = []
            for i in range(num_filters):
                if i % 4 == 0:  # 水平边缘
                    f = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
                elif i % 4 == 1:  # 垂直边缘
                    f = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
                elif i % 4 == 2:  # 对角线
                    f = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
                else:  # 反对角线
                    f = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
                filters.append(f + np.random.randn(3, 3) * 0.1)
        elif layer_depth == "中层":
            # 中层：更复杂的模式
            filters = []
            for i in range(num_filters):
                f = np.random.randn(filter_size, filter_size) * 0.5
                # 添加一些结构
                if SCIPY_AVAILABLE:
                    gaussian_1d = scipy_signal.windows.gaussian(filter_size, std=filter_size/4)
                else:
                    x = np.arange(filter_size) - filter_size//2
                    gaussian_1d = np.exp(-(x**2) / (2 * (filter_size/4)**2))
                    gaussian_1d = gaussian_1d / gaussian_1d.sum()
                f += gaussian_1d.reshape(-1, 1) * gaussian_1d.reshape(1, -1)
                filters.append(f)
        else:  # 深层
            # 深层：更精细的频率结构
            filters = []
            for i in range(num_filters):
                # 高频成分更多
                f = np.random.randn(filter_size, filter_size) * 0.3
                for j in range(2):
                    freq = np.random.randint(2, 5)
                    phase = np.random.rand() * 2 * np.pi
                    x = np.arange(filter_size)
                    y = np.arange(filter_size)
                    X, Y = np.meshgrid(x, y)
                    wave = np.sin(2 * np.pi * freq * (X * np.cos(phase) + Y * np.sin(phase)) / filter_size)
                    f += wave * 0.2
                filters.append(f)
        
        # 分析每个滤波器的频域特性
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[f"滤波器 {i+1}" for i in range(min(4, num_filters))],
            specs=[[{"type": "heatmap"}]*4, [{"type": "scatter"}]*4]
        )
        
        for i in range(min(4, num_filters)):
            f = filters[i]
            
            # 空域表示
            fig.add_trace(
                go.Heatmap(z=f, colorscale='RdBu', showscale=False),
                row=1, col=i+1
            )
            
            # 频域分析
            f_padded = np.zeros((32, 32))
            start = (32 - filter_size) // 2
            f_padded[start:start+filter_size, start:start+filter_size] = f
            
            if SCIPY_AVAILABLE:
                fft_f = fftshift(fft2(f_padded))
            else:
                fft_f = np.fft.fftshift(np.fft.fft2(f_padded))
            magnitude = np.abs(fft_f)
            
            # 径向平均
            center = 16
            y, x = np.ogrid[:32, :32]
            r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
            radial_mean = [magnitude[r == i].mean() if np.any(r == i) else 0 for i in range(0, 16)]
            
            fig.add_trace(
                go.Scatter(x=np.arange(len(radial_mean)), y=radial_mean, mode='lines'),
                row=2, col=i+1
            )
        
        fig.update_layout(
            title=f"CNN滤波器频域分析 ({layer_depth})",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="频率半径", row=2, col=1)
        fig.update_yaxes(title_text="平均幅度", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 频域特性统计
        st.markdown("### 📊 频域特性统计")
        
        freq_energies = []
        for f in filters:
            f_padded = np.zeros((32, 32))
            start = (32 - filter_size) // 2
            f_padded[start:start+filter_size, start:start+filter_size] = f
            if SCIPY_AVAILABLE:
                fft_f = fftshift(fft2(f_padded))
            else:
                fft_f = np.fft.fftshift(np.fft.fft2(f_padded))
            magnitude = np.abs(fft_f)
            
            # 计算低频能量比例
            center = 16
            y, x = np.ogrid[:32, :32]
            r = np.sqrt((x - center)**2 + (y - center)**2)
            
            low_freq_mask = r <= 8
            total_energy = np.sum(magnitude**2)
            low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
            freq_energies.append(low_freq_energy / total_energy if total_energy > 0 else 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均低频能量比", f"{np.mean(freq_energies):.3f}")
        with col2:
            st.metric("低频能量比标准差", f"{np.std(freq_energies):.3f}")
        with col3:
            st.metric("滤波器数量", num_filters)
        
        st.info(f"""
        **{layer_depth}滤波器特点**：
        - 浅层：主要是边缘检测器，能量集中在特定频率
        - 中层：学习更复杂的模式，频率分布更均匀
        - 深层：包含更多高频成分，学习精细特征
        """)
    
    @staticmethod
    def _render_graph_fourier():
        """图傅里叶变换"""
        st.markdown("### 🕸️ 图傅里叶变换")
        
        st.latex(r"""
        \text{图傅里叶变换: } \hat{x} = U^T x \\
        \text{谱卷积: } g_\theta * x = U g_\theta(\Lambda) U^T x
        """)
        
        with st.sidebar:
            graph_type = st.selectbox("图类型", ["环形图", "路径图", "随机图", "网格图"])
            num_nodes = st.slider("节点数量", 8, 32, 16)
            filter_type = st.selectbox("图滤波器类型", ["低通", "高通", "带通"])
        
        # 生成图结构
        if graph_type == "环形图":
            adj = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                adj[i, (i-1) % num_nodes] = 1
                adj[i, (i+1) % num_nodes] = 1
        elif graph_type == "路径图":
            adj = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes - 1):
                adj[i, i+1] = 1
                adj[i+1, i] = 1
        elif graph_type == "随机图":
            np.random.seed(42)
            adj = np.random.rand(num_nodes, num_nodes) < 0.3
            adj = adj | adj.T  # 对称化
            np.fill_diagonal(adj, 0)  # 无自环
        else:  # 网格图
            size = int(np.sqrt(num_nodes))
            adj = np.zeros((num_nodes, num_nodes))
            for i in range(size):
                for j in range(size):
                    idx = i * size + j
                    # 右邻居
                    if j < size - 1:
                        adj[idx, idx + 1] = 1
                        adj[idx + 1, idx] = 1
                    # 下邻居
                    if i < size - 1:
                        adj[idx, idx + size] = 1
                        adj[idx + size, idx] = 1
        
        # 计算图拉普拉斯矩阵
        degree = np.sum(adj, axis=1)
        L = np.diag(degree) - adj
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # 生成信号
        signal = np.random.randn(num_nodes)
        
        # 图傅里叶变换
        signal_gft = eigenvectors.T @ signal
        
        # 设计图滤波器
        if filter_type == "低通":
            g_lambda = np.exp(-eigenvalues / 2)  # 低通：衰减高频
        elif filter_type == "高通":
            g_lambda = 1 - np.exp(-eigenvalues / 2)  # 高通：保留高频
        else:  # 带通
            center_freq = len(eigenvalues) // 3
            g_lambda = np.exp(-(eigenvalues - center_freq)**2 / 10)
        
        # 应用图滤波器
        filtered_signal = eigenvectors @ (g_lambda * signal_gft)
        
        # 创建可视化
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "图结构", "特征值", "原始信号",
                "图傅里叶变换", "滤波器响应", "滤波后信号"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 图结构可视化
        pos = None
        if graph_type == "环形图":
            angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            pos = np.column_stack([np.cos(angles), np.sin(angles)])
        elif graph_type == "路径图":
            pos = np.column_stack([np.arange(num_nodes), np.zeros(num_nodes)])
        elif graph_type == "网格图":
            size = int(np.sqrt(num_nodes))
            pos = np.array([(i, j) for i in range(size) for j in range(size)])
        else:  # 随机图 - 使用弹簧布局简化版
            np.random.seed(42)
            pos = np.random.randn(num_nodes, 2)
        
        # 绘制图的边
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj[i, j] > 0:
                    fig.add_trace(
                        go.Scatter(x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]], 
                                 mode='lines', line=dict(color='lightgray', width=1),
                                 showlegend=False),
                        row=1, col=1
                    )
        
        # 绘制图的节点
        fig.add_trace(
            go.Scatter(x=pos[:, 0], y=pos[:, 1], mode='markers', 
                       marker=dict(size=10, color='blue'),
                       showlegend=False),
            row=1, col=1
        )
        
        # 特征值
        fig.add_trace(
            go.Scatter(x=np.arange(len(eigenvalues)), y=eigenvalues, mode='markers+lines',
                       showlegend=False),
            row=1, col=2
        )
        
        # 原始信号
        fig.add_trace(
            go.Scatter(x=np.arange(num_nodes), y=signal, mode='lines+markers',
                       showlegend=False),
            row=1, col=3
        )
        
        # 图傅里叶变换
        fig.add_trace(
            go.Scatter(x=np.arange(num_nodes), y=signal_gft, mode='lines+markers',
                       showlegend=False),
            row=2, col=1
        )
        
        # 滤波器响应
        fig.add_trace(
            go.Scatter(x=np.arange(num_nodes), y=g_lambda, mode='lines+markers',
                       showlegend=False),
            row=2, col=2
        )
        
        # 滤波后信号
        fig.add_trace(
            go.Scatter(x=np.arange(num_nodes), y=filtered_signal, mode='lines+markers',
                       showlegend=False),
            row=2, col=3
        )
        
        fig.update_layout(
            title="图傅里叶变换与谱卷积",
            height=700,
            showlegend=False
        )
        
        # 更新坐标轴标题
        fig.update_xaxes(title_text="节点索引", row=1, col=3)
        fig.update_xaxes(title_text="频率索引", row=2, col=1)
        fig.update_xaxes(title_text="特征值索引", row=2, col=2)
        fig.update_xaxes(title_text="节点索引", row=2, col=3)
        
        fig.update_yaxes(title_text="信号值", row=1, col=3)
        fig.update_yaxes(title_text="GFT系数", row=2, col=1)
        fig.update_yaxes(title_text="滤波器响应", row=2, col=2)
        fig.update_yaxes(title_text="滤波后信号", row=2, col=3)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 能量分析
        original_energy = np.sum(signal**2)
        filtered_energy = np.sum(filtered_signal**2)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("原始信号能量", f"{original_energy:.2f}")
        with col2:
            st.metric("滤波后能量", f"{filtered_energy:.2f}")
        with col3:
            energy_ratio = filtered_energy / original_energy if original_energy > 0 else 0
            st.metric("能量保留比", f"{energy_ratio:.3f}")
        
        st.info(f"""
        **图傅里叶变换特点**：
        - 特征值对应图的"频率"，小特征值=低频，大特征值=高频
        - {filter_type}滤波器：{'保留' if filter_type == '低通' else '衰减'}低频成分，{'衰减' if filter_type == '低通' else '保留'}高频成分
        - 谱卷积在图频域中实现，等价于复杂图卷积操作
        """)
        
        # 添加缺少的导入
        try:
            from scipy.fft import fft2
        except ImportError:
            # 如果scipy不可用，使用numpy的fft
            from numpy.fft import fft2


# 为了兼容性，添加numpy.fft导入
from numpy.fft import fft2

        # 添加交互式测验
