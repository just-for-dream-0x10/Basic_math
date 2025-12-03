"""
CNN数学基础交互式可视化
严格按照 11.CNN_Mathematical_Foundations.md.md 中的理论实现

核心内容：
1. 卷积定理与希尔伯特空间
2. 池化的多分辨率分析
3. ReLU的频带混合作用
4. 群论视角：平移群、置换群、欧几里得群
5. CNN vs Transformer vs Geometric Transformer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.fft import fft2, ifft2, fftshift
from scipy import signal


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveCNNMathFoundations:
    """交互式CNN数学基础可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🔬 CNN数学基础：从希尔伯特空间到群论")
        
        st.markdown(r"""
        **核心洞察**: 
        
        > "卷积神经网络的底层是傅里叶变换，傅里叶变换的底层是希尔伯特空间坐标变换"
        
        **这不是玄学，而是20世纪最深刻的数学洞察之一！**
        
        **三层理解**:
        1. **线性算子**: 卷积定理 → 频域对角化
        2. **非线性关键**: ReLU作为频带混合器
        3. **群论视角**: CNN/Transformer/Geometric Transformer的统一理解
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "卷积定理与希尔伯特空间",
                    "池化的多分辨率分析",
                    "ReLU的频带混合",
                    "群论视角：对称性",
                    "架构对比：CNN vs Transformer",
                    "完整思想体系"
                ]
            )
        
        if demo_type == "卷积定理与希尔伯特空间":
            InteractiveCNNMathFoundations._render_convolution_theorem()
        elif demo_type == "池化的多分辨率分析":
            InteractiveCNNMathFoundations._render_pooling()
        elif demo_type == "ReLU的频带混合":
            InteractiveCNNMathFoundations._render_relu_frequency()
        elif demo_type == "群论视角：对称性":
            InteractiveCNNMathFoundations._render_group_theory()
        elif demo_type == "架构对比：CNN vs Transformer":
            InteractiveCNNMathFoundations._render_architecture_comparison()
        elif demo_type == "完整思想体系":
            InteractiveCNNMathFoundations._render_complete_framework()
    

        # 添加交互式测验
        quiz_system = QuizSystem("cnn_math_foundations")
        quizzes = QuizTemplates.get_cnn_math_foundations_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_convolution_theorem():
        """卷积定理与希尔伯特空间可视化"""
        st.markdown("### 🎯 卷积定理：空域卷积 = 频域乘法")
        
        st.markdown(r"""
        **卷积定理 (Convolution Theorem)**:
        """)
        
        st.latex(r"""
        f * g = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g))
        """)
        
        st.markdown(r"""
        **深层含义**:
        - 空域: 卷积（复杂的滑动窗口运算）
        - 频域: 逐点乘法（简单的对角矩阵运算）
        - **傅里叶变换**将稠密矩阵运算变为稀疏的对角化运算
        
        **希尔伯特空间视角**:
        - $\mathcal{F}$ 是 $L^2$ 空间的**酉算子** (Unitary Operator)
        - 保持内积不变（能量守恒）
        - 只是基底旋转：从**位置基底**到**频率基底**
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            kernel_type = st.selectbox(
                "卷积核类型",
                ["边缘检测(Sobel)", "模糊(Gaussian)", "锐化(Laplacian)"]
            )
        
        # 创建示例图像（简单的几何形状）
        img_size = 64
        x, y = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
        
        # 创建一个正方形
        img = np.zeros((img_size, img_size))
        img[20:45, 20:45] = 1.0
        
        # 定义卷积核
        if kernel_type == "边缘检测(Sobel)":
            kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]) / 4
        elif kernel_type == "模糊(Gaussian)":
            kernel = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]]) / 16
        else:  # Laplacian
            kernel = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])
        
        # 空域卷积
        conv_result_spatial = signal.convolve2d(img, kernel, mode='same', boundary='wrap')

        # 添加交互式测验
        
        # 频域方法
        # 1. FFT图像
        img_fft = fft2(img)
        img_fft_shifted = fftshift(img_fft)
        
        # 2. FFT卷积核（需要padding到相同大小）
        kernel_padded = np.zeros_like(img)
        kh, kw = kernel.shape
        kernel_padded[:kh, :kw] = kernel
        kernel_fft = fft2(kernel_padded)
        kernel_fft_shifted = fftshift(kernel_fft)
        
        # 3. 频域相乘
        result_fft = img_fft * kernel_fft
        result_fft_shifted = fftshift(result_fft)
        
        # 4. IFFT回到空域
        conv_result_freq = np.real(ifft2(result_fft))
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "原始图像",
                "卷积核",
                "空域卷积结果",
                "频域：图像谱",
                "频域：核谱",
                "频域：逐点相乘后"
            ),
            specs=[[{"type": "xy"}] * 3,
                   [{"type": "xy"}] * 3]
        )
        
        # 第一行：空域
        fig.add_trace(go.Heatmap(z=img, colorscale='Greys', showscale=False),
                     row=1, col=1)
        
        fig.add_trace(go.Heatmap(z=kernel, colorscale='RdBu', zmid=0, showscale=False),
                     row=1, col=2)
        
        fig.add_trace(go.Heatmap(z=conv_result_spatial, colorscale='Viridis', showscale=False),
                     row=1, col=3)
        
        # 第二行：频域（对数幅度谱）
        fig.add_trace(go.Heatmap(z=np.log(np.abs(img_fft_shifted) + 1),
                                colorscale='Hot', showscale=False),
                     row=2, col=1)
        
        fig.add_trace(go.Heatmap(z=np.log(np.abs(kernel_fft_shifted) + 1),
                                colorscale='Hot', showscale=False),
                     row=2, col=2)
        
        fig.add_trace(go.Heatmap(z=np.log(np.abs(result_fft_shifted) + 1),
                                colorscale='Hot', showscale=False),
                     row=2, col=3)
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text=f"卷积定理演示 - {kernel_type}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 验证两种方法的等价性
        st.markdown("### 📊 验证：空域 = 频域")
        
        diff = np.max(np.abs(conv_result_spatial - conv_result_freq))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("空域卷积耗时", "O(N²M²)")
        
        with col2:
            st.metric("频域方法耗时", "O(N² log N)")
        
        with col3:
            st.metric("两者差异", f"{diff:.2e}")
        
        if diff < 1e-10:
            st.success("✅ 两种方法结果完全相同！")
        
        st.markdown("### 🎓 深层理解")
        
        st.success(r"""
        **CNN为什么高效？**
        
        1. **卷积核共享**: 同一个核在整个图像上滑动
           - 参数量: O(k²) 而非 O(N²)
           - 这隐式利用了**平移不变性**
        
        2. **频域对角化**: 
           - 卷积在频域变为逐点乘法
           - 相当于在最优坐标系下工作
           - GPU可以极致并行化
        
        3. **希尔伯特空间**: 
           - 傅里叶变换是酉变换（保持内积）
           - 能量守恒: $\|f\|^2 = \|\mathcal{F}(f)\|^2$
           - 从"位置基底"旋转到"频率基底"
        
        **结论**: CNN的成功不是偶然，而是数学必然！
        """)
        
        st.info("""
        **与希尔伯特空间模块的联系**:
        
        回顾Ch 12的希尔伯特空间笔记:
        - 傅里叶基是 $L^2$ 空间的完备正交基
        - 卷积核学习 = 在傅里叶基下学习对角矩阵
        - 这是为什么CNN可以用FFT加速的数学原因
        """)
    
    @staticmethod
    def _render_pooling():
        """池化的多分辨率分析可视化"""
        st.markdown("### 🔍 池化：数学显微镜的变焦")
        
        st.markdown(r"""
        **池化 = 小波变换的离散版本**
        
        **多分辨率分析 (Multiresolution Analysis)**:
        """)
        
        st.latex(r"""
        V_0 \subset V_1 \subset V_2 \subset \cdots \subset L^2(\mathbb{R})
        """)
        
        st.markdown(r"""
        **直观理解**:
        - $V_0$: 原始分辨率（看到所有细节）
        - $V_1$: 2×2 Max Pooling后（看到大尺度特征）
        - $V_2$: 4×4 Pooling后（看到更粗的特征）
        
        **关键洞察**: 
        - 池化不是"丢弃信息"，而是"提取尺度"
        - 每一层CNN学习的是不同尺度的特征
        - 这和小波变换的多分辨率分析完全一致！
        """)
        
        # 创建示例图像
        img_size = 64
        img = np.zeros((img_size, img_size))
        
        # 添加不同尺度的特征
        # 小尺度：细节纹理
        x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
        img += 0.3 * np.sin(x * 0.5) * np.sin(y * 0.5)
        
        # 中尺度：边缘
        img[20:25, :] = 1.0
        img[:, 30:35] = 1.0
        
        # 大尺度：整体结构
        img[10:30, 40:60] = 0.8
        
        # 不同的池化
        def pooling(img, pool_size, method='max'):
            h, w = img.shape
            new_h, new_w = h // pool_size, w // pool_size
            pooled = np.zeros((new_h, new_w))
            
            for i in range(new_h):
                for j in range(new_w):
                    block = img[i*pool_size:(i+1)*pool_size,
                               j*pool_size:(j+1)*pool_size]
                    if method == 'max':
                        pooled[i, j] = np.max(block)
                    elif method == 'avg':
                        pooled[i, j] = np.mean(block)
            
            return pooled
        
        pool_2x2 = pooling(img, 2, 'max')
        pool_4x4 = pooling(img, 4, 'max')
        pool_8x8 = pooling(img, 8, 'max')
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "V₀: 原始 (64×64)",
                "V₁: 2×2 Pooling (32×32)",
                "V₂: 4×4 Pooling (16×16)",
                "V₃: 8×8 Pooling (8×8)"
            ),
            specs=[[{"type": "xy"}] * 2,
                   [{"type": "xy"}] * 2]
        )
        
        fig.add_trace(go.Heatmap(z=img, colorscale='Viridis', showscale=False),
                     row=1, col=1)
        fig.add_trace(go.Heatmap(z=pool_2x2, colorscale='Viridis', showscale=False),
                     row=1, col=2)
        fig.add_trace(go.Heatmap(z=pool_4x4, colorscale='Viridis', showscale=False),
                     row=2, col=1)
        fig.add_trace(go.Heatmap(z=pool_8x8, colorscale='Viridis', showscale=False),
                     row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False,
                         title_text="多分辨率分析：池化的数学本质")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(r"""
        **池化的三重作用**:
        
        1. **降采样**: 减少计算量，从 $N^2$ 降到 $(N/2)^2$
        
        2. **平移不变性**: 
           - 输入微小偏移 → 输出几乎不变
           - Max Pooling容忍局部扰动
        
        3. **多尺度特征**:
           - 浅层：细节特征（纹理、边缘）
           - 深层：语义特征（物体、场景）
           - 这就是为什么ResNet、VGG逐层降分辨率
        
        **数学联系**: 
        - 小波变换: 连续的多分辨率分析
        - CNN池化: 离散的多分辨率分析
        - 两者本质相同！
        """)
    
    @staticmethod
    def _render_relu_frequency():
        """ReLU的频带混合可视化"""
        st.markdown("### ⚡ ReLU：频带混合器的秘密")
        
        st.markdown(r"""
        **核心问题**: 如果CNN只有线性卷积层，为什么需要非线性？
        
        **答案**: **ReLU作为频带混合器**
        
        **数学原理**:
        - 线性卷积: 频域对角化（每个频率独立处理）
        - ReLU: 打破对角结构，混合不同频率
        - 结果: 网络可以学习**非线性频域滤波器**
        """)
        
        # 创建一个简单信号
        t = np.linspace(0, 2*np.pi, 1000)
        
        # 单一频率信号
        signal_single = np.sin(5 * t)
        
        # 通过ReLU
        relu_output = np.maximum(signal_single, 0)
        
        # 计算频谱
        fft_original = np.fft.fft(signal_single)
        fft_relu = np.fft.fft(relu_output)
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        
        # 只取正频率
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        fft_original_pos = np.abs(fft_original[pos_mask])
        fft_relu_pos = np.abs(fft_relu[pos_mask])
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "原始信号: sin(5t)",
                "ReLU后: max(sin(5t), 0)",
                "原始频谱: 单一频率",
                "ReLU后频谱: 多个频率!"
            ),
            specs=[[{"type": "xy"}] * 2,
                   [{"type": "xy"}] * 2]
        )
        
        # 时域
        fig.add_trace(go.Scatter(x=t, y=signal_single, mode='lines',
                                name='原始', line=dict(color='blue', width=2)),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=t, y=relu_output, mode='lines',
                                name='ReLU', line=dict(color='red', width=2)),
                     row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 频域
        fig.add_trace(go.Bar(x=freqs_pos[:50], y=fft_original_pos[:50],
                            name='原始频谱', marker_color='blue'),
                     row=2, col=1)
        
        fig.add_trace(go.Bar(x=freqs_pos[:50], y=fft_relu_pos[:50],
                            name='ReLU频谱', marker_color='red'),
                     row=2, col=2)
        
        fig.update_xaxes(title_text="时间", row=1, col=1)
        fig.update_xaxes(title_text="时间", row=1, col=2)
        fig.update_xaxes(title_text="频率", row=2, col=1)
        fig.update_xaxes(title_text="频率", row=2, col=2)
        fig.update_yaxes(title_text="幅度", row=1, col=1)
        fig.update_yaxes(title_text="幅度", row=1, col=2)
        fig.update_yaxes(title_text="能量", row=2, col=1)
        fig.update_yaxes(title_text="能量", row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False,
                         title_text="ReLU的频带混合效应")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(r"""
        **关键发现**:
        
        1. **输入**: 单一频率 (5 Hz) → 频谱只有一根线
        
        2. **ReLU后**: 出现了多个频率分量！
           - 原始频率 (5 Hz)
           - 谐波 (10 Hz, 15 Hz, ...)
           - 直流分量 (0 Hz)
        
        3. **数学解释**: 
           - ReLU是分段线性函数
           - 傅里叶级数展开包含所有奇次谐波
           - $\text{ReLU}(\sin(x)) = \frac{1}{\pi} + \frac{1}{2}\sin(x) - \frac{2}{\pi}\sum_{k=1}^{\infty} \frac{\cos(2kx)}{4k^2-1}$
        
        **结论**: 
        - 线性层: 频率独立处理（对角矩阵）
        - ReLU: 频率耦合混合（非对角）
        - 这种组合使得CNN能学习任意复杂的频域滤波器！
        """)
        
        st.info("""
        **深度学习的本质**: 
        
        CNN = 线性滤波器 + 非线性混合器的级联
        
        - 每一层卷积: 在当前频域空间中线性变换
        - ReLU: 打破对角结构，引入频率耦合
        - 下一层卷积: 在新的频域空间继续变换
        
        深度网络通过多次"线性-非线性"交替，逐步构建复杂的特征表示
        """)
    
    @staticmethod
    def _render_group_theory():
        """群论视角：对称性可视化"""
        st.markdown("### 🎭 群论：对称性是深度学习的灵魂")
        
        st.markdown(r"""
        **核心思想**: 网络架构应该尊重数据的对称性
        
        **群 (Group)**: 满足封闭性、结合律、单位元、逆元的集合
        
        **三种重要的群**:
        """)
        
        # 创建对比表格
        import pandas as pd
        
        symmetry_table = pd.DataFrame({
            '群': ['平移群 (CNN)', '置换群 (Transformer)', '欧几里得群 (Geometric DL)'],
            '定义': [
                '所有平移变换 {t_x, t_y}',
                '所有排列 {π: π(1), π(2), ...}',
                '旋转+平移 {R, t}'
            ],
            '网络架构': ['CNN', 'Transformer', 'E(n)-GNN'],
            '等变性': [
                'f(T(x)) = T(f(x))',
                'f(π(x)) = π(f(x))',
                'f(g·x) = g·f(x)'
            ],
            '应用': ['图像、视频', '序列、集合', '分子、点云']
        })
        
        st.dataframe(symmetry_table, use_container_width=True)
        
        st.markdown("### 🔄 等变性 vs 不变性")
        
        st.info(r"""
        **等变性 (Equivariance)**:
        $$f(g \cdot x) = g \cdot f(x)$$
        
        输入变换 → 输出同样变换
        
        **例子**: 
        - 图像平移 → 特征图也平移
        - 这是CNN卷积层的性质
        
        **不变性 (Invariance)**:
        $$f(g \cdot x) = f(x)$$
        
        输入变换 → 输出不变
        
        **例子**: 
        - 图像旋转 → 分类结果不变
        - 这是全局池化层的性质
        
        **关系**: 
        - 中间层需要等变性（保持结构）
        - 最后层需要不变性（任务目标）
        - CNN = 等变层 + 不变层
        """)
        
        st.success("""
        **为什么这很重要？**
        
        1. **CNN的成功**: 
           - 尊重图像的平移对称性
           - 卷积核共享 = 平移等变性的数学必然
        
        2. **Transformer的威力**: 
           - 尊重序列的置换不变性（Self-Attention）
           - 位置编码打破对称性（加入归纳偏置）
        
        3. **几何深度学习的未来**: 
           - 分子、蛋白质的3D旋转对称性
           - E(3)-等变网络（AlphaFold 3）
           - 引力波、粒子物理的洛伦兹群
        
        **统一框架**: 
        > 所有好的深度学习架构都可以理解为某个群的等变映射！
        """)
    
    @staticmethod
    def _render_architecture_comparison():
        """架构对比：CNN vs Transformer vs Geometric Transformer"""
        st.markdown("### 🏗️ 架构演化：从群论视角理解")
        
        st.markdown("""
        **核心问题**: 为什么我们需要这么多不同的架构？
        
        **答案**: **不同的数据有不同的对称性**
        """)
        
        # 创建架构对比
        import pandas as pd
        
        arch_comparison = pd.DataFrame({
            '架构': ['CNN', 'Transformer', 'Geometric Transformer (E(n)-GNN)'],
            '尊重的群': ['平移群 T(2)', '置换群 S_n', '欧几里得群 E(3)'],
            '核心操作': [
                '卷积 (局部+权重共享)',
                'Self-Attention (全局+置换不变)',
                'E(n)-等变消息传递'
            ],
            '归纳偏置': [
                '局部性+平移不变性',
                '最小偏置（需要大数据）',
                '旋转+平移不变性'
            ],
            '数据效率': ['高', '低→中（需要预训练）', '极高'],
            '适用数据': [
                '图像（网格结构）',
                '序列、集合（无序）',
                '3D点云、分子'
            ],
            '代表模型': [
                'ResNet, EfficientNet',
                'BERT, GPT, ViT',
                'AlphaFold 3, EGNN'
            ]
        })
        
        st.dataframe(arch_comparison, use_container_width=True, height=300)
        
        st.markdown("### 📊 复杂度对比")
        
        # 绘制复杂度对比图
        n_range = np.arange(10, 1001, 10)
        
        # CNN: O(k² * n)，k是卷积核大小，假设k=3
        cnn_complexity = 9 * n_range
        
        # Transformer: O(n²)
        transformer_complexity = n_range ** 2
        
        # Geometric: O(n * k_neighbors)，假设k=32
        geometric_complexity = 32 * n_range
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=n_range, y=cnn_complexity,
            mode='lines', name='CNN: O(n)',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=n_range, y=transformer_complexity,
            mode='lines', name='Transformer: O(n²)',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=n_range, y=geometric_complexity,
            mode='lines', name='Geometric: O(n)',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="计算复杂度对比",
            xaxis_title="序列长度 / 像素数",
            yaxis_title="计算量",
            yaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 如何选择架构？")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**选择CNN**")
            st.success("""
            ✅ 图像、视频
            ✅ 局部特征重要
            ✅ 数据量中等
            ✅ 需要高效推理
            
            例子: 
            - 图像分类
            - 目标检测
            - 语义分割
            """)
        
        with col2:
            st.markdown("**选择Transformer**")
            st.info("""
            ✅ 长程依赖
            ✅ 集合、序列
            ✅ 大规模数据
            ✅ 可以预训练
            
            例子:
            - NLP (BERT/GPT)
            - ViT (图像也行)
            - 多模态融合
            """)
        
        with col3:
            st.markdown("**选择Geometric**")
            st.warning("""
            ✅ 3D几何数据
            ✅ 旋转对称性
            ✅ 小样本学习
            ✅ 物理约束
            
            例子:
            - 分子性质预测
            - 蛋白质结构
            - 粒子物理
            """)
        
        st.success("""
        **统一理解**:
        
        所有这些架构都是**群等变神经网络**的特例：
        
        - **CNN**: 平移群 T(2) 的等变网络
        - **Transformer**: 置换群 S_n 的不变网络（Self-Attention）
        - **E(n)-GNN**: 欧几里得群 E(n) 的等变网络
        
        **几何深度学习 (Geometric Deep Learning)**: 
        
        提供了统一的框架理解所有这些架构。不是"炼丹"，而是基于数学原理的设计！
        """)
    
    @staticmethod
    def _render_complete_framework():
        """完整思想体系"""
        st.markdown("### 🌌 完整思想体系：从希尔伯特空间到群论")
        
        st.markdown("""
        这张图展示了CNN数学基础的完整思想体系：
        """)
        
        # 使用Sankey图展示思想体系
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = [
                    "希尔伯特空间",
                    "傅里叶变换",
                    "卷积定理",
                    "CNN架构",
                    "非线性",
                    "多分辨率",
                    "群论",
                    "等变性",
                    "泛化能力"
                ],
                color = [
                    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                    "#95E1D3", "#FFD93D", "#6BCB77", "#FD79A8", "#A29BFE"
                ]
            ),
            link = dict(
                source = [0, 0, 1, 2, 2, 3, 3, 6, 7],
                target = [1, 6, 2, 3, 4, 5, 8, 7, 8],
                value = [1, 1, 1, 1, 0.5, 0.5, 1, 1, 1],
                label = [
                    "酉变换",
                    "对称性",
                    "频域对角化",
                    "卷积层",
                    "ReLU频带混合",
                    "池化多尺度",
                    "归纳偏置",
                    "平移等变",
                    "泛化"
                ]
            )
        )])
        
        fig.update_layout(
            title_text="CNN数学基础思想体系",
            font_size=12,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📚 三层理解")
        
        with st.expander("第一层：线性代数与泛函分析"):
            st.markdown(r"""
            **希尔伯特空间 $L^2$**:
            - 图像是 $L^2(\mathbb{R}^2)$ 中的向量
            - 内积定义能量: $\langle f, g \rangle = \int f(x)g(x)dx$
            
            **傅里叶变换**:
            - 酉算子: $\mathcal{F}: L^2 \to L^2$
            - 保持内积: $\langle f, g \rangle = \langle \mathcal{F}f, \mathcal{F}g \rangle$
            - 基底变换: 从位置基到频率基
            
            **卷积定理**:
            - $f * g = \mathcal{F}^{-1}(\mathcal{F}f \cdot \mathcal{F}g)$
            - 稠密矩阵 → 对角矩阵
            """)
        
        with st.expander("第二层：非线性理论"):
            st.markdown(r"""
            **为什么需要非线性？**
            
            - 纯线性网络: $f = W_L \cdots W_2 W_1 = W_{total}$
            - 等价于单层线性变换
            - 表达能力极度受限
            
            **ReLU的作用**:
            - 打破频域对角结构
            - 引入频率耦合
            - 使网络能逼近任意非线性函数
            
            **万能逼近定理**:
            - 单隐层神经网络可以逼近任何连续函数
            - 深度网络降低所需神经元数（指数优势）
            """)
        
        with st.expander("第三层：群论与几何"):
            st.markdown(r"""
            **为什么群论？**
            
            数据的对称性决定了最优架构：
            
            1. **图像**: 平移对称性 → CNN
            2. **序列**: 置换对称性 → Transformer
            3. **分子**: 旋转对称性 → E(3)-GNN
            
            **等变性原理**:
            $$f(g \cdot x) = g \cdot f(x)$$
            
            **好处**:
            - 减少需要学习的参数
            - 提高样本效率
            - 保证泛化能力
            
            **未来**: 更多群 → 更多架构
            - 时空群 → 视频理解
            - 洛伦兹群 → 粒子物理
            - 李群 → 机器人控制
            """)
        
        st.markdown("### 🎓 核心要点总结")
        
        st.success("""
        **CNN不是经验设计，而是数学必然**:
        
        1. **卷积**: 傅里叶空间的对角化（希尔伯特空间）
        2. **池化**: 多分辨率分析（小波理论）
        3. **ReLU**: 频带混合器（非线性泛函分析）
        4. **权重共享**: 平移等变性（群论）
        
        **深度学习 = 群等变 + 希尔伯特空间 + 非线性泛函分析**
        
        这不是事后解释，而是设计原则！
        """)
        
        st.info("""
        **与其他模块的联系**:
        
        - **Ch 1 卷积**: 卷积定理的工程实现
        - **Ch 12 希尔伯特空间**: CNN的理论基础
        - **Ch 25 信号处理**: 傅里叶、小波与CNN的统一
        - **Ch 20 GCN**: 从欧几里得空间到图空间的推广
        
        **推荐学习路径**:
        1. 先看卷积模块（工程直觉）
        2. 再看希尔伯特空间（理论基础）
        3. 最后看本模块（深层理解）
        """)

# 导入必要的包
import pandas as pd

        # 添加交互式测验
