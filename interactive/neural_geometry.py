"""
交互式神经几何维度可视化
严格按照 0.4.Neural_Geometry_Dimensions.md 中的公式实现
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

# 可选导入 torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class InteractiveNeuralGeometry:
    """交互式神经几何维度可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🧮 神经网络的几何构造：维度与参数流")
        st.markdown("""
        **核心思想**: 理解神经网络的维度设计与参数增长的数学关系
        
        关键洞察：
        - **维度 (Dimensions)** 是骨架：定义张量流动的拓扑结构
        - **参数 (Parameters)** 是血肉：承载知识的实数权重
        - 参数量通常与维度呈 **二次方关系** ($N \propto D^2$)
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["参数缩放定律", "架构对比分析", "几何流动分析", "LoRA低秩分解"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "参数缩放定律":
            InteractiveNeuralGeometry._render_scaling_laws()
        elif viz_type == "架构对比分析":
            InteractiveNeuralGeometry._render_architecture_comparison()
        elif viz_type == "几何流动分析":
            InteractiveNeuralGeometry._render_geometry_flow()
        elif viz_type == "LoRA低秩分解":
            InteractiveNeuralGeometry._render_lora_decomposition()
    

        # 添加交互式测验
        quiz_system = QuizSystem("neural_geometry")
        quizzes = QuizTemplates.get_neural_geometry_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_scaling_laws():
        """可视化不同架构的参数缩放定律"""
        st.markdown("### 📈 参数缩放定律 (Scaling Laws)")
        
        st.latex(r"""
        \text{核心洞察：所有现代架构的参数量都与 } d_{model} \text{ 呈二次方关系}
        """)
        
        with st.sidebar:
            max_dim = st.slider("最大维度", 512, 8192, 4096, 256)
            show_models = st.multiselect("显示模型", 
                ["Linear/RNN", "CNN", "Mamba", "Transformer"], 
                default=["Linear/RNN", "CNN", "Mamba", "Transformer"])
        
        # 定义维度范围
        d_model = np.linspace(128, max_dim, 100)
        
        fig = go.Figure()
        
        if "Linear/RNN" in show_models:
            # Linear/RNN: ~ d^2
            params_linear = d_model**2 + d_model
            fig.add_trace(go.Scatter(
                x=d_model, y=params_linear/1e6,
                mode='lines',
                name='Linear/RNN ($N \propto D^2$)',
                line=dict(dash='dash')
            ))
        
        if "CNN" in show_models:
            # CNN: 假设 3x3 kernel
            params_cnn = 9 * d_model**2
            fig.add_trace(go.Scatter(
                x=d_model, y=params_cnn/1e6,
                mode='lines',
                name='CNN 3×3 ($N \propto 9D^2$)'
            ))
        
        if "Mamba" in show_models:
            # Mamba: ~ 6 * d^2
            params_mamba = 6 * d_model**2
            fig.add_trace(go.Scatter(
                x=d_model, y=params_mamba/1e6,
                mode='lines',
                name='Mamba ($N \propto 6D^2$)'
            ))
        
        if "Transformer" in show_models:
            # Transformer: ~ 12 * d^2
            params_transformer = 12 * d_model**2
            fig.add_trace(go.Scatter(
                x=d_model, y=params_transformer/1e6,
                mode='lines',
                name='Transformer ($N \propto 12D^2$)',
                line=dict(width=3)
            ))
        
        # 标出著名模型的维度
        model_markers = [
            (768, "BERT Base"),
            (1024, "GPT-2 Small"),
            (2048, "GPT-2 Medium"),
            (4096, "LLaMA-7B"),
            (8192, "GPT-3")
        ]
        
        for dim, name in model_markers:
            if dim <= max_dim:
                fig.add_vline(x=dim, line_dash="dot", line_color="gray", opacity=0.5)
                fig.add_annotation(
                    x=dim, y=max_dim**2 * 15 / 1e6,
                    text=name,
                    showarrow=False,
                    textangle=-90
                )
        
        fig.update_layout(
            title="神经网络参数量随模型维度的增长趋势",
            xaxis_title="模型维度 ($d_{model}$)",
            yaxis_title="参数量 (Millions)",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示关键洞察
        st.markdown("### 🔍 关键洞察")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **二次方统治**：
            - 所有架构都遵循 $N \propto D^2$
            - 维度翻倍 → 参数量增4倍
            - 显存需求呈平方增长
            """)
        
        with col2:
            st.markdown("""
            **架构效率对比**：
            - Linear/RNN: 基准线
            - CNN: 9倍参数开销
            - Mamba: 6倍参数开销  
            - Transformer: 12倍参数开销
            """)
    
    @staticmethod
    def _render_architecture_comparison():
        """对比不同架构的参数构成"""
        st.markdown("### 🏗️ 架构对比分析")
        
        with st.sidebar:
            d_model = st.slider("模型维度 $d_{model}$", 128, 2048, 768, 128)
            seq_len = st.slider("序列长度", 128, 4096, 1024, 128)
            
        # 计算各组件参数
        components = {}
        
        # 1. Linear Layer
        components['Linear'] = d_model * d_model + d_model
        
        # 2. CNN Layer (假设 3x3, 输入输出通道相同)
        components['CNN'] = d_model * 3 * 3 * d_model + d_model
        
        # 3. RNN/LSTM
        components['RNN'] = (d_model + d_model) * d_model
        components['LSTM'] = 4 * components['RNN']
        
        # 4. Transformer Components
        components['Attention'] = 4 * d_model**2  # Q,K,V,O
        components['FFN'] = 8 * d_model**2        # 4*d_model expansion
        components['Transformer'] = components['Attention'] + components['FFN']
        
        # 5. Mamba (简化估计)
        components['Mamba'] = 6 * d_model**2
        
        # 创建参数对比图
        fig = go.Figure()
        
        names = list(components.keys())
        params = list(components.values())
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'pink', 'cyan', 'brown']
        
        fig.add_trace(go.Bar(
            x=names,
            y=[p/1e6 for p in params],
            marker_color=colors[:len(names)],
            text=[f'{p/1e6:.1f}M' for p in params],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"不同架构的参数量对比 ($d_{{model}}$ = {d_model})",
            xaxis_title="架构类型",
            yaxis_title="参数量 (Millions)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分解
        st.markdown("### 📊 参数构成分解")
        
        if 'Transformer' in components:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Attention", f"{components['Attention']/1e6:.1f}M")
                st.caption("Q, K, V, O 投影")
            
            with col2:
                st.metric("FFN", f"{components['FFN']/1e6:.1f}M")
                st.caption("两层的线性变换")
            
            with col3:
                st.metric("总计", f"{components['Transformer']/1e6:.1f}M")
                st.caption("一个Transformer Block")
        
        # 计算复杂度分析
        st.markdown("### ⚡ 计算复杂度分析")
        
        complexity_data = {
            '架构': ['Linear', 'CNN', 'RNN', 'LSTM', 'Attention', 'Mamba'],
            '参数复杂度': ['$O(D^2)$', '$O(D^2)$', '$O(D^2)$', '$O(4D^2)$', '$O(D^2)$', '$O(D^2)$'],
            '计算复杂度': ['$O(D^2)$', '$O(HW \cdot D^2)$', '$O(T \cdot D^2)$', '$O(4T \cdot D^2)$', '$O(T^2 \cdot D)$', '$O(T \cdot D^2)$'],
            '内存复杂度': ['$O(D^2)$', '$O(D^2)$', '$O(D^2)$', '$O(4D^2)$', '$O(T^2)$', '$O(T \cdot D)$']
        }
        
        df = pd.DataFrame(complexity_data)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def _render_geometry_flow():
        """分析张量在神经网络中的几何流动"""
        st.markdown("### 🌊 张量几何流动分析")
        
        with st.sidebar:
            batch_size = st.slider("批次大小", 1, 32, 4, 1)
            input_channels = st.slider("输入通道", 1, 64, 3, 1)
            height = st.slider("图像高度", 16, 128, 32, 8)
            width = st.slider("图像宽度", 16, 128, 32, 8)
            hidden_dim = st.slider("隐藏维度", 32, 512, 128, 32)
        
        # 模拟张量流动
        layers = []
        shapes = []
        params = []
        
        # 1. 输入层
        input_shape = (batch_size, input_channels, height, width)
        layers.append("Input")
        shapes.append(input_shape)
        params.append(0)
        
        # 2. 卷积层
        conv_out_channels = 16
        conv_kernel = 3
        conv_params = input_channels * conv_out_channels * conv_kernel**2 + conv_out_channels
        conv_shape = (batch_size, conv_out_channels, height, width)
        layers.append("Conv2d")
        shapes.append(conv_shape)
        params.append(conv_params)
        
        # 3. 展平层
        flattened_size = conv_out_channels * height * width
        flat_shape = (batch_size, flattened_size)
        layers.append("Flatten")
        shapes.append(flat_shape)
        params.append(0)
        
        # 4. 全连接层
        fc_shape = (batch_size, hidden_dim)
        fc_params = flattened_size * hidden_dim + hidden_dim
        layers.append("Linear")
        shapes.append(fc_shape)
        params.append(fc_params)
        
        # 5. 输出层
        output_dim = 10
        output_shape = (batch_size, output_dim)
        output_params = hidden_dim * output_dim + output_dim
        layers.append("Output")
        shapes.append(output_shape)
        params.append(output_params)
        
        # 可视化张量形状变化
        fig = go.Figure()
        
        # 计算每个张量的"体积"作为可视化指标
        volumes = [np.prod(shape) for shape in shapes]
        
        fig.add_trace(go.Scatter(
            x=layers,
            y=volumes,
            mode='markers+lines',
            marker=dict(size=[max(10, v/1000) for v in volumes]),
            text=[f'{shape}' for shape in shapes],
            hovertemplate='<b>%{x}</b><br>Shape: %{text}<br>Volume: %{y:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title="张量形状在网络中的变化",
            xaxis_title="网络层",
            yaxis_title="张量体积 (元素数量)",
            height=500,
            yaxis_type="log"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 参数分布
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=layers,
            y=params,
            marker_color='lightblue',
            text=[f'{p:,}' for p in params],
            textposition='outside'
        ))
        
        fig2.update_layout(
            title="各层参数量分布",
            xaxis_title="网络层",
            yaxis_title="参数量",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 详细信息表
        st.markdown("### 📋 详细流动信息")
        
        flow_data = {
            '层': layers,
            '输入形状': [str(shape) for shape in shapes],
            '参数量': [f'{p:,}' for p in params],
            '参数占比': [f'{p/sum(params)*100:.1f}%' if sum(params) > 0 else '0%' for p in params]
        }
        
        df = pd.DataFrame(flow_data)
        st.dataframe(df, use_container_width=True)
        
        st.info(f"""
        **总参数量**: {sum(params):,} ({sum(params)/1e6:.2f}M)
        
        **关键观察**：
        - 卷积层参数与空间尺寸无关
        - 全连接层参数量最大（稠密连接）
        - 展平操作不增加参数，但改变张量拓扑
        """)
    
    @staticmethod
    def _render_lora_decomposition():
        """LoRA低秩分解可视化"""
        st.markdown("### 🔧 LoRA低秩分解 (Low-Rank Adaptation)")
        
        st.latex(r"""
        \mathbf{W}_{new} = \mathbf{W}_{old} + \Delta \mathbf{W} = \mathbf{W}_{old} + \mathbf{B}\mathbf{A}
        """)
        
        with st.sidebar:
            d_model = st.slider("模型维度 $d_{model}$", 512, 8192, 4096, 512)
            rank = st.slider("LoRA 秩 $r$", 4, 64, 8, 4)
        
        # 计算参数量
        full_params = d_model * d_model
        lora_params = 2 * rank * d_model
        compression_ratio = lora_params / full_params
        
        # 可视化矩阵分解
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["原始权重矩阵", "LoRA 分解", "参数对比"],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. 原始权重矩阵 (简化可视化)
        matrix_size = min(d_model, 100)  # 限制显示大小
        W_original = np.random.randn(matrix_size, matrix_size)
        
        fig.add_trace(
            go.Heatmap(z=W_original, colorscale='Viridis', showscale=False),
            row=1, col=1
        )
        
        # 2. LoRA 分解示意图
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2, 3],
                y=[0, 0, 0, 0],
                mode='markers+lines',
                marker=dict(size=[20, 15, 15, 20]),
                line=dict(width=2),
                text=[f'W<br>{d_model}×{d_model}', f'B<br>{d_model}×{rank}', f'A<br>{rank}×{d_model}', f'ΔW<br>{d_model}×{d_model}'],
                textposition='bottom center'
            ),
            row=1, col=2
        )
        
        # 3. 参数对比
        fig.add_trace(
            go.Bar(
                x=['原始权重', 'LoRA 更新'],
                y=[full_params/1e6, lora_params/1e6],
                marker_color=['lightblue', 'lightgreen']
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title="LoRA 低秩分解原理",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("原始参数", f"{full_params/1e6:.1f}M")
        
        with col2:
            st.metric("LoRA参数", f"{lora_params/1e6:.1f}M")
        
        with col3:
            st.metric("压缩比", f"{compression_ratio*100:.2f}%")
        
        with col4:
            saved_params = full_params - lora_params
            st.metric("节省参数", f"{saved_params/1e6:.1f}M")
        
        # 公式验证
        st.markdown("### 📐 公式验证")
        
        st.latex(r"""
        \text{压缩比} = \frac{\text{LoRA Params}}{\text{Full Params}} = \frac{2 \times r \times d_{model}}{d_{model}^2} = \frac{2r}{d_{model}}
        """)
        
        st.code(f"""
        # 当前参数：
        d_model = {d_model}
        r = {rank}
        
        # 计算：
        原始参数 = {d_model} × {d_model} = {full_params:,}
        LoRA参数 = 2 × {rank} × {d_model} = {lora_params:,}
        压缩比 = {lora_params} / {full_params} = {compression_ratio:.6f} = {compression_ratio*100:.2f}%
        """)
        
        # 不同秩的对比
        st.markdown("### 📊 不同秩的效率对比")
        
        ranks = [4, 8, 16, 32, 64]
        efficiencies = []
        
        for r in ranks:
            if r <= d_model:
                ratio = (2 * r * d_model) / (d_model * d_model)
                efficiencies.append(ratio * 100)
            else:
                efficiencies.append(None)
        
        fig = go.Figure()
        
        valid_ranks = [r for r, e in zip(ranks, efficiencies) if e is not None]
        valid_efficiencies = [e for e in efficiencies if e is not None]
        
        fig.add_trace(go.Scatter(
            x=valid_ranks,
            y=valid_efficiencies,
            mode='markers+lines',
            marker=dict(size=10),
            text=[f'{e:.3f}%' for e in valid_efficiencies],
            textposition='top center'
        ))
        
        fig.add_hline(y=compression_ratio*100, line_dash="dash", line_color="red", 
                     annotation_text=f"当前设置 (r={rank})")
        
        fig.update_layout(
            title=f"LoRA 秩选择对参数效率的影响 ($d_{{model}}$ = {d_model})",
            xaxis_title="LoRA 秩 (r)",
            yaxis_title="参数占比 (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **LoRA 的几何意义**：
        - 权重更新发生在低维子空间
        - 大幅减少可训练参数
        - 保持模型性能的同时提升训练效率
        """)

        # 添加交互式测验
