"""
工程速查：维度分析与参数估算 - 交互式可视化
基于 AppxD_Dimensions_Parameters.md

核心内容：
1. Transformer参数计算器
2. CNN参数计算器
3. 显存占用估算
4. 架构对比（GPT vs BERT）
5. 训练显存解剖
6. 混合精度与量化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates
from common.smart_cache import cache_medium, cache_heavy, cache_numpy_computation

class InteractiveDimensionsParameters:
    """交互式维度分析与参数估算"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🔧 工程速查：维度分析与参数估算")
        
        st.markdown(r"""
        **工程师必备工具**：快速计算模型参数量、显存占用、训练成本
        
        本模块帮助你：
        - 📊 估算模型参数量
        - 💾 计算显存需求
        - ⚡ 优化资源配置
        - 🎯 理解架构差异
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择工具")
            tool_type = st.selectbox(
                "工具类型",
                [
                    "Transformer参数计算器",
                    "CNN参数计算器",
                    "显存占用估算",
                    "架构对比 (GPT vs BERT)",
                    "训练显存解剖",
                    "混合精度与量化"
                ]
            )
        
        if tool_type == "Transformer参数计算器":
            InteractiveDimensionsParameters._render_transformer_calculator()
        elif tool_type == "CNN参数计算器":
            InteractiveDimensionsParameters._render_cnn_calculator()
        elif tool_type == "显存占用估算":
            InteractiveDimensionsParameters._render_memory_calculator()
        elif tool_type == "架构对比 (GPT vs BERT)":
            InteractiveDimensionsParameters._render_architecture_comparison()
        elif tool_type == "训练显存解剖":
            InteractiveDimensionsParameters._render_memory_anatomy()
        elif tool_type == "混合精度与量化":
            InteractiveDimensionsParameters._render_precision_quantization()
    

        # 添加交互式测验
        quiz_system = QuizSystem("dimensions_parameters")
        quizzes = QuizTemplates.get_dimensions_parameters_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_transformer_calculator():
        """Transformer参数计算器"""
        st.markdown("### 🤖 Transformer参数计算器")
        
        st.markdown(r"""
        **核心公式**：
        """)
        
        st.latex(r"""
        \begin{align}
        \text{Embedding} &= V \times d_{model} \\
        \text{Self-Attention} &= 4 \times d_{model}^2 \\
        \text{FFN} &= 8 \times d_{model}^2 \\
        \text{One Layer} &\approx 12 \times d_{model}^2 \\
        \text{Total} &\approx V \times d_{model} + L \times 12 \times d_{model}^2
        \end{align}
        """)
        
        with st.sidebar:
            st.markdown("#### 模型配置")
            vocab_size = st.number_input("词表大小 (V)", 10000, 100000, 50257, 1000)
            d_model = st.number_input("隐藏维度 (d_model)", 256, 8192, 768, 64)
            n_layers = st.number_input("层数 (L)", 1, 96, 12, 1)
            n_heads = st.number_input("注意力头数", 1, 128, 12, 1)
            d_ff = st.number_input("FFN维度", 256, 32768, d_model * 4, 256)
            
            st.markdown("#### 额外选项")
            include_bias = st.checkbox("包含bias", value=True)
            include_ln = st.checkbox("包含LayerNorm", value=True)
        
        # 计算各部分参数
        # 1. Embedding
        embedding_params = vocab_size * d_model
        
        # 2. 每层的参数
        # Self-Attention: Q, K, V, O 各有 d_model x d_model
        attn_params = 4 * d_model * d_model
        if include_bias:
            attn_params += 4 * d_model
        
        # FFN: W1, W2
        ffn_params = d_model * d_ff + d_ff * d_model
        if include_bias:
            ffn_params += d_ff + d_model
        
        # LayerNorm: gamma, beta (两个LN)
        ln_params = 0
        if include_ln:
            ln_params = 2 * (2 * d_model)
        
        # 每层总参数
        params_per_layer = attn_params + ffn_params + ln_params
        
        # 所有层
        all_layers_params = n_layers * params_per_layer
        
        # 输出层（可选）
        output_params = d_model * vocab_size

        # 添加交互式测验
        
        # 总参数
        total_params = embedding_params + all_layers_params + output_params
        
        # 显示结果
        st.markdown("### 📊 参数量分解")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总参数量", f"{total_params/1e6:.1f}M")
        with col2:
            st.metric("层数", n_layers)
        with col3:
            st.metric("d_model", d_model)
        
        # 详细分解
        breakdown = {
            "组件": [
                "Embedding层",
                f"Transformer层 (×{n_layers})",
                "  - Self-Attention",
                "  - FFN",
                "  - LayerNorm" if include_ln else None,
                "输出层",
                "总计"
            ],
            "参数量": [
                f"{embedding_params/1e6:.2f}M",
                f"{all_layers_params/1e6:.2f}M",
                f"{attn_params/1e6:.2f}M (每层)",
                f"{ffn_params/1e6:.2f}M (每层)",
                f"{ln_params/1e3:.2f}K (每层)" if include_ln else None,
                f"{output_params/1e6:.2f}M",
                f"{total_params/1e6:.2f}M"
            ],
            "占比": [
                f"{embedding_params/total_params*100:.1f}%",
                f"{all_layers_params/total_params*100:.1f}%",
                f"{attn_params/params_per_layer*100:.1f}%",
                f"{ffn_params/params_per_layer*100:.1f}%",
                f"{ln_params/params_per_layer*100:.2f}%" if include_ln else None,
                f"{output_params/total_params*100:.1f}%",
                "100.0%"
            ]
        }
        
        # 过滤None
        breakdown = {k: [v for v in vals if v is not None] 
                    for k, vals in breakdown.items()}
        
        df = pd.DataFrame(breakdown)
        st.dataframe(df, use_container_width=True)
        
        # 可视化
        fig = go.Figure()
        
        # 饼图：参数分布
        labels = ["Embedding", "Transformer层", "输出层"]
        values = [embedding_params, all_layers_params, output_params]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hovertemplate='%{label}<br>%{value:.2f}M 参数<br>%{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title="参数分布",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 层参数详细分解
        st.markdown("### 🔍 单层参数详细分解")
        
        layer_breakdown = {
            "组件": ["Q矩阵", "K矩阵", "V矩阵", "O矩阵", "FFN-W1", "FFN-W2"],
            "形状": [
                f"({d_model}, {d_model})",
                f"({d_model}, {d_model})",
                f"({d_model}, {d_model})",
                f"({d_model}, {d_model})",
                f"({d_model}, {d_ff})",
                f"({d_ff}, {d_model})"
            ],
            "参数量": [
                f"{d_model * d_model:,}",
                f"{d_model * d_model:,}",
                f"{d_model * d_model:,}",
                f"{d_model * d_model:,}",
                f"{d_model * d_ff:,}",
                f"{d_ff * d_model:,}"
            ]
        }
        
        df_layer = pd.DataFrame(layer_breakdown)
        st.dataframe(df_layer, use_container_width=True)
        
        # 实际模型对比
        st.markdown("### 🎯 与实际模型对比")
        
        real_models = {
            "模型": ["GPT-2 Small", "GPT-2 Medium", "GPT-2 Large", "BERT-Base", "BERT-Large", "您的配置"],
            "层数": [12, 24, 36, 12, 24, n_layers],
            "d_model": [768, 1024, 1280, 768, 1024, d_model],
            "参数量 (M)": [117, 345, 774, 110, 340, total_params/1e6]
        }
        
        df_models = pd.DataFrame(real_models)
        
        # 高亮您的配置
        def highlight_yours(row):
            if row["模型"] == "您的配置":
                return ['background-color: #FFF3CD'] * len(row)
            return [''] * len(row)
        
        st.dataframe(df_models.style.apply(highlight_yours, axis=1), 
                    use_container_width=True)
        
        st.info(r"""
        **快速估算公式**:
        
        对于标准Transformer（d_ff = 4 × d_model）：
        
        $$\text{Total Params} \approx V \times d + L \times 12 \times d^2$$
        
        其中：
        - V: 词表大小
        - d: d_model
        - L: 层数
        
        **记忆技巧**：每层约 12d² 参数（4d² attention + 8d² FFN）
        """)
    
    @staticmethod
    def _render_cnn_calculator():
        """CNN参数计算器"""
        st.markdown("### 🖼️ CNN参数计算器")
        
        st.markdown(r"""
        **卷积层参数公式**：
        """)
        
        st.latex(r"""
        \text{Conv Params} = (K_h \times K_w \times C_{in} + 1) \times C_{out}
        """)
        
        st.markdown(r"""
        其中：
        - $K_h, K_w$: 卷积核高度和宽度
        - $C_{in}$: 输入通道数
        - $C_{out}$: 输出通道数
        - +1: bias项
        """)
        
        with st.sidebar:
            st.markdown("#### 网络配置")
            network_type = st.selectbox(
                "网络类型",
                ["自定义", "LeNet-5", "AlexNet", "VGG-16", "ResNet-50"]
            )
            
            if network_type == "自定义":
                n_conv_layers = st.slider("卷积层数量", 1, 10, 3)
        
        if network_type == "自定义":
            # 自定义配置
            st.markdown("#### 🔧 自定义CNN配置")
            
            layers_config = []
            total_params = 0
            
            for i in range(n_conv_layers):
                with st.expander(f"第 {i+1} 层"):
                    col1, col2 = st.columns(2)
                    with col1:
                        c_in = st.number_input(f"输入通道", 1, 2048, 
                                              3 if i == 0 else 64, 
                                              key=f"cin_{i}")
                        k_h = st.number_input(f"卷积核高度", 1, 11, 3, key=f"kh_{i}")
                    with col2:
                        c_out = st.number_input(f"输出通道", 1, 2048, 64, key=f"cout_{i}")
                        k_w = st.number_input(f"卷积核宽度", 1, 11, 3, key=f"kw_{i}")
                    
                    include_bias = st.checkbox(f"包含bias", value=True, key=f"bias_{i}")
                    
                    # 计算参数
                    params = k_h * k_w * c_in * c_out
                    if include_bias:
                        params += c_out
                    
                    layers_config.append({
                        "层": f"Conv{i+1}",
                        "输入通道": c_in,
                        "输出通道": c_out,
                        "卷积核": f"{k_h}×{k_w}",
                        "参数量": f"{params:,}",
                        "参数量(数值)": params
                    })
                    
                    total_params += params
            
            # 显示结果
            st.markdown("### 📊 参数统计")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总参数量", f"{total_params/1e6:.2f}M")
            with col2:
                st.metric("卷积层数", n_conv_layers)
            
            df = pd.DataFrame(layers_config)
            st.dataframe(df.drop('参数量(数值)', axis=1), use_container_width=True)
            
            # 可视化
            fig = go.Figure(data=[
                go.Bar(
                    x=[layer["层"] for layer in layers_config],
                    y=[layer["参数量(数值)"] for layer in layers_config],
                    text=[f"{layer['参数量(数值)']/1e3:.1f}K" for layer in layers_config],
                    textposition='outside',
                    marker_color='steelblue'
                )
            ])
            
            fig.update_layout(
                title="各层参数量分布",
                xaxis_title="层",
                yaxis_title="参数量",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # 预设网络
            preset_configs = {
                "LeNet-5": [
                    {"层": "Conv1", "输入": 1, "输出": 6, "卷积核": "5×5", "参数": 5*5*1*6 + 6},
                    {"层": "Conv2", "输入": 6, "输出": 16, "卷积核": "5×5", "参数": 5*5*6*16 + 16},
                    {"层": "FC1", "输入": 400, "输出": 120, "卷积核": "-", "参数": 400*120 + 120},
                    {"层": "FC2", "输入": 120, "输出": 84, "卷积核": "-", "参数": 120*84 + 84},
                    {"层": "FC3", "输入": 84, "输出": 10, "卷积核": "-", "参数": 84*10 + 10},
                ],
                "AlexNet": [
                    {"层": "Conv1", "输入": 3, "输出": 96, "卷积核": "11×11", "参数": 11*11*3*96 + 96},
                    {"层": "Conv2", "输入": 96, "输出": 256, "卷积核": "5×5", "参数": 5*5*96*256 + 256},
                    {"层": "Conv3", "输入": 256, "输出": 384, "卷积核": "3×3", "参数": 3*3*256*384 + 384},
                    {"层": "Conv4", "输入": 384, "输出": 384, "卷积核": "3×3", "参数": 3*3*384*384 + 384},
                    {"层": "Conv5", "输入": 384, "输出": 256, "卷积核": "3×3", "参数": 3*3*384*256 + 256},
                    {"层": "FC1", "输入": 9216, "输出": 4096, "卷积核": "-", "参数": 9216*4096 + 4096},
                    {"层": "FC2", "输入": 4096, "输出": 4096, "卷积核": "-", "参数": 4096*4096 + 4096},
                    {"层": "FC3", "输入": 4096, "输出": 1000, "卷积核": "-", "参数": 4096*1000 + 1000},
                ],
                "VGG-16": [
                    {"层": "Conv1-1", "输入": 3, "输出": 64, "卷积核": "3×3", "参数": 3*3*3*64 + 64},
                    {"层": "Conv1-2", "输入": 64, "输出": 64, "卷积核": "3×3", "参数": 3*3*64*64 + 64},
                    {"层": "Conv2-1", "输入": 64, "输出": 128, "卷积核": "3×3", "参数": 3*3*64*128 + 128},
                    {"层": "Conv2-2", "输入": 128, "输出": 128, "卷积核": "3×3", "参数": 3*3*128*128 + 128},
                    {"层": "...", "输入": "...", "输出": "...", "卷积核": "...", "参数": 0},
                    {"层": "FC", "输入": 25088, "输出": 4096, "卷积核": "-", "参数": 25088*4096 + 4096},
                ],
                "ResNet-50": [
                    {"层": "Conv1", "输入": 3, "输出": 64, "卷积核": "7×7", "参数": 7*7*3*64 + 64},
                    {"层": "Bottleneck×3", "输入": 64, "输出": 256, "卷积核": "mix", "参数": 3*(64*64 + 64*64*3*3 + 64*256)*3},
                    {"层": "Bottleneck×4", "输入": 256, "输出": 512, "卷积核": "mix", "参数": 4*(128*256 + 128*128*3*3 + 128*512)},
                    {"层": "...", "输入": "...", "输出": "...", "卷积核": "...", "参数": 0},
                    {"层": "FC", "输入": 2048, "输出": 1000, "卷积核": "-", "参数": 2048*1000 + 1000},
                ]
            }
            
            config = preset_configs[network_type]
            total = sum(layer["参数"] for layer in config)
            
            st.markdown(f"### 📊 {network_type} 参数统计")
            st.metric("总参数量", f"{total/1e6:.1f}M")
            
            df = pd.DataFrame([
                {
                    "层": layer["层"],
                    "输入通道": layer["输入"],
                    "输出通道": layer["输出"],
                    "卷积核": layer["卷积核"],
                    "参数量": f"{layer['参数']:,}" if isinstance(layer['参数'], int) else "..."
                }
                for layer in config
            ])
            
            st.dataframe(df, use_container_width=True)
        
        st.success(r"""
        **CNN参数的特点**:
        
        1. **卷积层参数少**：权重共享，参数量与输入大小无关
        2. **全连接层参数多**：通常占90%以上参数
        3. **优化策略**：
           - 用Global Average Pooling替代FC
           - 深度可分离卷积（Depthwise Separable）
           - 1×1卷积降维
        
        **经验法则**：
        - 3×3卷积：9 × C_in × C_out
        - 1×1卷积：C_in × C_out（降维利器）
        - FC层：Input_dim × Output_dim（大头所在）
        """)

    
    @staticmethod
    def _render_memory_calculator():
        """显存占用估算"""
        st.markdown("### 💾 显存占用估算器")
        
        st.markdown(r"""
        **训练显存组成**：
        
        $$\text{Total Memory} = \text{Model} + \text{Optimizer} + \text{Gradients} + \text{Activations}$$
        """)
        
        with st.sidebar:
            st.markdown("#### 模型配置")
            param_count = st.number_input("参数量 (M)", 1, 200000, 1000, 100)
            batch_size = st.number_input("Batch Size", 1, 256, 8, 1)
            seq_length = st.number_input("序列长度", 128, 8192, 512, 128)
            
            st.markdown("#### 训练配置")
            optimizer = st.selectbox("优化器", ["Adam", "SGD", "AdamW"])
            precision = st.selectbox("精度", ["FP32", "FP16", "BF16", "INT8"])
            gradient_checkpointing = st.checkbox("梯度检查点", value=False)
        
        # 精度对应的字节数
        precision_bytes = {
            "FP32": 4,
            "FP16": 2,
            "BF16": 2,
            "INT8": 1
        }
        
        bytes_per_param = precision_bytes[precision]
        param_count_actual = param_count * 1e6
        
        # 1. 模型参数
        model_memory = param_count_actual * bytes_per_param
        
        # 2. 梯度
        gradient_memory = param_count_actual * bytes_per_param
        
        # 3. 优化器状态
        if optimizer == "SGD":
            optimizer_memory = 0  # SGD无额外状态
        elif optimizer in ["Adam", "AdamW"]:
            # Adam需要两个状态：momentum和variance，都是FP32
            optimizer_memory = param_count_actual * 4 * 2
        
        # 4. 激活值（估算）
        # 对于Transformer，激活值约为：batch_size × seq_length × hidden_dim × n_layers × 常数
        # 这里简化估算
        activation_per_token = param_count_actual / 100  # 粗略估计
        activation_memory = batch_size * seq_length * activation_per_token * bytes_per_param
        
        if gradient_checkpointing:
            activation_memory *= 0.3  # 梯度检查点可节约约70%激活值内存
        
        # 总显存
        total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
        
        # 显示结果
        st.markdown("### 📊 显存分解")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总显存", f"{total_memory/1e9:.2f} GB")
        with col2:
            st.metric("模型参数", f"{model_memory/1e9:.2f} GB")
        with col3:
            st.metric("优化器", f"{optimizer_memory/1e9:.2f} GB")
        with col4:
            st.metric("激活值", f"{activation_memory/1e9:.2f} GB")
        
        # 详细分解表格
        breakdown_data = {
            "组件": ["模型参数", "梯度", "优化器状态", "激活值", "总计"],
            "大小 (GB)": [
                f"{model_memory/1e9:.2f}",
                f"{gradient_memory/1e9:.2f}",
                f"{optimizer_memory/1e9:.2f}",
                f"{activation_memory/1e9:.2f}",
                f"{total_memory/1e9:.2f}"
            ],
            "占比": [
                f"{model_memory/total_memory*100:.1f}%",
                f"{gradient_memory/total_memory*100:.1f}%",
                f"{optimizer_memory/total_memory*100:.1f}%",
                f"{activation_memory/total_memory*100:.1f}%",
                "100.0%"
            ]
        }
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("显存占比", "各组件大小")
        )
        
        # 饼图
        labels = ["模型参数", "梯度", "优化器", "激活值"]
        values = [model_memory, gradient_memory, optimizer_memory, activation_memory]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # 柱状图
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[v/1e9 for v in values],
                marker_color=colors,
                text=[f"{v/1e9:.2f}GB" for v in values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(title_text="显存 (GB)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # GPU建议
        st.markdown("### 🎯 GPU选型建议")
        
        gpu_options = {
            "GPU": ["RTX 3090", "RTX 4090", "A100 40GB", "A100 80GB", "H100 80GB"],
            "显存": ["24 GB", "24 GB", "40 GB", "80 GB", "80 GB"],
            "是否足够": [
                "✅" if total_memory/1e9 < 24 else "❌",
                "✅" if total_memory/1e9 < 24 else "❌",
                "✅" if total_memory/1e9 < 40 else "❌",
                "✅" if total_memory/1e9 < 80 else "❌",
                "✅" if total_memory/1e9 < 80 else "❌"
            ],
            "建议": [
                "消费级，性价比高" if total_memory/1e9 < 20 else "显存不足",
                "最新消费级" if total_memory/1e9 < 20 else "显存不足",
                "企业级，稳定" if total_memory/1e9 < 35 else "显存不足",
                "大模型训练" if total_memory/1e9 < 75 else "显存不足",
                "最强算力" if total_memory/1e9 < 75 else "考虑模型并行"
            ]
        }
        
        df_gpu = pd.DataFrame(gpu_options)
        st.dataframe(df_gpu, use_container_width=True)
        
        # 优化建议
        st.markdown("### 💡 显存优化策略")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 减少模型参数显存")
            savings_fp16 = (model_memory - param_count_actual * 2) / 1e9
            savings_int8 = (model_memory - param_count_actual * 1) / 1e9
            
            st.write(f"- **FP16/BF16**: 节省 {savings_fp16:.2f} GB")
            st.write(f"- **INT8量化**: 节省 {savings_int8:.2f} GB")
            st.write(f"- **LoRA微调**: 只训练少量参数")
        
        with col2:
            st.markdown("#### 减少激活值显存")
            savings_checkpoint = activation_memory * 0.7 / 1e9
            savings_batch = activation_memory * 0.5 / 1e9
            
            st.write(f"- **梯度检查点**: 节省 {savings_checkpoint:.2f} GB")
            st.write(f"- **减小batch_size**: 节省 {savings_batch:.2f} GB")
            st.write(f"- **梯度累积**: 保持有效batch_size")
        
        st.success(r"""
        **显存计算公式 (FP32, Adam)**:
        
        $$\text{Memory} \approx \text{Params} \times (4 + 4 + 8) = 16 \times \text{Params}$$
        
        - 4 bytes: 模型参数
        - 4 bytes: 梯度
        - 8 bytes: Adam状态 (m和v)
        
        **降低到FP16 + 梯度检查点**:
        
        $$\text{Memory} \approx \text{Params} \times (2 + 2 + 8) = 12 \times \text{Params}$$
        
        **记忆技巧**: FP32+Adam ≈ 16×参数量，FP16 ≈ 一半
        """)
    
    @staticmethod
    def _render_architecture_comparison():
        """架构对比（GPT vs BERT）"""
        st.markdown("### ⚔️ 架构对比：GPT vs BERT")
        
        st.markdown("""
        **核心区别**：
        - **GPT**: 单向语言模型（Causal/Decoder-only）
        - **BERT**: 双向编码器（Masked/Encoder-only）
        """)
        
        with st.sidebar:
            st.markdown("#### 模型规模")
            d_model = st.slider("d_model", 256, 2048, 768, 64)
            n_layers = st.slider("层数", 6, 48, 12, 1)
            vocab_size = st.slider("词表大小", 10000, 100000, 50000, 1000)
        
        # GPT计算
        gpt_params = vocab_size * d_model + n_layers * 12 * d_model**2 + d_model * vocab_size
        
        # BERT计算（额外的segment embedding和position embedding）
        bert_params = (vocab_size + 512 + 2) * d_model + n_layers * 12 * d_model**2 + d_model * vocab_size
        
        st.markdown("### 📊 参数量对比")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("GPT", f"{gpt_params/1e6:.1f}M")
        with col2:
            st.metric("BERT", f"{bert_params/1e6:.1f}M")
        with col3:
            diff_pct = (bert_params - gpt_params) / gpt_params * 100
            st.metric("差异", f"{diff_pct:.1f}%")
        
        # 详细对比表
        comparison_data = {
            "特性": [
                "注意力类型",
                "训练目标",
                "Embedding",
                "位置编码",
                "预训练任务",
                "下游任务",
                "参数量",
                "典型应用"
            ],
            "GPT (Decoder)": [
                "Causal (单向)",
                "下一个token预测",
                "Token + Position",
                "学习式",
                "Language Modeling",
                "生成任务",
                f"{gpt_params/1e6:.1f}M",
                "文本生成、对话"
            ],
            "BERT (Encoder)": [
                "Bidirectional (双向)",
                "MLM + NSP",
                "Token + Segment + Position",
                "学习式",
                "Masked LM + Next Sentence",
                "理解任务",
                f"{bert_params/1e6:.1f}M",
                "分类、NER、QA"
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # 可视化attention模式
        st.markdown("### 👁️ Attention模式对比")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("GPT: Causal Attention", "BERT: Full Attention"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # GPT的causal mask
        seq_len = 8
        gpt_mask = np.tril(np.ones((seq_len, seq_len)))
        
        # BERT的full attention
        bert_mask = np.ones((seq_len, seq_len))
        
        fig.add_trace(
            go.Heatmap(
                z=gpt_mask,
                colorscale=[[0, 'white'], [1, 'blue']],
                showscale=False,
                text=[[f"T{i+1}" for i in range(seq_len)] for _ in range(seq_len)],
                texttemplate="%{text}",
                hovertemplate='Query: %{y}<br>Key: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=bert_mask,
                colorscale=[[0, 'white'], [1, 'green']],
                showscale=False,
                text=[[f"T{i+1}" for i in range(seq_len)] for _ in range(seq_len)],
                texttemplate="%{text}",
                hovertemplate='Query: %{y}<br>Key: %{x}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Key Position", row=1, col=1)
        fig.update_yaxes(title_text="Query Position", row=1, col=1)
        fig.update_xaxes(title_text="Key Position", row=1, col=2)
        fig.update_yaxes(title_text="Query Position", row=1, col=2)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Attention模式含义**：
        
        - **GPT (Causal)**: 蓝色表示可以attend，白色表示mask掉
          - 只能看到当前和之前的token
          - 保证生成的自回归性质
        
        - **BERT (Full)**: 全绿，可以看到所有token
          - 双向上下文信息
          - 更适合理解任务
        """)
        
        st.success(r"""
        **选择建议**:
        
        | 任务类型 | 推荐架构 | 原因 |
        |---------|---------|------|
        | 文本生成 | GPT | 自回归生成 |
        | 文本分类 | BERT | 双向理解 |
        | 问答系统 | BERT | 需要全局信息 |
        | 对话系统 | GPT | 生成连贯回复 |
        | 实体识别 | BERT | 需要上下文 |
        | 代码生成 | GPT | 自回归生成 |
        
        **现代趋势**: Decoder-only (GPT风格) 统一生成和理解任务
        """)

    
    @staticmethod
    def _render_memory_anatomy():
        """训练显存解剖"""
        st.markdown("### 🔬 训练显存解剖学")
        
        st.markdown(r"""
        **深入剖析**: 训练一个7B模型到底需要多少显存？
        
        以 **LLaMA 7B** 为例：
        """)
        
        with st.sidebar:
            st.markdown("#### 模型参数")
            model_size_b = st.selectbox(
                "模型大小",
                ["7B (LLaMA)", "13B (LLaMA)", "70B (LLaMA)", "自定义"],
                index=0
            )
            
            if model_size_b == "自定义":
                param_count = st.number_input("参数量 (B)", 1, 200, 7)
            else:
                param_count = int(model_size_b.split('B')[0])
            
            st.markdown("#### 训练配置")
            training_mode = st.selectbox(
                "训练模式",
                ["全参数微调", "LoRA", "QLoRA", "仅推理"]
            )
            precision = st.selectbox("精度", ["FP32", "FP16", "BF16"], index=1)
            batch_size = st.number_input("Batch Size", 1, 64, 4, 1)
            seq_length = st.number_input("序列长度", 512, 4096, 2048, 512)
        
        param_count_actual = param_count * 1e9
        
        # 精度字节数
        bytes_map = {"FP32": 4, "FP16": 2, "BF16": 2}
        bytes_per_param = bytes_map[precision]
        
        # 1. 模型参数
        model_weights = param_count_actual * bytes_per_param / 1e9
        
        # 2. 梯度
        if training_mode == "仅推理":
            gradients = 0
            optimizer_states = 0
        elif training_mode == "LoRA":
            # LoRA只训练很少的参数（假设1%）
            trainable_params = param_count_actual * 0.01
            gradients = trainable_params * bytes_per_param / 1e9
            optimizer_states = trainable_params * 8 / 1e9  # Adam状态
        elif training_mode == "QLoRA":
            # QLoRA + 4bit量化
            model_weights = param_count_actual * 0.5 / 1e9  # 4bit ≈ 0.5 bytes
            trainable_params = param_count_actual * 0.01
            gradients = trainable_params * bytes_per_param / 1e9
            optimizer_states = trainable_params * 8 / 1e9
        else:  # 全参数
            gradients = param_count_actual * bytes_per_param / 1e9
            optimizer_states = param_count_actual * 8 / 1e9  # Adam: m + v
        
        # 3. 激活值（与batch size成正比）
        # 粗略估计：每个token约需 hidden_dim * n_layers * 常数 的激活值
        hidden_dim = 4096  # LLaMA 7B的隐藏维度
        n_layers = 32  # LLaMA 7B的层数
        activation_per_token = hidden_dim * n_layers * 20 * bytes_per_param / 1e9
        activations = batch_size * seq_length * activation_per_token
        
        # KV Cache (推理时)
        if training_mode == "仅推理":
            kv_cache = 2 * n_layers * hidden_dim * seq_length * bytes_per_param * batch_size / 1e9
        else:
            kv_cache = 0
        
        # 总显存
        total_memory = model_weights + gradients + optimizer_states + activations + kv_cache
        
        # 显示结果
        st.markdown("### 📊 显存详细分解")
        
        cols = st.columns(5)
        with cols[0]:
            st.metric("总计", f"{total_memory:.1f} GB")
        with cols[1]:
            st.metric("模型", f"{model_weights:.1f} GB")
        with cols[2]:
            st.metric("梯度", f"{gradients:.1f} GB")
        with cols[3]:
            st.metric("优化器", f"{optimizer_states:.1f} GB")
        with cols[4]:
            st.metric("激活值", f"{activations:.1f} GB")
        
        # 详细表格
        breakdown = []
        
        if model_weights > 0:
            breakdown.append(["模型权重", f"{model_weights:.2f}", f"{model_weights/total_memory*100:.1f}%", 
                            f"{param_count}B × {bytes_per_param} bytes"])
        if gradients > 0:
            breakdown.append(["梯度", f"{gradients:.2f}", f"{gradients/total_memory*100:.1f}%",
                            "与模型权重相同"])
        if optimizer_states > 0:
            breakdown.append(["优化器状态", f"{optimizer_states:.2f}", f"{optimizer_states/total_memory*100:.1f}%",
                            "Adam: momentum + variance"])
        if activations > 0:
            breakdown.append(["激活值", f"{activations:.2f}", f"{activations/total_memory*100:.1f}%",
                            f"Batch {batch_size} × Seq {seq_length}"])
        if kv_cache > 0:
            breakdown.append(["KV Cache", f"{kv_cache:.2f}", f"{kv_cache/total_memory*100:.1f}%",
                            "推理时缓存Key/Value"])
        
        df = pd.DataFrame(breakdown, columns=["组件", "大小 (GB)", "占比", "说明"])
        st.dataframe(df, use_container_width=True)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("显存占比", "各组件大小 (GB)")
        )
        
        labels = [item[0] for item in breakdown]
        values = [float(item[1]) for item in breakdown]
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.1f}GB" for v in values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(title_text="显存 (GB)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 实际案例对比
        st.markdown("### 🎯 实际案例：不同训练方式对比")
        
        cases = []
        
        # 全参数FP32
        full_fp32 = param_count * (4 + 4 + 8) + activations
        cases.append(["全参数 FP32 + Adam", f"{full_fp32:.1f}", "❌ 太大", "4×16GB A100"])
        
        # 全参数FP16
        full_fp16 = param_count * (2 + 2 + 8) + activations
        cases.append(["全参数 FP16 + Adam", f"{full_fp16:.1f}", 
                     "✅" if full_fp16 < 40 else "⚠️", "2×40GB A100"])
        
        # LoRA
        lora_mem = param_count * 2 + param_count * 0.01 * (2 + 8) + activations
        cases.append(["LoRA (1% 参数)", f"{lora_mem:.1f}", "✅", "1×24GB GPU"])
        
        # QLoRA
        qlora_mem = param_count * 0.5 + param_count * 0.01 * (2 + 8) + activations
        cases.append(["QLoRA (4bit + 1%)", f"{qlora_mem:.1f}", "✅", "1×16GB GPU"])
        
        # 仅推理
        inference_mem = param_count * 2 + kv_cache
        cases.append(["仅推理 FP16", f"{inference_mem:.1f}", "✅", "消费级GPU"])
        
        df_cases = pd.DataFrame(cases, columns=["模式", "显存 (GB)", "可行性", "推荐硬件"])
        st.dataframe(df_cases, use_container_width=True)
        
        st.success(f"""
        **{param_count}B 模型训练建议**:
        
        1. **全参数微调** (Full Fine-tuning):
           - 显存需求: ~{full_fp16:.0f} GB
           - 需要多卡并行 (至少2×A100)
           - 适合大厂和研究机构
        
        2. **LoRA微调**:
           - 显存需求: ~{lora_mem:.0f} GB
           - 单卡24GB可搞定
           - 效果接近全参数，推荐！
        
        3. **QLoRA微调**:
           - 显存需求: ~{qlora_mem:.0f} GB
           - 消费级GPU可用
           - 量化损失很小
        
        4. **仅推理**:
           - 显存需求: ~{inference_mem:.0f} GB
           - 笔记本/消费级GPU
           - 配合量化更省
        """)
    
    @staticmethod
    def _render_precision_quantization():
        """混合精度与量化"""
        st.markdown("### ⚡ 混合精度与量化")
        
        st.markdown("""
        **核心思想**: 用更少的bit表示数字，节省显存和计算
        """)
        
        # 精度对比
        st.markdown("### 📊 不同精度对比")
        
        precision_data = {
            "精度": ["FP32", "FP16", "BF16", "INT8", "INT4"],
            "字节数": [4, 2, 2, 1, 0.5],
            "数值范围": [
                "±3.4e38",
                "±65504",
                "±3.4e38",
                "0-255",
                "0-15"
            ],
            "精度": [
                "7位有效数字",
                "3位有效数字",
                "7位有效数字(范围大)",
                "整数",
                "整数"
            ],
            "用途": [
                "传统训练",
                "混合精度训练",
                "稳定的混合精度",
                "推理加速",
                "极限压缩"
            ],
            "相对速度": ["1×", "2×", "2×", "4×", "8×"]
        }
        
        df_precision = pd.DataFrame(precision_data)
        st.dataframe(df_precision, use_container_width=True)
        
        # 可视化显存节省
        st.markdown("### 💾 显存节省对比")
        
        with st.sidebar:
            st.markdown("#### 模型配置")
            model_params = st.slider("模型参数 (B)", 1, 100, 7)
        
        params = model_params * 1e9
        
        mem_fp32 = params * 4 / 1e9
        mem_fp16 = params * 2 / 1e9
        mem_int8 = params * 1 / 1e9
        mem_int4 = params * 0.5 / 1e9
        
        fig = go.Figure()
        
        precisions = ["FP32", "FP16/BF16", "INT8", "INT4"]
        memories = [mem_fp32, mem_fp16, mem_int8, mem_int4]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        fig.add_trace(go.Bar(
            x=precisions,
            y=memories,
            text=[f"{m:.1f}GB<br>({m/mem_fp32*100:.0f}%)" for m in memories],
            textposition='outside',
            marker_color=colors
        ))
        
        fig.update_layout(
            title=f"{model_params}B 模型在不同精度下的显存占用",
            xaxis_title="精度",
            yaxis_title="显存 (GB)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 量化方法对比
        st.markdown("### 🔧 量化方法")
        
        quant_methods = {
            "方法": [
                "混合精度 (AMP)",
                "训练后量化 (PTQ)",
                "量化感知训练 (QAT)",
                "QLoRA",
                "GPTQ"
            ],
            "精度": [
                "FP16/BF16",
                "INT8",
                "INT8",
                "4bit NF4",
                "3-4bit"
            ],
            "准确度损失": [
                "< 0.1%",
                "1-2%",
                "< 0.5%",
                "< 1%",
                "1-3%"
            ],
            "实现难度": [
                "简单",
                "简单",
                "中等",
                "中等",
                "复杂"
            ],
            "适用场景": [
                "训练加速",
                "推理加速",
                "高精度推理",
                "大模型微调",
                "极限压缩"
            ]
        }
        
        df_quant = pd.DataFrame(quant_methods)
        st.dataframe(df_quant, use_container_width=True)
        
        # 实战建议
        st.markdown("### 💡 实战建议")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 训练阶段")
            st.info("""
            1. **自动混合精度 (AMP)**
               ```python
               from torch.cuda.amp import autocast
               with autocast():
                   output = model(input)
               ```
               - 开箱即用，2倍加速
               - 几乎无精度损失
            
            2. **梯度累积**
               - 减小batch_size
               - 累积多步后更新
               - 节省激活值显存
            
            3. **梯度检查点**
               - 重计算代替存储
               - 节省70%激活值显存
               - 速度降低20-30%
            """)
        
        with col2:
            st.markdown("#### 推理阶段")
            st.info("""
            1. **INT8量化**
               ```python
               model = torch.quantization.quantize_dynamic(
                   model, {torch.nn.Linear}, dtype=torch.qint8
               )
               ```
               - 4倍显存节省
               - 2-4倍速度提升
            
            2. **模型蒸馏**
               - 用小模型模仿大模型
               - 保留90%性能
               - 10倍参数减少
            
            3. **剪枝**
               - 去除不重要的权重
               - 结构化剪枝更快
               - 配合量化效果更好
            """)
        
        st.success(f"""
        **{model_params}B 模型优化路径**:
        
        1. **开发阶段**: FP32 (精度最高)
        2. **训练阶段**: FP16/BF16 AMP (2倍加速)
        3. **微调阶段**: LoRA/QLoRA (显存节省)
        4. **部署阶段**: INT8 (4倍显存节省)
        5. **边缘设备**: INT4 + 剪枝 (极限优化)
        
        **记住**: 精度降低是节省资源的主要手段！
        """)


# 注册到__all__
__all__ = ['InteractiveDimensionsParameters']

        # 添加交互式测验
