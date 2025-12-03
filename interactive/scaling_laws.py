"""
Scaling Laws交互式可视化
严格按照 AppxB_ScalingLaws.md 中的理论实现

核心内容：
1. 幂律现象
2. 计算预算优化
3. Chinchilla最优前沿
4. 训练最优 vs 推理最优
5. Llama 3的策略
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

class InteractiveScalingLaws:
    """交互式Scaling Laws可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("📈 Scaling Laws：预知未来的数学")
        
        st.markdown(r"""
        **核心发现**: 深度学习不是炼金术，而是遵循严格的**幂律**
        
        **Scaling Law**:
        """)
        
        st.latex(r"""
        L(X) = E + \frac{A}{X^\alpha}
        """)
        
        st.markdown(r"""
        **意义**:
        - $X$: 模型参数、数据量、计算量
        - $L$: 损失（测试集交叉熵）
        - $\alpha$: 幂律指数（通常0.3-0.5）
        - $E$: 不可约损失（语言熵的下界）
        
        **惊人结论**: 只要持续增加资源，性能就会可预测地提升！
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "幂律现象",
                    "Chinchilla最优前沿",
                    "计算预算优化",
                    "训练最优 vs 推理最优",
                    "Llama 3策略"
                ]
            )
        
        if demo_type == "幂律现象":
            InteractiveScalingLaws._render_power_law()
        elif demo_type == "Chinchilla最优前沿":
            InteractiveScalingLaws._render_chinchilla()
        elif demo_type == "计算预算优化":
            InteractiveScalingLaws._render_compute_optimal()
        elif demo_type == "训练最优 vs 推理最优":
            InteractiveScalingLaws._render_train_vs_inference()
        elif demo_type == "Llama 3策略":
            InteractiveScalingLaws._render_llama3()
    

        # 添加交互式测验
        quiz_system = QuizSystem("scaling_laws")
        quizzes = QuizTemplates.get_scaling_laws_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_power_law():
        """幂律现象可视化"""
        st.markdown("### 📐 幂律现象：Loss与资源的关系")
        
        st.markdown(r"""
        **OpenAI & DeepMind的发现**: 在双对数坐标系下，Loss呈线性！
        """)
        
        st.latex(r"""
        \log(L - E) \approx \log A - \alpha \log X
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            resource_type = st.selectbox("资源类型", ["模型参数N", "数据量D", "计算量C"])
            alpha = st.slider("幂律指数 α", 0.1, 0.8, 0.4, 0.05)
            E = st.slider("不可约损失 E", 1.0, 2.0, 1.69, 0.01)
            A = st.slider("缩放系数 A", 1.0, 100.0, 10.0, 1.0)
        
        # 生成数据
        if resource_type == "模型参数N":
            X = np.logspace(6, 12, 100)  # 1M到1T参数
            x_label = "模型参数 (N)"
            x_unit = "参数"
        elif resource_type == "数据量D":
            X = np.logspace(9, 13, 100)  # 1B到10T tokens
            x_label = "数据量 (D)"
            x_unit = "tokens"
        else:
            X = np.logspace(18, 24, 100)  # 1e18到1e24 FLOPs
            x_label = "计算量 (C)"
            x_unit = "FLOPs"
        
        # 计算Loss
        L = E + A / (X ** alpha)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "线性坐标：幂律曲线",
                "双对数坐标：线性关系"
            )
        )
        
        # 线性坐标
        fig.add_trace(
            go.Scatter(
                x=X,
                y=L,
                mode='lines',
                name='Loss',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # 不可约损失基线
        fig.add_hline(y=E, line_dash="dash", line_color="red",
                     annotation_text=f"不可约损失 E={E}",
                     row=1, col=1)
        
        # 双对数坐标
        fig.add_trace(
            go.Scatter(
                x=X,
                y=L - E,
                mode='lines',
                name='Loss - E',
                line=dict(color='green', width=3)
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text=x_label, type="linear", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text=x_label, type="log", row=1, col=2)
        fig.update_yaxes(title_text="Loss - E", type="log", row=1, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text=f"幂律现象: L = {E} + {A}/X^{alpha}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 幂律分析")
        
        # 选几个关键点
        points = [1e6, 1e9, 1e12] if resource_type == "模型参数N" else \
                 [1e9, 1e11, 1e13] if resource_type == "数据量D" else \
                 [1e18, 1e21, 1e24]
        
        cols = st.columns(len(points))
        
        for i, point in enumerate(points):
            loss_at_point = E + A / (point ** alpha)
            with cols[i]:
                if resource_type == "模型参数N":
                    label = f"{point/1e9:.1f}B" if point < 1e12 else f"{point/1e12:.1f}T"
                elif resource_type == "数据量D":
                    label = f"{point/1e9:.0f}B" if point < 1e12 else f"{point/1e12:.1f}T"
                else:
                    label = f"1e{int(np.log10(point))}"
                
                st.metric(label, f"{loss_at_point:.3f}")
        
        st.success(r"""
        **幂律的深层含义**:
        
        1. **可预测性**: 
           - 在小规模验证实验后，可以预测大规模性能
           - OpenAI用这个预测了GPT-3的性能
        
        2. **无饱和**: 
           - 没有性能天花板（除了$E$）
           - "Scaling is all you need"
        
        3. **资源效率**: 
           - $\alpha \approx 0.4$ 意味着10倍资源 → 约2.5倍性能提升
           - 边际收益递减，但永不为零
        
        4. **指导投资**: 
           - 知道需要多少GPU才能达到目标性能
           - 避免盲目scaling
        """)
    
    @staticmethod
    def _render_chinchilla():
        """Chinchilla最优前沿可视化"""
        st.markdown("### 🐭 Chinchilla最优前沿")
        
        st.markdown(r"""
        **核心问题**: 给定计算预算$C$，如何分配给参数$N$和数据$D$？
        
        **约束**: $C = 6ND$ (FLOPs)
        
        **目标**: 最小化联合损失
        """)
        
        st.latex(r"""
        L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
        """)
        
        st.markdown(r"""
        **拉格朗日求解**: (详见笔记推导)
        """)
        
        st.latex(r"""
        N_{opt} \propto C^{\frac{\beta}{\alpha + \beta}}, \quad 
        D_{opt} \propto C^{\frac{\alpha}{\alpha + \beta}}
        """)
        
        with st.sidebar:
            alpha = st.slider("α (模型指数)", 0.2, 0.6, 0.34, 0.02)
            beta = st.slider("β (数据指数)", 0.2, 0.6, 0.28, 0.02)
        
        # 计算预算范围
        C_range = np.logspace(20, 25, 100)  # 1e20到1e25 FLOPs
        
        # Chinchilla最优
        N_opt = C_range ** (beta / (alpha + beta))
        D_opt = C_range ** (alpha / (alpha + beta))
        
        # 其他策略（非最优）
        # GPT-3策略：参数更大，数据更少
        N_gpt3 = C_range ** 0.7
        D_gpt3 = C_range / (6 * N_gpt3)
        
        # Llama策略：参数更小，数据更多（推理优先）
        N_llama = C_range ** 0.4
        D_llama = C_range / (6 * N_llama)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "参数量 vs 计算预算",
                "数据量 vs 计算预算",
                "N-D平面上的三种策略",
                "相对损失对比"
            )
        )
        
        # 参数量
        fig.add_trace(go.Scatter(x=C_range, y=N_opt, mode='lines',
                                name='Chinchilla', line=dict(color='green', width=3)),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=C_range, y=N_gpt3, mode='lines',
                                name='GPT-3风格', line=dict(color='blue', width=2, dash='dash')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=C_range, y=N_llama, mode='lines',
                                name='Llama风格', line=dict(color='red', width=2, dash='dot')),
                     row=1, col=1)
        
        # 数据量
        fig.add_trace(go.Scatter(x=C_range, y=D_opt, mode='lines',
                                name='Chinchilla', line=dict(color='green', width=3), showlegend=False),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=C_range, y=D_gpt3, mode='lines',
                                name='GPT-3风格', line=dict(color='blue', width=2, dash='dash'), showlegend=False),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=C_range, y=D_llama, mode='lines',
                                name='Llama风格', line=dict(color='red', width=2, dash='dot'), showlegend=False),
                     row=1, col=2)
        
        # N-D平面
        fig.add_trace(go.Scatter(x=N_opt, y=D_opt, mode='lines',
                                name='Chinchilla前沿', line=dict(color='green', width=3)),
                     row=2, col=1)
        
        # 等计算线（C = 6ND）
        for C_val in [1e22, 1e23, 1e24]:
            N_iso = np.logspace(9, 13, 50)
            D_iso = C_val / (6 * N_iso)
            fig.add_trace(go.Scatter(
                x=N_iso, y=D_iso,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name=f'C=1e{int(np.log10(C_val))}',
                showlegend=False
            ), row=2, col=1)
        
        # 相对损失（简化计算）
        A, B, E = 100, 100, 1.69
        Loss_opt = E + A/(N_opt**alpha) + B/(D_opt**beta)
        Loss_gpt3 = E + A/(N_gpt3**alpha) + B/(D_gpt3**beta)
        Loss_llama = E + A/(N_llama**alpha) + B/(D_llama**beta)
        
        relative_loss_gpt3 = (Loss_gpt3 - Loss_opt) / Loss_opt * 100
        relative_loss_llama = (Loss_llama - Loss_opt) / Loss_opt * 100
        
        fig.add_trace(go.Scatter(x=C_range, y=relative_loss_gpt3, mode='lines',
                                name='GPT-3多付出', line=dict(color='blue', width=2)),
                     row=2, col=2)
        fig.add_trace(go.Scatter(x=C_range, y=relative_loss_llama, mode='lines',
                                name='Llama多付出', line=dict(color='red', width=2)),
                     row=2, col=2)
        
        fig.update_xaxes(type="log", title_text="计算预算 C", row=1, col=1)
        fig.update_yaxes(type="log", title_text="参数 N", row=1, col=1)
        fig.update_xaxes(type="log", title_text="计算预算 C", row=1, col=2)
        fig.update_yaxes(type="log", title_text="数据 D (tokens)", row=1, col=2)
        fig.update_xaxes(type="log", title_text="参数 N", row=2, col=1)
        fig.update_yaxes(type="log", title_text="数据 D", row=2, col=1)
        fig.update_xaxes(type="log", title_text="计算预算 C", row=2, col=2)
        fig.update_yaxes(title_text="相对损失增加 (%)", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True,
                         title_text=f"Chinchilla最优前沿 (α={alpha}, β={beta})")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(r"""
        **Chinchilla的发现** (2022):
        
        1. **GPT-3被过度训练**: 
           - 175B参数，300B tokens
           - 按Chinchilla：应该是70B参数，1.4T tokens
        
        2. **最优比例**: $N : D \approx 1 : 20$
           - 每个参数应该看到约20个tokens
           - GPT-3只看到了1.7个tokens/参数
        
        3. **损失差异**: 
           - GPT-3在相同计算下损失比最优高约10%
           - 这10%很关键！
        
        **但2024年开始偏离Chinchilla**:
        - Llama 3: 8B参数，15T tokens (1875 tokens/参数！)
        - 为什么？→ 推理优先策略
        """)
    
    @staticmethod
    def _render_compute_optimal():
        """计算预算优化可视化"""
        st.markdown("### 💰 计算预算优化：如何花钱？")
        
        st.markdown(r"""
        **问题**: 有$1M预算（或1e24 FLOPs），怎么配置N和D？
        
        **三种策略**:
        """)
        
        import pandas as pd
        
        budget_strategies = pd.DataFrame({
            '策略': ['Chinchilla最优', 'GPT-3风格', 'Llama 3风格'],
            '目标': ['训练时最低Loss', '大模型能力', '推理成本最低'],
            '参数N': ['中等', '很大', '较小'],
            '数据D': ['中等', '较少', '很多'],
            '训练成本': ['最优', '最优', '最优'],
            '推理成本': ['中等', '高', '低'],
            '最终Loss': ['最低', '稍高', '稍高'],
            '适用场景': ['学术研究', '炫技Benchmark', '大规模部署']
        })
        
        st.dataframe(budget_strategies, use_container_width=True)
        
        st.markdown("### 📊 成本分析")
        
        # 模拟不同规模的成本
        scales = ['小型(1e22)', '中型(1e23)', '大型(1e24)', '超大(1e25)']
        compute_budget = [1e22, 1e23, 1e24, 1e25]
        
        # Chinchilla配置
        N_chin = [b**(0.28/(0.34+0.28)) for b in compute_budget]
        D_chin = [b**(0.34/(0.34+0.28)) for b in compute_budget]
        
        # 训练成本相同（都是预算C）
        train_cost = compute_budget
        
        # 推理成本 ∝ N（每次推理的FLOPs）
        inference_cost_chin = [n / 1e9 for n in N_chin]  # 归一化
        
        # GPT-3风格（参数大2倍）
        inference_cost_gpt3 = [2 * ic for ic in inference_cost_chin]
        
        # Llama风格（参数小2倍）
        inference_cost_llama = [0.5 * ic for ic in inference_cost_chin]
        
        # 总成本（假设推理100万次）
        n_inferences = 1e6
        total_cost_chin = [t + n_inferences * i for t, i in zip(train_cost, inference_cost_chin)]
        total_cost_gpt3 = [t + n_inferences * i for t, i in zip(train_cost, inference_cost_gpt3)]
        total_cost_llama = [t + n_inferences * i for t, i in zip(train_cost, inference_cost_llama)]
        
        # 可视化
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scales,
            y=total_cost_chin,
            name='Chinchilla',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            x=scales,
            y=total_cost_gpt3,
            name='GPT-3风格',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=scales,
            y=total_cost_llama,
            name='Llama风格',
            marker_color='red'
        ))
        
        fig.update_layout(
            title="总成本对比（训练 + 100万次推理）",
            xaxis_title="模型规模",
            yaxis_title="总FLOPs",
            yaxis_type="log",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **关键洞察**:
        
        - **训练一次，推理百万次**: 
          - 训练：固定成本
          - 推理：边际成本（每次调用都要付）
        
        - **Llama的权衡**:
          - 训练时Loss稍高（+5%）
          - 但推理成本低50%
          - 如果推理次数 > 100万，总成本更低！
        
        - **工业界选择**:
          - 研究：Chinchilla最优
          - 产品：Llama推理最优
          - 这是商业vs学术的根本分歧
        """)
    
    @staticmethod
    def _render_train_vs_inference():
        """训练最优vs推理最优可视化"""
        st.markdown("### ⚖️ 训练最优 vs 推理最优")
        
        st.markdown("""
        **范式转变**: 2022→2024
        
        | 时期 | 代表 | 策略 | 原因 |
        |------|------|------|------|
        | 2020-2022 | GPT-3, Chinchilla | 训练最优 | 追求SOTA |
        | 2023+ | Llama 2/3, Phi | **推理最优** | 大规模部署 |
        """)
        
        with st.sidebar:
            n_inference_calls = st.slider("推理调用次数（百万）", 0.1, 100.0, 10.0, 0.1)
        
        # 模型配置
        models = {
            'Chinchilla': {'N': 70e9, 'D': 1.4e12, 'train_compute': 5e23},
            'GPT-3': {'N': 175e9, 'D': 0.3e12, 'train_compute': 3.1e23},
            'Llama 3 8B': {'N': 8e9, 'D': 15e12, 'train_compute': 7.2e23}
        }
        
        # 计算成本
        results = []
        for name, config in models.items():
            train_cost = config['train_compute']
            inference_cost_per_call = 2 * config['N']  # 前向传播
            total_inference = inference_cost_per_call * n_inference_calls * 1e6
            total_cost = train_cost + total_inference
            
            results.append({
                '模型': name,
                '参数': config['N'],
                '训练Tokens': config['D'],
                '训练成本': train_cost,
                '推理成本': total_inference,
                '总成本': total_cost
            })
        
        # 归一化显示
        fig = go.Figure()
        
        for model_data in results:
            fig.add_trace(go.Bar(
                name=model_data['模型'],
                x=['训练', '推理', '总计'],
                y=[
                    model_data['训练成本'],
                    model_data['推理成本'],
                    model_data['总成本']
                ],
                text=[
                    f"{model_data['训练成本']:.2e}",
                    f"{model_data['推理成本']:.2e}",
                    f"{model_data['总成本']:.2e}"
                ],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=f"成本对比（{n_inference_calls}M次推理）",
            yaxis_title="FLOPs",
            yaxis_type="log",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Break-even分析
        st.markdown("### 🎯 Break-Even分析")
        
        # 计算break-even点
        # Llama vs Chinchilla
        llama_train = models['Llama 3 8B']['train_compute']
        chinchilla_train = models['Chinchilla']['train_compute']
        
        llama_inference = 2 * models['Llama 3 8B']['N']
        chinchilla_inference = 2 * models['Chinchilla']['N']
        
        # Break-even: llama_train + n * llama_inf = chin_train + n * chin_inf
        break_even = (llama_train - chinchilla_train) / (chinchilla_inference - llama_inference)
        
        st.success(f"""
        **Break-Even点**: {break_even/1e6:.1f}M 次推理
        
        - 如果推理次数 < {break_even/1e6:.1f}M: Chinchilla更优
        - 如果推理次数 > {break_even/1e6:.1f}M: Llama更优
        
        **工业界现实**:
        - ChatGPT: 每天数亿次推理
        - Claude: 每天数千万次
        - 开源模型: 推理次数远超训练
        
        **结论**: 推理最优是必然选择！
        """)
    
    @staticmethod
    def _render_llama3():
        """Llama 3策略解析"""
        st.markdown("### 🦙 Llama 3：推理时代的胜利")
        
        st.markdown("""
        **Llama 3的激进选择**:
        - 8B参数
        - 15T tokens训练
        - **1875 tokens/参数**（Chinchilla的94倍！）
        """)
        
        # 对比数据
        import pandas as pd
        
        comparison = pd.DataFrame({
            '模型': ['GPT-3', 'Chinchilla', 'Llama 2', 'Llama 3 8B'],
            '参数N': ['175B', '70B', '7B', '8B'],
            'Tokens D': ['0.3T', '1.4T', '2T', '15T'],
            'Tokens/参数': [1.7, 20, 286, 1875],
            '训练FLOPs': ['3.1e23', '5.0e23', '0.8e23', '7.2e23'],
            '推理FLOPs': ['350B/call', '140B/call', '14B/call', '16B/call'],
            '策略': ['大力出奇迹', 'Chinchilla最优', '推理优先', '极致推理优先']
        })
        
        st.dataframe(comparison, use_container_width=True)
        
        st.markdown("### 📈 Llama 3的优势")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**训练阶段**")
            st.warning("""
            ❌ 训练时间更长
            ❌ 数据需求巨大（15T tokens！）
            ❌ 训练Loss略高于Chinchilla
            
            但这些是**一次性成本**
            """)
        
        with col2:
            st.markdown("**推理阶段**")
            st.success("""
            ✅ 推理速度快（模型小）
            ✅ 内存占用少
            ✅ 可在消费级GPU运行
            ✅ API成本低
            
            这些是**持续收益**
            """)
        
        st.markdown("### 🎯 为什么Llama 3成功？")
        
        st.success("""
        **1. Over-training Works**:
        - Chinchilla说：每参数20 tokens最优
        - Llama 3: 每参数1875 tokens仍在提升！
        - 说明Scaling Laws在极端区域仍成立
        
        **2. 数据质量 > 数量**:
        - 15T tokens是精选的高质量数据
        - 不是简单堆砌，而是精心策划
        - 包括代码、数学、多语言
        
        **3. 推理民主化**:
        - 8B可在单GPU运行（RTX 4090）
        - 4-bit量化后仅需6GB显存
        - 人人都能部署自己的LLM
        
        **4. 商业模式**:
        - Meta免费提供（开源）
        - 推理成本转嫁给用户
        - 通过生态获利
        
        **结论**: 
        
        Llama 3不是违反Scaling Laws，而是在不同约束下的最优解：
        - Chinchilla: 训练预算约束
        - Llama 3: 推理成本约束
        
        都是数学最优，只是目标函数不同！
        """)
        
        st.info("""
        **未来趋势**:
        
        1. **持续Over-training**: 
           - Llama 4可能100T+ tokens
           - "训练永远不够"成为新共识
        
        2. **小模型复兴**:
           - 1B-10B模型的"蒸馏+over-training"
           - Phi-3, Gemma等跟进
        
        3. **混合架构**:
           - 大模型（思考） + 小模型（执行）
           - Mixture of Experts的回归
        
        **Scaling Laws没有死，只是进化了！**
        """)

        # 添加交互式测验
