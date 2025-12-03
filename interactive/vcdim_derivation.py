"""
VC维详细推导交互式可视化
严格按照 7.VCdimeDerivationProcess.md 中的理论实现

核心内容：
1. Hoeffding不等式 - 概率集中
2. 增长函数与Sauer-Shelah引理
3. VC泛化界完整推导
4. Radon定理 - VC维上界
5. 有效VC维
6. 理论局限性分析
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.special import comb
from scipy.stats import binom


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveVCDimDerivation:
    """交互式VC维详细推导可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("📐 VC维详细推导：从Hoeffding到泛化界")
        
        st.markdown(r"""
        **核心目标**: 严格推导为什么有限VC维能保证泛化
        
        **推导链条**:
        1. **Hoeffding不等式** → 单一假设的泛化
        2. **Union Bound** → 有限假设类的泛化
        3. **增长函数** → 无限假设类的"有效假设数"
        4. **Sauer-Shelah引理** → 增长函数的上界
        5. **VC泛化界** → 最终的理论保证
        
        **数学之美**: 从概率不等式到学习理论的完整逻辑链！
        """)
        
        # 添加导航链接
        st.info("""
        💡 **想先建立直觉？** → 查看 **VC维理论** 模块获取概念理解
        
        本模块适合：理论研究者、数学爱好者、想深入理解证明的学习者
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "Hoeffding不等式",
                    "增长函数与Sauer-Shelah",
                    "VC泛化界推导",
                    "Radon定理",
                    "有效VC维",
                    "理论局限性",
                    "完整推导流程图"
                ]
            )
        
        if demo_type == "Hoeffding不等式":
            InteractiveVCDimDerivation._render_hoeffding()
        elif demo_type == "增长函数与Sauer-Shelah":
            InteractiveVCDimDerivation._render_growth_function()
        elif demo_type == "VC泛化界推导":
            InteractiveVCDimDerivation._render_vc_bound()
        elif demo_type == "Radon定理":
            InteractiveVCDimDerivation._render_radon()
        elif demo_type == "有效VC维":
            InteractiveVCDimDerivation._render_effective_vcdim()
        elif demo_type == "理论局限性":
            InteractiveVCDimDerivation._render_limitations()
        elif demo_type == "完整推导流程图":
            InteractiveVCDimDerivation._render_derivation_flow()
    

        # 添加交互式测验
        quiz_system = QuizSystem("vcdim_derivation")
        quizzes = QuizTemplates.get_vcdim_derivation_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_hoeffding():
        """Hoeffding不等式可视化"""
        st.markdown("### 📊 Hoeffding不等式：概率集中现象")
        
        st.markdown(r"""
        **定理 (Hoeffding, 1963)**:
        
        设 $X_1, ..., X_N$ 独立同分布，取值在 $[0,1]$，则：
        """)
        
        st.latex(r"""
        P\left(|\bar{X} - \mathbb{E}[X]| > \epsilon\right) \leq 2e^{-2N\epsilon^2}
        """)
        
        st.markdown(r"""
        **物理直观**: 
        - 抛硬币：抛的次数越多，频率越接近概率
        - 样本均值以指数速度集中到期望
        - **这是统计学习的基石！**
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            N = st.slider("样本数量 N", 10, 1000, 100, 10)
            epsilon = st.slider("偏差 ε", 0.01, 0.5, 0.1, 0.01)
            true_mean = st.slider("真实期望 μ", 0.0, 1.0, 0.5, 0.05)
        
        # 计算Hoeffding界
        hoeffding_bound = 2 * np.exp(-2 * N * epsilon**2)
        
        # 蒙特卡洛模拟实际概率
        np.random.seed(42)
        n_simulations = 10000
        violations = 0
        
        sample_means = []
        for _ in range(n_simulations):
            # 生成伯努利随机变量
            samples = np.random.binomial(1, true_mean, N)
            sample_mean = np.mean(samples)
            sample_means.append(sample_mean)
            
            if abs(sample_mean - true_mean) > epsilon:
                violations += 1
        
        actual_prob = violations / n_simulations
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "样本均值分布",
                "界的紧密度 vs 样本数",
                "界的紧密度 vs 偏差",
                "Hoeffding vs 实际概率"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. 样本均值的直方图
        fig.add_trace(
            go.Histogram(
                x=sample_means,
                nbinsx=50,
                name='样本均值分布',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 添加真实均值
        fig.add_vline(x=true_mean, line_dash="dash", line_color="red",
                     annotation_text=f"μ={true_mean}",
                     row=1, col=1)
        
        # 添加ε区间
        fig.add_vrect(
            x0=true_mean - epsilon, x1=true_mean + epsilon,
            fillcolor="green", opacity=0.2,
            annotation_text=f"ε={epsilon}",
            row=1, col=1
        )
        
        # 2. 界随样本数变化
        N_range = np.arange(10, 1001, 10)
        bounds = 2 * np.exp(-2 * N_range * epsilon**2)
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=bounds,
                mode='lines',
                name='Hoeffding界',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )
        
        # 标注当前N
        fig.add_trace(
            go.Scatter(
                x=[N],
                y=[hoeffding_bound],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name=f'当前N={N}',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 3. 界随epsilon变化
        eps_range = np.linspace(0.01, 0.5, 50)
        bounds_eps = 2 * np.exp(-2 * N * eps_range**2)
        
        fig.add_trace(
            go.Scatter(
                x=eps_range,
                y=bounds_eps,
                mode='lines',
                name='Hoeffding界',
                line=dict(color='purple', width=3)
            ),
            row=2, col=1
        )
        
        # 标注当前epsilon
        fig.add_trace(
            go.Scatter(
                x=[epsilon],
                y=[hoeffding_bound],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name=f'当前ε={epsilon}',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # 4. Hoeffding界 vs 实际概率
        categories = ['Hoeffding界', '实际概率']
        values = [hoeffding_bound, actual_prob]
        colors = ['red', 'blue']
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f'{v:.6f}' for v in values],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="样本均值", row=1, col=1)
        fig.update_yaxes(title_text="频数", row=1, col=1)
        fig.update_xaxes(title_text="样本数 N", row=1, col=2)
        fig.update_yaxes(title_text="概率上界", type="log", row=1, col=2)
        fig.update_xaxes(title_text="偏差 ε", row=2, col=1)
        fig.update_yaxes(title_text="概率上界", type="log", row=2, col=1)
        fig.update_yaxes(title_text="概率", type="log", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Hoeffding不等式 (N={N}, ε={epsilon})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 统计分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Hoeffding界", f"{hoeffding_bound:.6f}")
        
        with col2:
            st.metric("实际概率", f"{actual_prob:.6f}")
        
        with col3:
            tightness = actual_prob / hoeffding_bound if hoeffding_bound > 0 else 0
            st.metric("界的紧密度", f"{tightness:.2%}")
        
        with col4:
            safety_margin = hoeffding_bound / (actual_prob + 1e-10)
            st.metric("安全边际", f"{safety_margin:.1f}x")
        
        # 理论解释
        st.markdown("### 🎓 理论要点")
        
        st.success(r"""
        **Hoeffding不等式的深层含义**:
        
        1. **指数衰减**: 概率以 $e^{-2N\epsilon^2}$ 速度衰减
           - $N$ 翻倍 → 概率平方衰减
           - 这是为什么大数据有用！
        
        2. **与VC维的联系**: 
           - 这个界适用于**单一假设**
           - 要推广到假设类，需要Union Bound
           - 但假设类可能无限大 → 引入增长函数
        
        3. **PAC学习的基石**:
           - Probably (概率 ≥ 1-δ)
           - Approximately (误差 ≤ ε)
           - Correct (泛化误差有界)
        """)
        
        if hoeffding_bound < 0.05:
            st.success(f"""
            ✅ **界很紧**: Hoeffding界 = {hoeffding_bound:.6f} < 0.05
            
            在当前参数下，样本均值以高概率接近期望。这就是为什么经验风险最小化(ERM)有效！
            """)
        else:
            st.warning(f"""
            ⚠️ **界较松**: Hoeffding界 = {hoeffding_bound:.6f} > 0.05
            
            建议: 增大样本数N 或 接受更大的偏差ε
            """)
    
    @staticmethod
    def _render_growth_function():
        """增长函数与Sauer-Shelah引理可视化"""
        st.markdown("### 📈 增长函数：从指数到多项式的奇迹")
        
        st.markdown(r"""
        **问题**: 无限假设类怎么办？不能用Union Bound！
        
        **解决方案**: 增长函数 $\Pi_\mathcal{H}(N)$
        
        **定义**: 在 $N$ 个点上，假设类能产生的最多不同二分类数
        """)
        
        st.latex(r"""
        \Pi_\mathcal{H}(N) = \max_{x_1,...,x_N} |\{(h(x_1),...,h(x_N)) : h \in \mathcal{H}\}|
        """)
        
        st.markdown(r"""
        **Sauer-Shelah引理 (1972)**:
        
        如果 $\text{VC-dim}(\mathcal{H}) = d < \infty$，则：
        """)
        
        st.latex(r"""
        \Pi_\mathcal{H}(N) \leq \sum_{i=0}^{d} \binom{N}{i} \leq \left(\frac{eN}{d}\right)^d
        """)
        
        st.markdown("**关键转折**: 从 $2^N$ (指数) 变为 $O(N^d)$ (多项式)！")
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            vc_dim = st.slider("VC维 d", 1, 10, 3, 1)
            max_n = st.slider("最大样本数", 10, 100, 50, 5)
        
        # 计算增长函数
        N_range = np.arange(1, max_n + 1)
        
        # 指数增长（如果没有VC维限制）
        exponential = 2 ** N_range
        
        # Sauer-Shelah上界（精确）
        sauer_bound = []
        for n in N_range:
            bound = sum(comb(n, i, exact=True) for i in range(min(vc_dim + 1, n + 1)))
            sauer_bound.append(bound)
        
        # 多项式近似
        polynomial_approx = (np.e * N_range / vc_dim) ** vc_dim
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "增长函数对比 (对数尺度)",
                "Break Point现象",
                "多项式 vs 指数增长",
                "Sauer-Shelah界的紧密度"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. 增长函数对比
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=exponential,
                mode='lines',
                name='2^N (无限制)',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=sauer_bound,
                mode='lines+markers',
                name=f'Π(N) with VC-dim={vc_dim}',
                line=dict(color='blue', width=3),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=polynomial_approx,
                mode='lines',
                name=f'(eN/d)^d 近似',
                line=dict(color='green', width=2, dash='dot')
            ),
            row=1, col=1
        )
        
        # Break point标注
        break_point = vc_dim + 1
        if break_point <= max_n:
            fig.add_vline(x=break_point, line_dash="dash", line_color="orange",
                         annotation_text=f"Break Point={break_point}",
                         row=1, col=1)
        
        # 2. Break Point现象
        # 计算增长率
        growth_rates = []
        for i in range(1, len(sauer_bound)):
            if sauer_bound[i-1] > 0:
                rate = sauer_bound[i] / sauer_bound[i-1]
            else:
                rate = 0
            growth_rates.append(rate)
        
        fig.add_trace(
            go.Scatter(
                x=N_range[1:],
                y=growth_rates,
                mode='lines+markers',
                name='增长率 Π(N)/Π(N-1)',
                line=dict(color='purple', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # 标注2（指数增长率）
        fig.add_hline(y=2, line_dash="dash", line_color="red",
                     annotation_text="指数增长率=2",
                     row=1, col=2)
        
        if break_point <= max_n:
            fig.add_vline(x=break_point, line_dash="dash", line_color="orange",
                         row=1, col=2)
        
        # 3. 线性尺度对比（看清差异）
        n_small = min(20, max_n)
        fig.add_trace(
            go.Scatter(
                x=N_range[:n_small],
                y=exponential[:n_small],
                mode='lines+markers',
                name='2^N',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=N_range[:n_small],
                y=sauer_bound[:n_small],
                mode='lines+markers',
                name=f'Π(N) VC-dim={vc_dim}',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 4. 界的紧密度
        tightness = np.array(sauer_bound) / np.array(polynomial_approx)
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=tightness,
                mode='lines+markers',
                name='精确值/近似值',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        fig.add_hline(y=1, line_dash="dash", line_color="gray",
                     annotation_text="完美",
                     row=2, col=2)
        
        fig.update_xaxes(title_text="样本数 N", row=1, col=1)
        fig.update_yaxes(title_text="增长函数", type="log", row=1, col=1)
        fig.update_xaxes(title_text="样本数 N", row=1, col=2)
        fig.update_yaxes(title_text="增长率", row=1, col=2)
        fig.update_xaxes(title_text="样本数 N", row=2, col=1)
        fig.update_yaxes(title_text="增长函数", row=2, col=1)
        fig.update_xaxes(title_text="样本数 N", row=2, col=2)
        fig.update_yaxes(title_text="紧密度", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"增长函数与Sauer-Shelah引理 (VC-dim={vc_dim})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 数量级对比")
        
        n_test = min(20, max_n)
        idx = n_test - 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("N=20时", "")
            st.caption(f"2^N = {exponential[idx]:,.0f}")
        
        with col2:
            st.metric("Π(N)", "")
            st.caption(f"{sauer_bound[idx]:,.0f}")
        
        with col3:
            ratio = exponential[idx] / sauer_bound[idx] if sauer_bound[idx] > 0 else float('inf')
            st.metric("压缩比", "")
            st.caption(f"{ratio:,.0f}x")
        
        with col4:
            st.metric("Break Point", f"{vc_dim + 1}")
        
        # 理论解释
        st.markdown("### 🎓 Sauer-Shelah引理的深层意义")
        
        st.success(r"""
        **为什么这个定理如此重要？**
        
        1. **从无限到有限**:
           - 假设类可能有无限多个假设（如所有超平面）
           - 但"有效假设数"是有限的：$\Pi(N) \leq O(N^d)$
        
        2. **Break Point**:
           - 在 $N = d+1$ 处，增长率从2骤降
           - 之后变为多项式增长
           - **这是VC维的定义来源！**
        
        3. **泛化界的关键**:
           - 用 $\Pi(N)$ 替代 $|\mathcal{H}|$ 在Union Bound中
           - $P(\text{bad}) \leq 2\Pi(2N)e^{-2N\epsilon^2}$
           - 多项式 × 指数衰减 = 仍然衰减！
        """)
        
        st.info(r"""
        **证明思路** (组合数学):
        
        **引理**: 如果能打散 $d+1$ 个点，就能打散某个 $d$ 个点的子集
        
        **递推**: $\Pi(N) = \Pi(N-1) + \Pi_{\text{restrict}}(N-1)$
        
        **归纳**: 最终得到 $\Pi(N) \leq \sum_{i=0}^{d} \binom{N}{i}$
        
        这个证明被称为"组合数学的珍珠"！
        """)
    
    @staticmethod
    def _render_vc_bound():
        """VC泛化界完整推导可视化"""
        st.markdown("### 🎯 VC泛化界：完整推导链条")
        
        st.markdown(r"""
        **目标**: 证明经验风险和真实风险的差距有界
        
        **VC泛化不等式**:
        """)
        
        st.latex(r"""
        P\left(\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| > \epsilon\right) 
        \leq 4\Pi_\mathcal{H}(2N) e^{-\frac{1}{8}N\epsilon^2}
        """)
        
        st.markdown(r"""
        **推导步骤**:
        1. 单个假设 → Hoeffding不等式
        2. 有限假设 → Union Bound
        3. 无限假设 → 增长函数替代
        4. 对称化技巧 → Ghost样本
        5. VC维上界 → Sauer-Shelah引理
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            vc_dim = st.slider("VC维 d", 1, 10, 3, 1)
            N = st.slider("样本数 N", 50, 2000, 500, 50)
            confidence = st.slider("置信度 1-δ", 0.90, 0.99, 0.95, 0.01)
        
        delta = 1 - confidence
        
        # 计算不同的界
        N_range = np.arange(vc_dim + 1, 2001, 10)
        
        # 1. 朴素Union Bound (假设100个假设)
        naive_bound_epsilon = lambda n: np.sqrt(np.log(200/delta) / (2*n))
        
        # 2. VC泛化界
        vc_bound_epsilon = lambda n: np.sqrt(8 * (vc_dim * np.log(2*np.e*n/vc_dim) + np.log(4/delta)) / n)
        
        # 3. Rademacher复杂度（更紧）
        rademacher_epsilon = lambda n: np.sqrt(2 * vc_dim * np.log(n) / n) + np.sqrt(np.log(1/delta) / (2*n))
        
        naive_epsilons = [naive_bound_epsilon(n) for n in N_range]
        vc_epsilons = [vc_bound_epsilon(n) for n in N_range]
        rademacher_epsilons = [rademacher_epsilon(n) for n in N_range]
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "泛化误差界 vs 样本数",
                "推导步骤流程",
                "样本复杂度",
                "不同界的对比"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. 泛化误差界
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=naive_epsilons,
                mode='lines',
                name='朴素Union Bound',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=vc_epsilons,
                mode='lines',
                name='VC泛化界',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=rademacher_epsilons,
                mode='lines',
                name='Rademacher界',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # 标注当前N
        current_vc_eps = vc_bound_epsilon(N)
        fig.add_trace(
            go.Scatter(
                x=[N],
                y=[current_vc_eps],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name=f'当前N={N}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. 推导流程图（使用Bar模拟）
        steps = ['Hoeffding', 'Union\nBound', '增长函数', '对称化', 'Sauer-\nShelah']
        step_values = [1, 0.8, 0.5, 0.4, 0.3]  # 相对宽松程度
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#95E1D3']
        
        fig.add_trace(
            go.Bar(
                x=steps,
                y=step_values,
                marker_color=colors,
                text=['单假设', '有限类', '无限类', 'Ghost样本', 'VC维界'],
                textposition='inside'
            ),
            row=1, col=2
        )
        
        # 3. 样本复杂度（给定ε，需要多少样本）
        epsilon_targets = np.linspace(0.01, 0.5, 50)
        
        # 反解N：使得 vc_bound_epsilon(N) ≤ ε
        sample_complexity = []
        for eps in epsilon_targets:
            # 粗略估计：N ~ O(d/ε^2 * log(1/ε))
            n_approx = int(vc_dim / eps**2 * np.log(1/eps) * 10)
            sample_complexity.append(n_approx)
        
        fig.add_trace(
            go.Scatter(
                x=epsilon_targets,
                y=sample_complexity,
                mode='lines',
                name='样本复杂度',
                line=dict(color='purple', width=3),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.1)'
            ),
            row=2, col=1
        )
        
        # 4. 不同VC维的对比
        for d in [1, 3, 5, 10]:
            epsilons_d = [np.sqrt(8 * (d * np.log(2*np.e*n/d) + np.log(4/delta)) / n) 
                         for n in N_range]
            fig.add_trace(
                go.Scatter(
                    x=N_range,
                    y=epsilons_d,
                    mode='lines',
                    name=f'VC-dim={d}',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="样本数 N", row=1, col=1)
        fig.update_yaxes(title_text="泛化误差 ε", row=1, col=1)
        fig.update_yaxes(title_text="界的相对宽松度", row=1, col=2)
        fig.update_xaxes(title_text="目标精度 ε", row=2, col=1)
        fig.update_yaxes(title_text="所需样本数", row=2, col=1)
        fig.update_xaxes(title_text="样本数 N", row=2, col=2)
        fig.update_yaxes(title_text="泛化误差 ε", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"VC泛化界 (d={vc_dim}, δ={delta:.2f})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 当前配置的分析
        st.markdown("### 📊 当前配置分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VC维 d", vc_dim)
        
        with col2:
            st.metric("样本数 N", N)
        
        with col3:
            st.metric("置信度", f"{confidence:.2%}")
        
        with col4:
            st.metric("泛化误差界 ε", f"{current_vc_eps:.4f}")
        
        # PAC可学习性判断
        st.markdown("### 🎯 PAC可学习性")
        
        # PAC学习条件：样本复杂度 = O(d/ε^2 * log(1/δ))
        pac_sample_complexity = int(vc_dim / 0.01**2 * np.log(1/delta) * 100)
        
        if N >= pac_sample_complexity * 0.1:
            st.success(f"""
            ✅ **PAC可学习**
            
            当前样本数 N={N} 足够学习VC维为 {vc_dim} 的假设类。
            
            **泛化保证**: 以至少 {confidence:.1%} 的概率，真实误差与经验误差之差 ≤ {current_vc_eps:.4f}
            
            这意味着: $R(h) \leq \hat{R}(h) + {current_vc_eps:.4f}$
            """)
        else:
            st.warning(f"""
            ⚠️ **样本不足**
            
            建议样本数: 至少 {pac_sample_complexity} （用于ε=0.01）
            当前样本数: {N}
            
            需要更多数据！
            """)
        
        # 理论深入
        st.markdown("### 🎓 推导的关键技巧")
        
        st.success(r"""
        **1. 对称化 (Symmetrization)**:
        
        引入"Ghost样本" $\{x_1', ..., x_N'\}$，同分布但独立
        
        $$P(|R - \hat{R}| > \epsilon) \leq 2P\left(\sup_h |\hat{R}(h) - \hat{R}'(h)| > \frac{\epsilon}{2}\right)$$
        
        **巧妙之处**: 把依赖于分布的真实误差转化为只依赖于样本的经验误差！
        """)
        
        st.info(r"""
        **2. 增长函数的作用**:
        
        在对称化后：
        $$P \leq 2 \cdot \Pi_\mathcal{H}(2N) \cdot \exp(-2N\epsilon^2)$$
        
        **关键**: $\Pi(2N) \leq (2N)^d$ 是多项式，而 $\exp(-N\epsilon^2)$ 是指数衰减
        
        **结论**: 多项式 × 指数衰减 = 仍然衰减到0！
        """)
    
    @staticmethod
    def _render_radon():
        """Radon定理可视化"""
        st.markdown("### 🔺 Radon定理：为什么VC维=d+1？")
        
        st.markdown(r"""
        **Radon定理**: 在 $\mathbb{R}^d$ 中，任意 $d+2$ 个点都可以分成两组，使得它们的凸包相交
        
        **推论**: 线性分类器在 $\mathbb{R}^d$ 中的VC维 ≤ $d+1$
        
        **几何直观**: 
        - 在平面($d=2$)上，4个点必有"内点"
        - 在空间($d=3$)上，5个点必有"内点"
        - 无论怎么放，总有一个配置无法被线性分类器打散
        """)
        
        st.info("""
        **证明思路** (简化版):
        
        1. 取 $d+2$ 个点 $x_1, ..., x_{d+2}$
        2. 由于在 $d$ 维空间中，$d+2$ 个点线性相关
        3. 存在系数 $\lambda_i$ 使得 $\sum \lambda_i x_i = 0$
        4. 将正系数和负系数的点分成两组
        5. 证明两组的凸包相交 → 必有一个配置无法线性分离
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            dimension = st.slider("维度 d", 1, 3, 2, 1)
        
        if dimension == 1:
            st.markdown("#### 1维情况 (直线上)")
            
            st.markdown("""
            - VC维 = 2
            - 任意2个点都可以被一个阈值分类器打散
            - 但3个点不行（如果中间点单独一类）
            """)
            
            # 1D可视化
            points_1d = np.array([-1, 0, 1])
            labels_impossible = np.array([-1, 1, -1])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=points_1d,
                y=[0, 0, 0],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=['blue' if l == -1 else 'red' for l in labels_impossible]
                ),
                text=['点1<br>(蓝)', '点2<br>(红)', '点3<br>(蓝)'],
                textposition='top center',
                name='无法分离的配置'
            ))
            
            fig.update_layout(
                title="1维: 3个点的无法分离配置",
                xaxis_title="x",
                yaxis=dict(range=[-1, 1], showticklabels=False),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif dimension == 2:
            st.markdown("#### 2维情况 (平面上)")
            
            st.markdown("""
            - VC维 = 3
            - 任意3个点都可以被一条直线打散
            - 但4个点不行（XOR问题）
            """)
            
            # 2D可视化：XOR配置
            points_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            labels_xor = np.array([1, -1, -1, 1])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=points_2d[:, 0],
                y=points_2d[:, 1],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=['red' if l == 1 else 'blue' for l in labels_xor]
                ),
                text=['红', '蓝', '蓝', '红'],
                textposition='top center',
                name='XOR配置'
            ))
            
            # 画凸包
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            for simplex in hull.simplices:
                fig.add_trace(go.Scatter(
                    x=points_2d[simplex, 0],
                    y=points_2d[simplex, 1],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="2维: XOR配置无法被直线分离",
                xaxis_title="x₁",
                yaxis_title="x₂",
                height=500,
                xaxis=dict(range=[-0.5, 1.5]),
                yaxis=dict(range=[-0.5, 1.5], scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # 3D
            st.markdown("#### 3维情况 (空间中)")
            
            st.markdown("""
            - VC维 = 4
            - 任意4个点都可以被一个平面打散
            - 但5个点不行
            """)
            
            st.warning("3D可视化较复杂，这里展示概念理解")
        
        # 通用的VC维表格
        st.markdown("### 📋 常见模型的VC维")
        
        vc_table = pd.DataFrame({
            '模型': [
                '1D阈值分类器',
                '2D线性分类器',
                'd维线性分类器',
                'd维感知机',
                'k-NN (k固定)',
                '决策树 (深度h)',
                '神经网络 (W个权重)'
            ],
            'VC维': [
                '2',
                '3',
                'd+1',
                'd+1',
                '∞',
                'O(节点数)',
                'O(W log W)'
            ],
            '说明': [
                '一个阈值',
                '直线分离',
                'Radon定理保证',
                '同线性分类器',
                '无界，过拟合风险高',
                '指数复杂度',
                '远小于参数数量！'
            ]
        })
        
        st.dataframe(vc_table, use_container_width=True)
        
        st.success(r"""
        **Radon定理的深层含义**:
        
        1. **几何限制**: 高维空间的几何结构限制了分类能力
        2. **VC维上界**: 为什么线性模型的VC维是 $d+1$ 而不是无穷大
        3. **深度学习**: 神经网络的VC维 $O(W \log W)$ 而非 $O(2^W)$
        
        **哲学意义**: 世界的结构性（几何约束）使得学习成为可能！
        """)
    
    @staticmethod
    def _render_effective_vcdim():
        """有效VC维可视化"""
        st.markdown("### 🎯 有效VC维：正则化的理论解释")
        
        st.markdown(r"""
        **问题**: 深度神经网络的VC维巨大（$O(W \log W)$），但为什么不过拟合？
        
        **答案**: **有效VC维** 远小于理论VC维
        
        **定义**:
        """)
        
        st.latex(r"""
        d_{eff} = \frac{N \cdot (R_{train} - R_{opt})}{R_{train}}
        """)
        
        st.markdown(r"""
        或使用数据依赖的界:
        """)
        
        st.latex(r"""
        d_{eff} \leq \frac{trace(H)}{\|w\|^2} \text{ (谱正则化)}
        """)
        
        st.markdown("""
        **物理意义**: 
        - 理论VC维：假设类的"容量"
        - 有效VC维：实际使用的"容量"
        - 正则化、Early Stopping降低有效VC维
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_params = st.slider("参数数量 W", 100, 10000, 1000, 100)
            regularization = st.slider("正则化强度 λ", 0.0, 1.0, 0.1, 0.05)
            n_samples = st.slider("样本数 N", 100, 5000, 1000, 100)
        
        # 计算不同的VC维
        # 理论VC维（神经网络）
        theoretical_vc = n_params * np.log2(n_params)
        
        # 有效VC维（简化模型：随正则化降低）
        effective_vc = theoretical_vc * (1 - regularization) ** 2
        
        # Rademacher复杂度（数据依赖）
        rademacher_complexity = np.sqrt(n_params / n_samples)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "VC维 vs 正则化强度",
                "泛化界对比",
                "有效容量 vs 样本数",
                "过拟合风险"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. VC维随正则化变化
        lambda_range = np.linspace(0, 1, 50)
        theoretical_line = [theoretical_vc] * len(lambda_range)
        effective_line = [theoretical_vc * (1 - l) ** 2 for l in lambda_range]
        
        fig.add_trace(
            go.Scatter(
                x=lambda_range,
                y=theoretical_line,
                mode='lines',
                name='理论VC维',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=lambda_range,
                y=effective_line,
                mode='lines',
                name='有效VC维',
                line=dict(color='blue', width=3),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        # 标注当前正则化
        fig.add_trace(
            go.Scatter(
                x=[regularization],
                y=[effective_vc],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name='当前配置',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. 泛化界对比
        # 使用理论VC维的界
        epsilon_theoretical = np.sqrt(8 * theoretical_vc * np.log(n_samples) / n_samples)
        # 使用有效VC维的界
        epsilon_effective = np.sqrt(8 * effective_vc * np.log(n_samples) / n_samples)
        # Rademacher界
        epsilon_rademacher = 2 * rademacher_complexity
        
        bounds = ['理论VC维', '有效VC维', 'Rademacher']
        values = [epsilon_theoretical, epsilon_effective, epsilon_rademacher]
        colors = ['red', 'blue', 'green']
        
        fig.add_trace(
            go.Bar(
                x=bounds,
                y=values,
                marker_color=colors,
                text=[f'{v:.4f}' for v in values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. 有效容量 vs 样本数
        n_range = np.arange(100, 5001, 100)
        capacity_ratio = [min(effective_vc / n, 1.0) for n in n_range]
        
        fig.add_trace(
            go.Scatter(
                x=n_range,
                y=capacity_ratio,
                mode='lines',
                name='d_eff / N',
                line=dict(color='purple', width=3),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.1)'
            ),
            row=2, col=1
        )
        
        # 安全区域
        fig.add_hrect(
            y0=0, y1=0.1,
            fillcolor="green", opacity=0.1,
            annotation_text="安全区",
            row=2, col=1
        )
        
        # 标注当前样本数
        current_ratio = effective_vc / n_samples
        fig.add_trace(
            go.Scatter(
                x=[n_samples],
                y=[current_ratio],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 过拟合风险评估
        # 简化模型：risk = exp(-N/(d_eff * k))
        n_risk_range = np.arange(100, 5001, 100)
        risk_theoretical = np.exp(-n_risk_range / (theoretical_vc * 2))
        risk_effective = np.exp(-n_risk_range / (effective_vc * 2))
        
        fig.add_trace(
            go.Scatter(
                x=n_risk_range,
                y=risk_theoretical,
                mode='lines',
                name='无正则化',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=n_risk_range,
                y=risk_effective,
                mode='lines',
                name='有正则化',
                line=dict(color='blue', width=3)
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="正则化强度 λ", row=1, col=1)
        fig.update_yaxes(title_text="VC维", row=1, col=1)
        fig.update_yaxes(title_text="泛化误差界", row=1, col=2)
        fig.update_xaxes(title_text="样本数 N", row=2, col=1)
        fig.update_yaxes(title_text="容量比 d_eff/N", row=2, col=1)
        fig.update_xaxes(title_text="样本数 N", row=2, col=2)
        fig.update_yaxes(title_text="过拟合风险", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"有效VC维 (W={n_params}, λ={regularization}, N={n_samples})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 当前配置分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("理论VC维", f"{theoretical_vc:.0f}")
        
        with col2:
            st.metric("有效VC维", f"{effective_vc:.0f}")
        
        with col3:
            compression = theoretical_vc / effective_vc if effective_vc > 0 else float('inf')
            st.metric("压缩比", f"{compression:.1f}x")
        
        with col4:
            st.metric("d_eff/N", f"{current_ratio:.3f}")
        
        # 建议
        if current_ratio > 0.5:
            st.error("""
            ❌ **严重过拟合风险**
            
            有效VC维过大相对于样本数：d_eff/N > 0.5
            
            **建议**:
            - 增大正则化强度
            - 增加训练数据
            - 使用Early Stopping
            - 考虑模型简化
            """)
        elif current_ratio > 0.1:
            st.warning("""
            ⚠️ **中等过拟合风险**
            
            建议增加正则化或数据量
            """)
        else:
            st.success("""
            ✅ **泛化性能良好**
            
            有效VC维相对样本数很小，模型不会过拟合
            """)
        
        # 理论解释
        st.markdown("### 🎓 深度学习的VC维悖论")
        
        st.info(r"""
        **悖论**: 
        - ResNet-50: 25M参数 → 理论VC维 $\sim 10^8$
        - ImageNet: 1.2M样本
        - 按照VC理论：应该严重过拟合！
        - **但实际**: 泛化很好
        
        **解释**:
        1. **隐式正则化**: SGD本身就是正则化
        2. **结构先验**: 卷积、归一化降低有效容量
        3. **数据增强**: 有效样本数远大于1.2M
        4. **有效VC维**: 实际使用的容量 ≪ 理论容量
        """)
        
        st.success("""
        **现代理解**:
        
        VC维理论是**充分条件**，不是必要条件：
        - 有限VC维 → 能泛化 ✅
        - 无限VC维 → 不一定不能泛化 ⚠️
        
        **新理论**:
        - Rademacher复杂度（数据依赖）
        - 算法稳定性
        - PAC-Bayes
        - 神经正切核(NTK)
        """)
    
    @staticmethod
    def _render_limitations():
        """理论局限性分析"""
        st.markdown("### ⚠️ VC维理论的局限性")
        
        st.markdown("""
        **VC维理论是伟大的**，但它有局限性：
        
        1. **界过于保守** - 实际泛化远好于理论预测
        2. **与数据无关** - 只看假设类，不看数据分布
        3. **深度学习悖论** - 无法解释现代大模型
        4. **忽略算法** - 只看假设空间，不看优化过程
        """)
        
        # 对比表
        comparison_data = {
            '理论框架': ['VC维', 'Rademacher', 'PAC-Bayes', 'NTK', '算法稳定性'],
            '数据依赖': ['❌', '✅', '✅', '✅', '✅'],
            '算法依赖': ['❌', '❌', '✅', '✅', '✅'],
            '深度学习': ['❌', '⚠️', '✅', '✅', '⚠️'],
            '界的紧密度': ['松', '中', '紧', '紧', '中']
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 📚 现代泛化理论进展")
        
        st.info("""
        **1. Rademacher复杂度** (Ch 7笔记有详细介绍):
        - 数据依赖的复杂度度量
        - 比VC维更紧
        - $\mathcal{R}_S(\mathcal{H}) = \mathbb{E}_\sigma\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(x_i)\right]$
        
        **2. PAC-Bayes**:
        - 考虑先验分布
        - 适用于贝叶斯方法
        - 可以解释随机网络的泛化
        
        **3. 神经正切核 (NTK)** (Ch 18训练动力学):
        - 无限宽网络的极限
        - 懒惰训练regime
        - 过参数化的理论解释
        
        **4. 算法稳定性**:
        - SGD的隐式正则化
        - 噪声注入的泛化效果
        - 解释Early Stopping
        """)
        
        st.success("""
        **结论**:
        
        VC维理论是学习理论的**基石**，但不是**全部**。
        
        **价值**:
        - ✅ 建立了PAC学习框架
        - ✅ 证明了学习的可能性
        - ✅ 给出了样本复杂度的数量级
        
        **局限**:
        - ❌ 界过于保守
        - ❌ 无法解释深度学习
        - ❌ 忽略了算法和数据的作用
        
        **现代研究**: 结合VC维、Rademacher、PAC-Bayes、NTK等多种工具
        """)
    
    @staticmethod
    def _render_derivation_flow():
        """完整推导流程图"""
        st.markdown("### 🗺️ 完整推导流程图")
        
        st.markdown("""
        这是从概率不等式到泛化界的完整逻辑链条：
        """)
        
        # 使用Sankey图展示推导流程
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = [
                    "Hoeffding不等式",
                    "Union Bound",
                    "增长函数",
                    "对称化",
                    "Sauer-Shelah",
                    "VC泛化界",
                    "PAC可学习"
                ],
                color = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#95E1D3", "#FFD93D", "#6BCB77"]
            ),
            link = dict(
                source = [0, 1, 2, 3, 4, 5],
                target = [1, 2, 3, 4, 5, 6],
                value = [1, 1, 1, 1, 1, 1],
                label = [
                    "单假设→有限假设",
                    "有限→无限",
                    "依赖分布→依赖样本",
                    "无限→多项式",
                    "多项式界→泛化界",
                    "理论→应用"
                ]
            )
        )])
        
        fig.update_layout(
            title_text="VC维理论推导流程",
            font_size=12,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细步骤
        st.markdown("### 📝 逐步推导")
        
        with st.expander("步骤1: Hoeffding不等式 → 单假设泛化"):
            st.latex(r"""
            P(|R(h) - \hat{R}(h)| > \epsilon) \leq 2e^{-2N\epsilon^2}
            """)
            st.markdown("**说明**: 单个假设的经验风险收敛到真实风险")
        
        with st.expander("步骤2: Union Bound → 有限假设类"):
            st.latex(r"""
            P\left(\exists h: |R(h) - \hat{R}(h)| > \epsilon\right) \leq 2|\mathcal{H}|e^{-2N\epsilon^2}
            """)
            st.markdown("**说明**: 对所有假设取并集，但假设类可能无限大")
        
        with st.expander("步骤3: 增长函数 → 无限假设类"):
            st.latex(r"""
            |\mathcal{H}| \rightarrow \Pi_\mathcal{H}(N)
            """)
            st.markdown("**说明**: 用增长函数（有效假设数）代替假设类大小")
        
        with st.expander("步骤4: 对称化 → Ghost样本"):
            st.latex(r"""
            P \leq 2P\left(\sup_h |\hat{R}(h) - \hat{R}'(h)| > \frac{\epsilon}{2}\right) \leq 4\Pi(2N)e^{-\frac{1}{8}N\epsilon^2}
            """)
            st.markdown("**说明**: 引入Ghost样本，将依赖分布的问题转化为依赖样本")
        
        with st.expander("步骤5: Sauer-Shelah → 多项式上界"):
            st.latex(r"""
            \Pi_\mathcal{H}(N) \leq \left(\frac{eN}{d}\right)^d
            """)
            st.markdown("**说明**: 增长函数从指数变为多项式")
        
        with st.expander("步骤6: VC泛化界 → 最终结果"):
            st.latex(r"""
            P\left(\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| > \epsilon\right) \leq 4\left(\frac{2eN}{d}\right)^d e^{-\frac{1}{8}N\epsilon^2}
            """)
            st.markdown("**说明**: 多项式 × 指数衰减 = 仍然收敛！")
        
        # 导航回VC维基础
        st.info("""
        💡 **想回顾基础概念？** → 返回 **VC维理论** 模块
        
        两个模块互补：
        - **VC维理论**: 直觉理解、应用案例
        - **VC维详细推导**: 数学证明、理论深度
        """)

        # 添加交互式测验

# 导入必要的包
import pandas as pd

        # 添加交互式测验
