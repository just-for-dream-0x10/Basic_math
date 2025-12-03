"""
交互式概率与信息论可视化
严格按照 0.3.Probability_Information.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveProbability:
    """交互式概率与信息论可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎲 交互式概率与信息论")
        st.markdown("""
        **概率论**: 研究随机现象的数学分支
        
        **信息论**: 量化信息的数学理论，由Claude Shannon创立
        
        **核心概念**:
        - 熵 (Entropy): 不确定性的度量
        - KL散度 (KL Divergence): 两个分布的差异
        - 互信息 (Mutual Information): 变量间的依赖关系
        - 交叉熵 (Cross Entropy): 机器学习中最常用的损失函数
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择主题")
            topic = st.selectbox("主题", [
                "概率分布",
                "熵 (Entropy)",
                "KL散度",
                "交叉熵与损失函数",
                "互信息",
                "贝叶斯推断"
            ])
        
        if topic == "概率分布":
            InteractiveProbability._render_distributions()
        elif topic == "熵 (Entropy)":
            InteractiveProbability._render_entropy()
        elif topic == "KL散度":
            InteractiveProbability._render_kl_divergence()
        elif topic == "交叉熵与损失函数":
            InteractiveProbability._render_cross_entropy()
        elif topic == "互信息":
            InteractiveProbability._render_mutual_information()
        elif topic == "贝叶斯推断":
            InteractiveProbability._render_bayes()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("probability")
        quizzes = QuizTemplates.get_probability_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_distributions():
        """概率分布可视化"""
        st.markdown("### 📊 常见概率分布")
        
        with st.sidebar:
            st.markdown("### 🎛️ 分布类型")
            dist_type = st.selectbox("分布", [
                "正态分布 (Gaussian)",
                "伯努利分布 (Bernoulli)",
                "二项分布 (Binomial)",
                "泊松分布 (Poisson)",
                "指数分布 (Exponential)",
                "Beta分布"
            ])
        
        if dist_type == "正态分布 (Gaussian)":
            st.latex(r"p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)")
            
            mu = st.sidebar.slider("均值 μ", -5.0, 5.0, 0.0, 0.1)
            sigma = st.sidebar.slider("标准差 σ", 0.1, 3.0, 1.0, 0.1)
            
            x = np.linspace(-10, 10, 1000)
            y = stats.norm.pdf(x, mu, sigma)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='PDF'))
            
            # 标注关键点
            fig.add_vline(x=mu, line_dash="dash", line_color="red",
                         annotation_text=f"μ = {mu}")
            fig.add_vline(x=mu-sigma, line_dash="dot", line_color="orange",
                         annotation_text=f"μ-σ")
            fig.add_vline(x=mu+sigma, line_dash="dot", line_color="orange",
                         annotation_text=f"μ+σ")
            
            fig.update_layout(
                title=f"正态分布 N({mu}, {sigma}²)",
                xaxis_title="x",
                yaxis_title="概率密度 p(x)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **正态分布性质**:
            - 均值 = 中位数 = 众数 = {mu}
            - 68.27% 的数据在 [μ-σ, μ+σ] = [{mu-sigma:.2f}, {mu+sigma:.2f}]
            - 95.45% 的数据在 [μ-2σ, μ+2σ] = [{mu-2*sigma:.2f}, {mu+2*sigma:.2f}]
            - 熵: $H = \\frac{1}{2}\\log(2\\pi e \\sigma^2) = {0.5 * np.log(2*np.pi*np.e*sigma**2):.3f}$ nats
            """)
        
        elif dist_type == "伯努利分布 (Bernoulli)":
            st.latex(r"P(X=1) = p, \quad P(X=0) = 1-p")
            
            p = st.sidebar.slider("成功概率 p", 0.0, 1.0, 0.5, 0.01)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[0, 1], y=[1-p, p],
                                marker_color=['blue', 'red'],
                                text=[f'{1-p:.3f}', f'{p:.3f}'],
                                textposition='outside'))
            
            fig.update_layout(
                title=f"伯努利分布 Bernoulli({p})",
                xaxis_title="X",
                yaxis_title="概率 P(X)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 熵
            if p > 0 and p < 1:
                entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
            else:
                entropy = 0
            
            st.markdown(f"""
            **伯努利分布性质**:
            - 期望: $E[X] = p = {p}$
            - 方差: $Var[X] = p(1-p) = {p*(1-p):.4f}$
            - 熵: $H(X) = -p\\log_2 p - (1-p)\\log_2(1-p) = {entropy:.4f}$ bits
            - 熵在 $p=0.5$ 时最大 = 1 bit (最不确定)
            """)
        
        elif dist_type == "二项分布 (Binomial)":
            st.latex(r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}")
            
            n = st.sidebar.slider("试验次数 n", 1, 50, 10)
            p = st.sidebar.slider("成功概率 p", 0.0, 1.0, 0.5, 0.01)
            
            x = np.arange(0, n+1)
            y = stats.binom.pmf(x, n, p)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=y, name='PMF'))
            
            fig.update_layout(
                title=f"二项分布 Binomial(n={n}, p={p})",
                xaxis_title="成功次数 k",
                yaxis_title="概率 P(X=k)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            mean = n * p
            var = n * p * (1-p)
            
            st.markdown(f"""
            **二项分布性质**:
            - 期望: $E[X] = np = {mean:.2f}$
            - 方差: $Var[X] = np(1-p) = {var:.2f}$
            - 标准差: $\\sigma = \\sqrt{{np(1-p)}} = {np.sqrt(var):.2f}$
            - 当 $n$ 很大时，近似正态分布 $N(np, np(1-p))$
            """)
        
        elif dist_type == "泊松分布 (Poisson)":
            st.latex(r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}")
            
            lambda_val = st.sidebar.slider("速率参数 λ", 0.1, 20.0, 3.0, 0.1)
            
            x = np.arange(0, int(lambda_val * 3 + 10))
            y = stats.poisson.pmf(x, lambda_val)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=y, name='PMF'))
            
            fig.update_layout(
                title=f"泊松分布 Poisson(λ={lambda_val})",
                xaxis_title="事件次数 k",
                yaxis_title="概率 P(X=k)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **泊松分布性质**:
            - 期望: $E[X] = \\lambda = {lambda_val}$
            - 方差: $Var[X] = \\lambda = {lambda_val}$
            - 用于建模单位时间内事件发生次数
            - 例如: 网站访问量、放射性衰变、电话呼叫
            """)
        
        elif dist_type == "指数分布 (Exponential)":
            st.latex(r"p(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
            
            lambda_val = st.sidebar.slider("速率参数 λ", 0.1, 5.0, 1.0, 0.1)
            
            x = np.linspace(0, 10, 1000)
            y = stats.expon.pdf(x, scale=1/lambda_val)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='PDF'))
            
            fig.update_layout(
                title=f"指数分布 Exponential(λ={lambda_val})",
                xaxis_title="x",
                yaxis_title="概率密度 p(x)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **指数分布性质**:
            - 期望: $E[X] = \\frac{{1}}{{\\lambda}} = {1/lambda_val:.3f}$
            - 方差: $Var[X] = \\frac{{1}}{{\\lambda^2}} = {1/lambda_val**2:.3f}$
            - 无记忆性: $P(X > s+t | X > s) = P(X > t)$
            - 用于建模等待时间、寿命分布
            """)
        
        else:  # Beta分布
            st.latex(r"p(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0,1]")
            
            alpha = st.sidebar.slider("参数 α", 0.1, 5.0, 2.0, 0.1)
            beta = st.sidebar.slider("参数 β", 0.1, 5.0, 2.0, 0.1)
            
            x = np.linspace(0, 1, 1000)
            y = stats.beta.pdf(x, alpha, beta)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='PDF'))
            
            fig.update_layout(
                title=f"Beta分布 Beta(α={alpha}, β={beta})",
                xaxis_title="x",
                yaxis_title="概率密度 p(x)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            mean = alpha / (alpha + beta)
            mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else None
            
            st.markdown(f"""
            **Beta分布性质**:
            - 期望: $E[X] = \\frac{{\\alpha}}{{\\alpha + \\beta}} = {mean:.3f}$
            - 众数: $\\text{{Mode}} = \\frac{{\\alpha - 1}}{{\\alpha + \\beta - 2}} = {mode if mode else 'N/A'}$
            - Beta分布是[0,1]区间上的共轭先验
            - 用于贝叶斯推断中的概率建模
            - α=β=1 时为均匀分布
            """)
    
    @staticmethod
    def _render_entropy():
        """熵的可视化"""
        st.markdown("### 📏 熵 (Entropy): 不确定性的度量")
        
        st.latex(r"""
        H(X) = -\sum_{i} p(x_i) \log p(x_i)
        """)
        
        st.markdown("""
        **熵的直觉**:
        - 香农熵量化了随机变量的"平均惊奇度"
        - 熵越大，分布越均匀，越不确定
        - 熵越小，分布越集中，越确定
        
        **单位**:
        - 以2为底 (log₂): bits
        - 以e为底 (ln): nats
        - 以10为底 (log₁₀): dits
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 分布设置")
            n_categories = st.slider("类别数", 2, 10, 4)
            dist_type = st.radio("分布类型", ["自定义", "均匀分布", "单峰分布", "双峰分布"])
        
        # 生成概率分布
        if dist_type == "自定义":
            st.markdown("#### 调整概率分布")
            probs = []
            for i in range(n_categories):
                p = st.slider(f"P(X={i})", 0.0, 1.0, 1.0/n_categories, 0.01, key=f"p_{i}")
                probs.append(p)
            probs = np.array(probs)
            probs = probs / probs.sum()  # 归一化
        elif dist_type == "均匀分布":
            probs = np.ones(n_categories) / n_categories
        elif dist_type == "单峰分布":
            probs = np.random.dirichlet(np.ones(n_categories) * 5)
            peak = np.argmax(probs)
            probs = np.exp(-((np.arange(n_categories) - peak)**2) / 2)
            probs = probs / probs.sum()
        else:  # 双峰分布
            probs = np.zeros(n_categories)
            if n_categories >= 2:
                probs[0] = 0.4
                probs[-1] = 0.4
                if n_categories > 2:
                    probs[1:-1] = 0.2 / (n_categories - 2)
        
        # 计算熵
        entropy_bits = -np.sum(probs * np.log2(probs + 1e-10))
        entropy_nats = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log2(n_categories)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("概率分布", "信息量")
        )
        
        # 概率分布
        fig.add_trace(
            go.Bar(x=list(range(n_categories)), y=probs,
                  name='概率', marker_color='blue',
                  text=[f'{p:.3f}' for p in probs],
                  textposition='outside'),
            row=1, col=1
        )
        
        # 信息量 -log(p)
        information = -np.log2(probs + 1e-10)
        fig.add_trace(
            go.Bar(x=list(range(n_categories)), y=information,
                  name='信息量 -log₂(p)', marker_color='red',
                  text=[f'{i:.2f}' for i in information],
                  textposition='outside'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="类别", row=1, col=1)
        fig.update_xaxes(title_text="类别", row=1, col=2)
        fig.update_yaxes(title_text="概率 p(x)", row=1, col=1)
        fig.update_yaxes(title_text="信息量 (bits)", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示熵
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("熵 H(X)", f"{entropy_bits:.4f} bits")
        with col2:
            st.metric("最大熵", f"{max_entropy:.4f} bits")
        with col3:
            st.metric("熵/最大熵", f"{entropy_bits/max_entropy:.2%}")
        
        st.markdown(f"""
        ### 📊 熵的解释
        
        - **当前熵**: {entropy_bits:.4f} bits = {entropy_nats:.4f} nats
        - **最大熵**: {max_entropy:.4f} bits (均匀分布时达到)
        - **归一化熵**: {entropy_bits/max_entropy:.2%}
        
        **含义**:
        - 平均需要 {entropy_bits:.2f} bits 来编码一个样本
        - 如果分布完全确定 (某个概率=1)，熵=0
        - 如果分布完全均匀，熵=log₂({n_categories})={max_entropy:.2f}
        
        **在机器学习中**:
        - 决策树: 选择使信息增益最大的特征
        - 交叉熵损失: 最小化预测分布和真实分布的交叉熵
        - 生成模型: 最大化数据的熵（增加多样性）
        """)
    
    @staticmethod
    def _render_kl_divergence():
        """KL散度可视化"""
        st.markdown("### 📐 KL散度: 分布差异的度量")
        
        st.latex(r"""
        D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
        """)
        
        st.markdown("""
        **KL散度的性质**:
        - ✅ 非负性: $D_{KL}(P \\| Q) \\geq 0$
        - ✅ 当且仅当 $P = Q$ 时等于0
        - ❌ 不对称: $D_{KL}(P \\| Q) \\neq D_{KL}(Q \\| P)$
        - ❌ 不满足三角不等式（不是真正的距离度量）
        
        **物理意义**: 
        - 用分布Q来近似P时的"额外信息量"
        - VAE中的正则化项
        - 强化学习中的策略更新约束
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 分布设置")
            n_categories = st.slider("类别数", 3, 10, 5)
            
            st.markdown("#### 分布P (真实)")
            p_type = st.selectbox("P类型", ["均匀", "单峰", "双峰", "自定义"])
            
            st.markdown("#### 分布Q (近似)")
            q_type = st.selectbox("Q类型", ["均匀", "单峰", "双峰", "自定义"])
        
        # 生成P分布
        if p_type == "均匀":
            P = np.ones(n_categories) / n_categories
        elif p_type == "单峰":
            peak = n_categories // 2
            P = np.exp(-((np.arange(n_categories) - peak)**2) / 2)
            P = P / P.sum()
        elif p_type == "双峰":
            P = np.zeros(n_categories)
            P[0] = 0.4
            P[-1] = 0.4
            if n_categories > 2:
                P[1:-1] = 0.2 / (n_categories - 2)
        else:  # 自定义
            st.markdown("##### P分布:")
            P = np.array([st.slider(f"P({i})", 0.0, 1.0, 1.0/n_categories, 0.01, 
                                   key=f"p_dist_{i}") for i in range(n_categories)])
            P = P / P.sum()
        
        # 生成Q分布
        if q_type == "均匀":
            Q = np.ones(n_categories) / n_categories
        elif q_type == "单峰":
            peak = n_categories // 2
            Q = np.exp(-((np.arange(n_categories) - peak)**2) / 2)
            Q = Q / Q.sum()
        elif q_type == "双峰":
            Q = np.zeros(n_categories)
            Q[0] = 0.4
            Q[-1] = 0.4
            if n_categories > 2:
                Q[1:-1] = 0.2 / (n_categories - 2)
        else:  # 自定义
            st.markdown("##### Q分布:")
            Q = np.array([st.slider(f"Q({i})", 0.0, 1.0, 1.0/n_categories, 0.01,
                                   key=f"q_dist_{i}") for i in range(n_categories)])
            Q = Q / Q.sum()
        
        # 计算KL散度
        kl_pq = np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))
        kl_qp = np.sum(Q * np.log((Q + 1e-10) / (P + 1e-10)))
        
        # JS散度 (对称版本)
        M = (P + Q) / 2
        js_divergence = 0.5 * np.sum(P * np.log((P + 1e-10) / (M + 1e-10))) + \
                       0.5 * np.sum(Q * np.log((Q + 1e-10) / (M + 1e-10)))
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("分布对比", "逐点贡献")
        )
        
        # 分布对比
        x = list(range(n_categories))
        fig.add_trace(go.Bar(x=x, y=P, name='P (真实)', marker_color='blue', opacity=0.7),
                     row=1, col=1)
        fig.add_trace(go.Bar(x=x, y=Q, name='Q (近似)', marker_color='red', opacity=0.7),
                     row=1, col=1)
        
        # KL散度的逐点贡献
        pointwise_kl = P * np.log((P + 1e-10) / (Q + 1e-10))
        fig.add_trace(go.Bar(x=x, y=pointwise_kl, name='P log(P/Q)', marker_color='green'),
                     row=1, col=2)
        
        fig.update_xaxes(title_text="类别", row=1, col=1)
        fig.update_xaxes(title_text="类别", row=1, col=2)
        fig.update_yaxes(title_text="概率", row=1, col=1)
        fig.update_yaxes(title_text="KL贡献", row=1, col=2)
        
        fig.update_layout(height=500, barmode='overlay')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("D_KL(P || Q)", f"{kl_pq:.4f} nats")
        with col2:
            st.metric("D_KL(Q || P)", f"{kl_qp:.4f} nats")
        with col3:
            st.metric("JS散度", f"{js_divergence:.4f} nats")
        
        # 非对称性演示
        if abs(kl_pq - kl_qp) > 0.01:
            st.warning(f"⚠️ **不对称性**: D_KL(P||Q) ≠ D_KL(Q||P), 差异 = {abs(kl_pq - kl_qp):.4f}")
        
        st.markdown("""
        ### 🔍 KL散度的应用
        
        **1. VAE (变分自编码器)**:
        $$\\mathcal{L} = \\mathbb{E}[\\log p(x|z)] - D_{KL}(q(z|x) \\| p(z))$$
        - 第一项: 重构损失
        - 第二项: KL散度正则化（使编码接近先验）
        
        **2. 强化学习 (TRPO, PPO)**:
        $$D_{KL}(\\pi_{old} \\| \\pi_{new}) < \\delta$$
        - 约束策略更新不要太激进
        
        **3. 知识蒸馏**:
        $$\\mathcal{L} = D_{KL}(P_{teacher} \\| P_{student})$$
        - 让小模型模仿大模型的输出分布
        
        **4. 贝叶斯推断**:
        - 用变分分布q(z)近似后验p(z|x)
        - 最小化 D_KL(q(z) || p(z|x))
        """)
    
    @staticmethod
    def _render_cross_entropy():
        """交叉熵与损失函数"""
        st.markdown("### 🎯 交叉熵: 机器学习的核心损失函数")
        
        st.latex(r"""
        H(P, Q) = -\sum_i P(i) \log Q(i)
        """)
        
        st.markdown("""
        **交叉熵与KL散度的关系**:
        $$H(P, Q) = H(P) + D_{KL}(P \\| Q)$$
        
        在分类问题中:
        - P: 真实标签分布 (one-hot)
        - Q: 模型预测分布 (softmax输出)
        - 最小化交叉熵 = 最小化KL散度
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 分类任务")
            n_classes = st.slider("类别数", 2, 10, 3)
            true_class = st.selectbox("真实类别", list(range(n_classes)))
            
            st.markdown("#### 模型预测")
            prediction_quality = st.slider("预测质量", 0.0, 1.0, 0.7, 0.05,
                                          help="1.0=完美预测, 0.0=随机猜测")
        
        # 真实分布 (one-hot)
        P = np.zeros(n_classes)
        P[true_class] = 1.0
        
        # 模型预测 (带噪声的softmax)
        Q = np.random.rand(n_classes)
        Q[true_class] = Q[true_class] + prediction_quality * 10
        Q = np.exp(Q) / np.sum(np.exp(Q))  # softmax
        
        # 计算损失
        cross_entropy = -np.sum(P * np.log(Q + 1e-10))
        entropy_p = 0  # one-hot的熵为0
        kl_div = cross_entropy - entropy_p
        
        # 预测准确性
        predicted_class = np.argmax(Q)
        is_correct = (predicted_class == true_class)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("真实 vs 预测分布", "交叉熵分解")
        )
        
        x = list(range(n_classes))
        
        # 分布对比
        fig.add_trace(go.Bar(x=x, y=P, name='真实 (P)', marker_color='blue', opacity=0.7),
                     row=1, col=1)
        fig.add_trace(go.Bar(x=x, y=Q, name='预测 (Q)', marker_color='red', opacity=0.7),
                     row=1, col=1)
        
        # 交叉熵分解
        components = [entropy_p, kl_div]
        labels = ['H(P)', 'D_KL(P||Q)']
        colors = ['blue', 'orange']
        
        fig.add_trace(go.Bar(x=labels, y=components, marker_color=colors,
                            text=[f'{c:.3f}' for c in components],
                            textposition='outside'),
                     row=1, col=2)
        
        fig.update_xaxes(title_text="类别", row=1, col=1)
        fig.update_xaxes(title_text="组成部分", row=1, col=2)
        fig.update_yaxes(title_text="概率", row=1, col=1)
        fig.update_yaxes(title_text="值 (nats)", row=1, col=2)
        
        fig.update_layout(height=500, barmode='group')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示结果
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("交叉熵损失", f"{cross_entropy:.4f}")
        with col2:
            st.metric("预测概率", f"{Q[true_class]:.2%}")
        with col3:
            st.metric("预测类别", f"{predicted_class}")
        with col4:
            if is_correct:
                st.metric("判断", "✅ 正确", delta="准确")
            else:
                st.metric("判断", "❌ 错误", delta="失败", delta_color="inverse")
        
        st.markdown(f"""
        ### 📊 损失分析
        
        **当前状态**:
        - 真实类别: {true_class}
        - 预测类别: {predicted_class}
        - 预测置信度: {Q[true_class]:.2%}
        - 交叉熵损失: {cross_entropy:.4f}
        
        **损失的含义**:
        - 如果模型预测完全正确 (Q[{true_class}]=1): 损失=0
        - 如果模型预测完全错误 (Q[{true_class}]→0): 损失→∞
        - 当前损失 {cross_entropy:.4f} 表示模型需要 {cross_entropy:.2f} nats 的"惊奇"
        
        **梯度方向**:
        $$\\frac{{\\partial L}}{{\\partial Q_i}} = -\\frac{{P_i}}{{Q_i}}$$
        
        对于正确类别 (i={true_class}):
        - 梯度 = -1/{Q[true_class]:.3f} = {-1/Q[true_class]:.3f}
        - 优化器会增大 Q[{true_class}]
        """)
        
        # 不同预测的损失对比
        st.markdown("### 📉 预测质量与损失的关系")
        
        confidences = np.linspace(0.01, 0.99, 100)
        losses = -np.log(confidences)
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=confidences, y=losses, mode='lines',
                                      line=dict(color='red', width=3),
                                      name='交叉熵损失'))
        
        # 标注当前点
        fig_loss.add_trace(go.Scatter(x=[Q[true_class]], y=[cross_entropy],
                                      mode='markers',
                                      marker=dict(size=15, color='blue'),
                                      name='当前状态'))
        
        fig_loss.update_layout(
            title="预测概率 vs 交叉熵损失",
            xaxis_title="对正确类别的预测概率",
            yaxis_title="交叉熵损失",
            height=400
        )
        
        st.plotly_chart(fig_loss, use_container_width=True)
        
        st.markdown("""
        **观察**:
        - 损失随着预测概率增加而快速下降
        - 在低概率区域，损失的梯度很大（学习快）
        - 在高概率区域，损失的梯度变小（学习慢）
        - 这就是为什么模型在"接近正确"时收敛变慢
        """)
    
    @staticmethod
    def _render_mutual_information():
        """互信息可视化"""
        st.markdown("### 🔗 互信息: 变量依赖性的度量")
        
        st.latex(r"""
        I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
        """)
        
        st.markdown("""
        **互信息的直觉**:
        - 测量知道X后，对Y的不确定性减少了多少
        - $I(X; Y) = H(Y) - H(Y|X)$
        - $I(X; Y) = H(X) + H(Y) - H(X,Y)$
        - 对称: $I(X; Y) = I(Y; X)$
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 相关性设置")
            correlation = st.slider("相关程度", 0.0, 1.0, 0.7, 0.05,
                                   help="0=独立, 1=完全相关")
        
        # 生成二维数据
        n = 1000
        X = np.random.randn(n)
        Y = correlation * X + np.sqrt(1 - correlation**2) * np.random.randn(n)
        
        # 离散化计算互信息
        n_bins = 10
        H_X, edges_X = np.histogram(X, bins=n_bins, density=True)
        H_Y, edges_Y = np.histogram(Y, bins=n_bins, density=True)
        H_X = H_X / H_X.sum()
        H_Y = H_Y / H_Y.sum()
        
        # 联合分布
        H_XY, _, _ = np.histogram2d(X, Y, bins=n_bins, density=True)
        H_XY = H_XY / H_XY.sum()
        
        # 计算熵和互信息
        entropy_X = -np.sum(H_X * np.log(H_X + 1e-10))
        entropy_Y = -np.sum(H_Y * np.log(H_Y + 1e-10))
        entropy_XY = -np.sum(H_XY * np.log(H_XY + 1e-10))
        mutual_info = entropy_X + entropy_Y - entropy_XY
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("联合分布", "信息关系")
        )
        
        # 联合分布热力图
        fig.add_trace(go.Heatmap(z=H_XY, colorscale='Blues', showscale=True),
                     row=1, col=1)
        
        # 信息关系（文氏图风格）
        info_data = {
            'H(X)': entropy_X,
            'H(Y)': entropy_Y,
            'H(X,Y)': entropy_XY,
            'I(X;Y)': mutual_info
        }
        
        fig.add_trace(go.Bar(
            x=list(info_data.keys()),
            y=list(info_data.values()),
            marker_color=['blue', 'red', 'purple', 'green'],
            text=[f'{v:.3f}' for v in info_data.values()],
            textposition='outside'
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        fig.update_xaxes(title_text="信息量", row=1, col=2)
        fig.update_yaxes(title_text="熵 (nats)", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("H(X)", f"{entropy_X:.3f} nats")
        with col2:
            st.metric("H(Y)", f"{entropy_Y:.3f} nats")
        with col3:
            st.metric("I(X; Y)", f"{mutual_info:.3f} nats")
        
        # 归一化互信息
        normalized_mi = mutual_info / min(entropy_X, entropy_Y) if min(entropy_X, entropy_Y) > 0 else 0
        
        st.markdown(f"""
        ### 📊 互信息分析
        
        **计算结果**:
        - 联合熵: H(X,Y) = {entropy_XY:.3f}
        - 互信息: I(X;Y) = {mutual_info:.3f}
        - 归一化互信息: {normalized_mi:.2%}
        
        **关系验证**:
        - H(X) + H(Y) = {entropy_X + entropy_Y:.3f}
        - H(X,Y) + I(X;Y) = {entropy_XY + mutual_info:.3f}
        - 应该相等: {'✅' if abs((entropy_X + entropy_Y) - (entropy_XY + mutual_info)) < 0.01 else '❌'}
        
        **在机器学习中的应用**:
        - **特征选择**: 选择与标签互信息高的特征
        - **信息瓶颈理论**: 神经网络层间的互信息演化
        - **对比学习**: 最大化正样本对的互信息
        - **生成模型**: 最大化生成数据与真实数据的互信息
        """)
    
    @staticmethod
    def _render_bayes():
        """贝叶斯推断可视化"""
        st.markdown("### 🎲 贝叶斯推断: 从先验到后验")
        
        st.latex(r"""
        P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}
        """)
        
        st.markdown("""
        **贝叶斯定理的组成**:
        - $P(\\theta)$: 先验 (Prior) - 观测数据前的信念
        - $P(D | \\theta)$: 似然 (Likelihood) - 数据在参数下的概率
        - $P(\\theta | D)$: 后验 (Posterior) - 观测数据后的更新信念
        - $P(D)$: 证据 (Evidence) - 归一化常数
        """)
        
        st.markdown("#### 示例: 硬币抛掷的贝叶斯推断")
        
        with st.sidebar:
            st.markdown("### 🎛️ 先验设置")
            prior_alpha = st.slider("先验 α", 0.5, 10.0, 1.0, 0.5)
            prior_beta = st.slider("先验 β", 0.5, 10.0, 1.0, 0.5)
            
            st.markdown("### 🎲 观测数据")
            n_heads = st.slider("正面次数", 0, 100, 7, 1)
            n_tails = st.slider("反面次数", 0, 100, 3, 1)
        
        # 先验: Beta分布
        theta = np.linspace(0, 1, 500)
        prior = stats.beta.pdf(theta, prior_alpha, prior_beta)
        
        # 似然: 二项分布
        likelihood = stats.binom.pmf(n_heads, n_heads + n_tails, theta)
        
        # 后验: Beta分布 (共轭先验)
        posterior_alpha = prior_alpha + n_heads
        posterior_beta = prior_beta + n_tails
        posterior = stats.beta.pdf(theta, posterior_alpha, posterior_beta)
        
        # 可视化
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=theta, y=prior, mode='lines',
                                name='先验 P(θ)', line=dict(color='blue', width=2)))
        
        fig.add_trace(go.Scatter(x=theta, y=likelihood / likelihood.max() * prior.max(),
                                name='似然 P(D|θ) (归一化)', 
                                line=dict(color='orange', width=2, dash='dash')))
        
        fig.add_trace(go.Scatter(x=theta, y=posterior, mode='lines',
                                name='后验 P(θ|D)', line=dict(color='red', width=3)))
        
        # 后验均值
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        fig.add_vline(x=posterior_mean, line_dash="dot", line_color="red",
                     annotation_text=f"后验均值 = {posterior_mean:.3f}")
        
        fig.update_layout(
            title=f"贝叶斯更新: 观测到 {n_heads} 正面, {n_tails} 反面",
            xaxis_title="θ (正面概率)",
            yaxis_title="密度",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示统计量
        prior_mean = prior_alpha / (prior_alpha + prior_beta)
        mle = n_heads / (n_heads + n_tails) if (n_heads + n_tails) > 0 else 0.5
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("先验均值", f"{prior_mean:.3f}")
        with col2:
            st.metric("MLE估计", f"{mle:.3f}")
        with col3:
            st.metric("后验均值", f"{posterior_mean:.3f}")
        
        st.markdown(f"""
        ### 🔍 贝叶斯 vs 频率派
        
        **频率派 (MLE)**:
        - 参数是固定但未知的常数
        - 估计: $\\hat{{\\theta}}_{{MLE}} = \\frac{{{n_heads}}}{{{n_heads + n_tails}}} = {mle:.3f}$
        - 只依赖数据，忽略先验知识
        
        **贝叶斯派**:
        - 参数是随机变量，有分布
        - 估计: $\\mathbb{{E}}[\\theta|D] = {posterior_mean:.3f}$
        - 结合先验知识和观测数据
        - 给出完整的后验分布，而不只是点估计
        
        **后验更新规则 (Beta-Binomial共轭)**:
        $$\\text{{Beta}}(\\alpha, \\beta) + \\text{{Data}}(h, t) \\to \\text{{Beta}}(\\alpha+h, \\beta+t)$$
        
        **观察**:
        - 数据量少时，先验影响大
        - 数据量多时，似然主导，贝叶斯→MLE
        - 当前: 先验({prior_alpha}, {prior_beta}) + 数据({n_heads}, {n_tails}) → 后验({posterior_alpha}, {posterior_beta})
        """)
