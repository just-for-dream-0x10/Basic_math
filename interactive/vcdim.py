"""
交互式VC维可视化
严格按照 7.VCdime.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveVCDim:
    """交互式VC维可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("📐 交互式VC维理论")
        st.markdown("""
        **VC维 (Vapnik-Chervonenkis Dimension)**: 度量模型复杂度的核心指标
        
        **定义**: 假设空间 $\\mathcal{H}$ 能够打散 (shatter) 的最大样本数量
        
        **打散 (Shattering)**: 对于 $n$ 个点，如果存在 $2^n$ 种标记方式都能被 $\\mathcal{H}$ 中某个函数实现，则称 $\\mathcal{H}$ 能打散这 $n$ 个点
        
        **VC界 (VC Bound)**:
        $$P(R(h) \\leq R_{emp}(h) + \\sqrt{\\frac{d(\\log(2n/d) + 1) - \\log(\\delta/4)}{n}})  \\geq 1 - \\delta$$
        
        其中:
        - $R(h)$: 泛化误差 (真实风险)
        - $R_{emp}(h)$: 经验误差 (训练误差)
        - $d$: VC维
        - $n$: 样本数量
        - $\\delta$: 置信度
        
        **关键结论**: 
        - VC维越大，模型容量越大，更容易过拟合
        - 需要的样本数 $n \\approx O(d/\\epsilon)$
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox("演示类型", [
                "打散演示 (Shattering)",
                "VC维计算",
                "VC界可视化",
                "模型复杂度对比",
                "样本复杂度曲线"
            ])
        
        if demo_type == "打散演示 (Shattering)":
            InteractiveVCDim._render_shattering()
        elif demo_type == "VC维计算":
            InteractiveVCDim._render_vc_calculation()
        elif demo_type == "VC界可视化":
            InteractiveVCDim._render_vc_bound()
        elif demo_type == "模型复杂度对比":
            InteractiveVCDim._render_model_comparison()
        elif demo_type == "样本复杂度曲线":
            InteractiveVCDim._render_sample_complexity()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("vcdim")
        quizzes = QuizTemplates.get_vcdim_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_shattering():
        """打散演示"""
        st.markdown("### 🎯 打散演示 (Shattering Demo)")
        
        st.markdown("""
        **目标**: 演示线性分类器在不同点数下的打散能力
        
        - 2个点: 可以打散 (VC维 ≥ 2)
        - 3个点: 可以打散 (VC维 ≥ 3)
        - 4个点: 不能打散 (VC维 = 3)
        """)
        
        with st.sidebar:
            n_points = st.radio("点的数量", [2, 3, 4])
            point_config = st.selectbox("点的配置", ["一般位置", "共线", "XOR"])
        
        # 生成点
        if point_config == "一般位置":
            if n_points == 2:
                points = np.array([[0, 0], [1, 1]])
            elif n_points == 3:
                points = np.array([[0, 0], [1, 0], [0.5, 1]])
            else:  # 4 points
                points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        elif point_config == "共线":
            points = np.array([[i, 0] for i in range(n_points)])
        else:  # XOR
            if n_points >= 4:
                points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            else:
                points = np.array([[0, 0], [1, 1], [0.5, 0.5]])[:n_points]
        
        # 生成所有可能的标记
        n_labelings = 2 ** n_points
        
        st.markdown(f"#### 📊 {n_points}个点的所有 {n_labelings} 种标记")
        
        # 创建子图
        cols_per_row = 4
        n_rows = (n_labelings + cols_per_row - 1) // cols_per_row
        
        fig = make_subplots(
            rows=n_rows, 
            cols=cols_per_row,
            subplot_titles=[f"标记 {i}" for i in range(n_labelings)],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        can_shatter = True
        unrealizable_count = 0
        
        for i in range(n_labelings):
            # 二进制标记
            labels = np.array([int(b) for b in format(i, f'0{n_points}b')])
            labels = labels * 2 - 1  # 转换为 {-1, 1}
            
            row = i // cols_per_row + 1
            col = i % cols_per_row + 1
            
            # 绘制点
            for label_val in [-1, 1]:
                mask = labels == label_val
                if np.any(mask):
                    color = 'red' if label_val == 1 else 'blue'
                    fig.add_trace(
                        go.Scatter(
                            x=points[mask, 0],
                            y=points[mask, 1],
                            mode='markers',
                            marker=dict(size=15, color=color, line=dict(color='black', width=1)),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
            
            # 尝试用线性分类器分类
            realizable = InteractiveVCDim._check_linear_separable(points, labels)
            
            if not realizable:
                can_shatter = False
                unrealizable_count += 1
                # 添加红叉标记
                fig.add_annotation(
                    text="✗",
                    xref=f"x{i+1}", yref=f"y{i+1}",
                    x=np.mean(points[:, 0]), y=np.mean(points[:, 1]),
                    showarrow=False,
                    font=dict(size=30, color="red"),
                    row=row, col=col
                )
        
        # 更新布局
        fig.update_xaxes(showticklabels=False, range=[-0.5, 1.5])
        fig.update_yaxes(showticklabels=False, range=[-0.5, 1.5])
        fig.update_layout(
            height=200 * n_rows,
            title_text=f"线性分类器{'能够' if can_shatter else '不能'}打散这{n_points}个点",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 结论
        st.markdown(f"""
        ### 📊 结果分析
        
        - 总标记数: {n_labelings}
        - 可实现: {n_labelings - unrealizable_count} 种
        - 不可实现: {unrealizable_count} 种
        - **结论**: 线性分类器 {'✅ 能够' if can_shatter else '❌ 不能'} 打散这 {n_points} 个点
        
        **VC维的含义**:
        - 线性分类器在 $\\mathbb{{R}}^2$ 的 VC维 = 3
        - 能打散任意3个点（非共线）
        - 不能打散某些4个点的配置（如XOR）
        """)
    
    @staticmethod
    def _check_linear_separable(points, labels):
        """检查是否线性可分"""
        from sklearn.svm import LinearSVC
        
        try:
            clf = LinearSVC(C=1e10, max_iter=10000)
            clf.fit(points, labels)
            predictions = clf.predict(points)
            return np.all(predictions == labels)
        except:
            return False
    
    @staticmethod
    def _render_vc_calculation():
        """VC维计算"""
        st.markdown("### 🧮 常见模型的VC维")
        
        st.markdown("""
        | 模型 | VC维 | 说明 |
        |------|------|------|
        | 线性分类器 ($\\mathbb{R}^d$) | $d + 1$ | 参数数量 |
        | 感知机 | $d + 1$ | 同线性分类器 |
        | 多项式分类器 (度$k$) | $\\binom{d+k}{k}$ | 组合数 |
        | 神经网络 (单隐层) | $O(VD)$ | $V$=参数数, $D$=输入维度 |
        | 决策树 (深度$h$) | $O(N \\log N)$ | $N$=节点数 |
        | kNN | $\\infty$ | 无限VC维 |
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 参数设置")
            model_type = st.selectbox("模型类型", [
                "线性分类器", 
                "多项式分类器", 
                "单隐层神经网络"
            ])
        
        if model_type == "线性分类器":
            d = st.slider("输入维度 d", 1, 10, 2)
            vc_dim = d + 1
            
            st.markdown(f"""
            #### 线性分类器
            
            **决策函数**: $f(x) = \\text{{sign}}(w^T x + b)$
            
            **参数**: $w \\in \\mathbb{{R}}^{d}, b \\in \\mathbb{{R}}$ (共 {d+1} 个参数)
            
            **VC维**: $d + 1 = {vc_dim}$
            
            **解释**: 
            - 在 $\\mathbb{{R}}^{d}$ 空间中，线性超平面由 $d+1$ 个参数确定
            - 可以打散任意 $d+1$ 个"一般位置"的点
            - 不能打散某些 $d+2$ 个点的配置
            """)
        
        elif model_type == "多项式分类器":
            d = st.slider("输入维度 d", 1, 5, 2)
            k = st.slider("多项式度数 k", 1, 4, 2)
            
            # 计算组合数
            from math import comb
            vc_dim = comb(d + k, k)
            
            st.markdown(f"""
            #### 多项式分类器 (度 {k})
            
            **特征映射**: $\\phi: \\mathbb{{R}}^{d} \\to \\mathbb{{R}}^{{{vc_dim}}}$
            
            例如 $d=2, k=2$:
            $$\\phi(x_1, x_2) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)$$
            
            **VC维**: $\\binom{{{d}+{k}}}{{{k}}} = {vc_dim}$
            
            **含义**: 
            - 通过升维可以增加模型容量
            - 但也增加了过拟合风险
            - 需要更多样本来保证泛化
            """)
        
        else:  # 单隐层神经网络
            d = st.slider("输入维度", 1, 10, 2)
            h = st.slider("隐层神经元数", 1, 20, 5)
            
            num_params = d * h + h + h + 1  # W1 + b1 + W2 + b2
            vc_dim_estimate = num_params * d
            
            st.markdown(f"""
            #### 单隐层神经网络
            
            **结构**: 输入({d}) → 隐层({h}) → 输出(1)
            
            **参数数量**: {num_params}
            - 第一层权重: {d} × {h} = {d*h}
            - 第一层偏置: {h}
            - 第二层权重: {h}
            - 第二层偏置: 1
            
            **VC维估计**: $O(VD) \\approx {vc_dim_estimate}$
            
            其中 $V = {num_params}$ (参数数), $D = {d}$ (输入维度)
            
            ⚠️ **注意**: 
            - 这只是上界估计
            - 实际VC维取决于激活函数和网络结构
            - ReLU网络的VC维分析更复杂
            """)
    
    @staticmethod
    def _render_vc_bound():
        """VC界可视化"""
        st.markdown("### 📈 VC界 (VC Bound) 可视化")
        
        st.latex(r"""
        R(h) \leq R_{emp}(h) + \sqrt{\frac{d(\log(2n/d) + 1) - \log(\delta/4)}{n}}
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 参数设置")
            vc_dim = st.slider("VC维 d", 1, 100, 10, 1)
            delta = st.slider("置信度 δ", 0.01, 0.2, 0.05, 0.01)
            emp_error = st.slider("经验误差", 0.0, 0.5, 0.1, 0.01)
        
        # 样本数量范围
        n_samples = np.logspace(np.log10(vc_dim), 4, 100).astype(int)
        
        # 计算VC界
        def vc_bound(n, d, delta):
            if n <= d:
                return 1.0  # 界失效
            term1 = d * (np.log(2 * n / d) + 1)
            term2 = np.log(delta / 4)
            return np.sqrt((term1 - term2) / n)
        
        bounds = np.array([vc_bound(n, vc_dim, delta) for n in n_samples])
        generalization_error = bounds
        true_risk = emp_error + generalization_error
        
        # 创建图表
        fig = go.Figure()
        
        # 经验误差（训练误差）
        fig.add_trace(go.Scatter(
            x=n_samples,
            y=[emp_error] * len(n_samples),
            mode='lines',
            name='经验误差 R_emp',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # 泛化误差上界
        fig.add_trace(go.Scatter(
            x=n_samples,
            y=true_risk,
            mode='lines',
            name='泛化误差上界 R(h)',
            line=dict(color='red', width=3),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        # VC项
        fig.add_trace(go.Scatter(
            x=n_samples,
            y=generalization_error,
            mode='lines',
            name='VC项 (泛化gap)',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title=f"VC界: d={vc_dim}, δ={delta}, R_emp={emp_error}",
            xaxis_title="样本数量 n",
            yaxis_title="误差",
            xaxis_type="log",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 关键观察
        st.markdown("""
        ### 🔍 关键观察
        
        1. **样本数量的作用**:
           - $n$ 越大，泛化gap越小 ($O(1/\\sqrt{n})$)
           - 需要 $n \\gg d$ 才能保证good generalization
        
        2. **VC维的影响**:
           - $d$ 越大，需要更多样本
           - 过复杂的模型容易过拟合
        
        3. **trade-off**:
           - 模型太简单 → 高偏差（欠拟合）
           - 模型太复杂 → 高方差（过拟合）
        
        4. **实践建议**:
           - 样本数 $n \\approx 10d$ (经验法则)
           - 使用交叉验证选择模型复杂度
           - 正则化可以有效控制VC维
        """)
    
    @staticmethod
    def _render_model_comparison():
        """模型复杂度对比"""
        st.markdown("### 🔍 不同模型的VC维对比")
        
        d = st.sidebar.slider("特征维度 d", 2, 10, 5)
        
        # 计算不同模型的VC维
        from math import comb
        
        models = {
            "线性": d + 1,
            "2次多项式": comb(d + 2, 2),
            "3次多项式": comb(d + 3, 3),
            "5层神经网络(10单元)": (d * 10 + 10) * 5 * d,
            "10层神经网络(20单元)": (d * 20 + 20) * 10 * d,
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(models.keys()),
            y=list(models.values()),
            marker_color=['blue', 'green', 'orange', 'red', 'purple'],
            text=[f"{v:,}" for v in models.values()],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"模型VC维对比 (特征维度 d={d})",
            xaxis_title="模型类型",
            yaxis_title="VC维",
            yaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 所需样本数
        st.markdown("### 📊 所需样本数估计 (n ≈ 10d)")
        
        sample_needs = {name: 10 * vc for name, vc in models.items()}
        
        col1, col2 = st.columns(2)
        with col1:
            for name, samples in list(sample_needs.items())[:3]:
                st.metric(name, f"{samples:,} 样本")
        with col2:
            for name, samples in list(sample_needs.items())[3:]:
                st.metric(name, f"{samples:,} 样本")
        
        st.warning("""
        ⚠️ **深度神经网络的悖论**:
        - 理论VC维极高 → 应该严重过拟合
        - 实践中却泛化很好 → 为什么？
        
        **可能的解释**:
        - 隐式正则化 (Implicit Regularization)
        - 梯度下降的归纳偏置 (Inductive Bias)
        - 过参数化的好处 (Overparameterization)
        - VC维可能不是最好的度量
        """)
    
    @staticmethod
    def _render_sample_complexity():
        """样本复杂度曲线"""
        st.markdown("### 📉 学习曲线: 样本复杂度")
        
        st.markdown("""
        **PAC学习理论**: 要达到 $(\\epsilon, \\delta)$ 学习，需要的样本数:
        
        $$n \\geq \\frac{1}{\\epsilon} \\left( d \\log \\frac{1}{\\epsilon} + \\log \\frac{1}{\\delta} \\right)$$
        """)
        
        with st.sidebar:
            vc_dim = st.slider("VC维", 1, 50, 10)
            epsilon = st.slider("目标误差 ε", 0.01, 0.3, 0.1, 0.01)
        
        # 不同置信度
        deltas = [0.01, 0.05, 0.1, 0.2]
        
        fig = go.Figure()
        
        d_range = np.arange(1, 51)
        
        for delta in deltas:
            n_required = (1/epsilon) * (d_range * np.log(1/epsilon) + np.log(1/delta))
            
            fig.add_trace(go.Scatter(
                x=d_range,
                y=n_required,
                mode='lines',
                name=f'δ = {delta}',
                line=dict(width=2)
            ))
        
        # 添加当前VC维的垂直线
        fig.add_vline(x=vc_dim, line_dash="dash", line_color="red",
                     annotation_text=f"当前VC维 = {vc_dim}")
        
        fig.update_layout(
            title=f"样本复杂度 vs VC维 (ε = {epsilon})",
            xaxis_title="VC维 d",
            yaxis_title="所需样本数 n",
            yaxis_type="log",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 计算当前配置需要的样本数
        current_n = (1/epsilon) * (vc_dim * np.log(1/epsilon) + np.log(1/0.05))
        
        st.markdown(f"""
        ### 📊 当前配置
        
        - VC维: {vc_dim}
        - 目标误差: ε = {epsilon}
        - 置信度: δ = 0.05
        
        **所需样本数**: $n \\geq {current_n:,.0f}$
        
        **含义**:
        - 要保证误差 < {epsilon}，至少需要 {current_n:,.0f} 个训练样本
        - 这是理论上界，实践中可能需要更少（归纳偏置）
        - 也可能需要更多（数据噪声、分布不匹配）
        """)
