"""
交互式噪声可视化
严格按照 9.noise.md 中的理论实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveNoise:
    """交互式噪声理论可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🔊 交互式噪声理论")
        st.markdown("""
        **噪声 (Noise)**: 数据中不可预测的随机成分
        
        **数学表示**: 
        $$Y = f(X) + \\epsilon$$
        
        其中:
        - $Y$: 观测值
        - $f(X)$: 真实函数（确定性部分）
        - $\\epsilon$: 噪声项，$\\mathbb{E}[\\epsilon] = 0$, $\\text{Var}(\\epsilon) = \\sigma^2$
        
        **核心洞察**:
        - 噪声不是过拟合的原因，而是背景因素
        - 模型复杂度决定是否会拟合噪声
        - 数据量越大，噪声影响越小
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox("演示类型", [
                "噪声的本质理解",
                "过拟合与噪声",
                "训练误差vs测试误差",
                "学习曲线分析",
                "模型复杂度三角平衡",
                "噪声鲁棒性策略"
            ])
        
        if demo_type == "噪声的本质理解":
            InteractiveNoise._render_noise_nature()
        elif demo_type == "过拟合与噪声":
            InteractiveNoise._render_overfitting()
        elif demo_type == "训练误差vs测试误差":
            InteractiveNoise._render_train_test_error()
        elif demo_type == "学习曲线分析":
            InteractiveNoise._render_learning_curves()
        elif demo_type == "模型复杂度三角平衡":
            InteractiveNoise._render_triangle_balance()
        elif demo_type == "噪声鲁棒性策略":
            InteractiveNoise._render_robustness()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("noise")
        quizzes = QuizTemplates.get_noise_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_noise_nature():
        """演示噪声的本质"""
        st.markdown("### 🎯 噪声的本质理解")
        st.markdown("""
        **噪声 = 任何不能被模型捕捉的、随机的、不可预测的变化**
        
        让我们通过一个简单例子理解：假设真实关系是 $y = 2x + 1$
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_samples = st.slider("样本数量", 20, 200, 50, 10)
            noise_std = st.slider("噪声标准差 σ", 0.0, 5.0, 1.0, 0.1)
            show_true_function = st.checkbox("显示真实函数", True)
        
        # 生成数据
        np.random.seed(42)
        X = np.linspace(0, 10, n_samples)
        true_y = 2 * X + 1  # 真实函数
        noise = np.random.normal(0, noise_std, n_samples)  # 噪声
        observed_y = true_y + noise  # 观测值
        
        # 创建可视化
        fig = go.Figure()
        
        # 真实函数
        if show_true_function:
            fig.add_trace(go.Scatter(
                x=X, y=true_y,
                mode='lines',
                name='真实函数 f(X) = 2X + 1',
                line=dict(color='green', width=3, dash='dash')
            ))
        
        # 观测数据
        fig.add_trace(go.Scatter(
            x=X, y=observed_y,
            mode='markers',
            name='观测值 Y = f(X) + ε',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # 噪声可视化（垂直线）
        for i in range(0, n_samples, max(1, n_samples // 20)):
            fig.add_trace(go.Scatter(
                x=[X[i], X[i]],
                y=[true_y[i], observed_y[i]],
                mode='lines',
                line=dict(color='red', width=1, dash='dot'),
                showlegend=(i == 0),
                name='噪声 ε' if i == 0 else None
            ))
        
        fig.update_layout(
            title=f"噪声的本质：Y = f(X) + ε (σ = {noise_std:.1f})",
            xaxis_title="X",
            yaxis_title="Y",
            height=500,
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("噪声方差 σ²", f"{noise_std**2:.2f}")
        with col2:
            st.metric("实际噪声方差", f"{np.var(noise):.2f}")
        with col3:
            st.metric("信噪比 SNR", f"{np.var(true_y) / (noise_std**2 + 1e-10):.2f}")
        
        st.markdown("""
        **观察**:
        - 🟢 绿色虚线：真实函数（确定性部分）
        - 🔵 蓝色点：实际观测到的数据
        - 🔴 红色虚线：噪声（偏离真实值的随机误差）
        
        **关键理解**:
        1. 噪声是数据生成过程的一部分，无法消除
        2. 噪声使得即使知道真实函数，预测也不可能完美
        3. $\\sigma^2$ 是模型误差的理论下界
        """)
    
    @staticmethod
    def _render_overfitting():
        """演示过拟合与噪声的关系"""
        st.markdown("### 🎪 过拟合：模型学习了噪声")
        st.markdown("""
        **核心问题**: 过拟合的原因是**模型太复杂**，而非噪声本身
        
        **比喻**: 
        - 真实规律 = 老师讲的知识点
        - 噪声 = 老师的口误、咳嗽
        - 过拟合 = 学生连口误都背下来了
        
        问题不在口误，而在学生"太认真"（模型太复杂）
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            noise_level = st.slider("噪声水平", 0.1, 3.0, 1.0, 0.1)
            model_degree = st.slider("多项式阶数（模型复杂度）", 1, 15, 3, 1)
            n_samples = st.slider("训练样本数", 10, 100, 30, 10)
        
        # 生成数据
        np.random.seed(42)
        X_train = np.sort(np.random.uniform(0, 10, n_samples))
        true_y_train = np.sin(X_train) * 2  # 真实函数：正弦波
        y_train = true_y_train + np.random.normal(0, noise_level, n_samples)
        
        # 测试数据（密集，用于可视化）
        X_test = np.linspace(0, 10, 200)
        true_y_test = np.sin(X_test) * 2
        
        # 训练模型
        model = make_pipeline(PolynomialFeatures(model_degree), Ridge(alpha=0.01))
        model.fit(X_train.reshape(-1, 1), y_train)
        y_pred_train = model.predict(X_train.reshape(-1, 1))
        y_pred_test = model.predict(X_test.reshape(-1, 1))
        
        # 计算误差
        train_error = np.mean((y_train - y_pred_train) ** 2)
        true_train_error = np.mean((true_y_train - y_pred_train) ** 2)
        test_error = np.mean((true_y_test - y_pred_test) ** 2)
        
        # 可视化
        fig = go.Figure()
        
        # 真实函数
        fig.add_trace(go.Scatter(
            x=X_test, y=true_y_test,
            mode='lines',
            name='真实函数 f(X)',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # 训练数据
        fig.add_trace(go.Scatter(
            x=X_train, y=y_train,
            mode='markers',
            name='训练数据（含噪声）',
            marker=dict(color='blue', size=10, symbol='circle')
        ))
        
        # 模型预测
        fig.add_trace(go.Scatter(
            x=X_test, y=y_pred_test,
            mode='lines',
            name=f'模型预测（阶数={model_degree}）',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"过拟合演示：多项式阶数={model_degree}",
            xaxis_title="X",
            yaxis_title="Y",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 误差分析
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练误差", f"{train_error:.4f}", 
                     help="模型在训练数据上的MSE")
        with col2:
            st.metric("测试误差", f"{test_error:.4f}",
                     help="模型在真实函数上的MSE")
        with col3:
            overfitting = test_error - train_error
            st.metric("过拟合程度", f"{overfitting:.4f}",
                     delta=f"{overfitting:.4f}",
                     delta_color="inverse")
        
        # 过拟合诊断
        if model_degree <= 3:
            st.success("✅ 模型复杂度适中，欠拟合或刚好")
        elif model_degree <= 7:
            st.warning("⚠️ 模型开始拟合噪声，注意过拟合风险")
        else:
            st.error("❌ 模型严重过拟合，学习了训练数据中的噪声！")
        
        st.markdown("""
        **关键观察**:
        1. **低阶多项式**（如1-3阶）：模型太简单，欠拟合，无法学习噪声
        2. **中阶多项式**（如4-7阶）：模型适中，泛化较好
        3. **高阶多项式**（如8-15阶）：模型过于复杂，开始拟合噪声
        
        **结论**: 过拟合是"模型容量过剩"的结果，噪声只是被拟合的对象
        """)
    
    @staticmethod
    def _render_train_test_error():
        """训练误差vs测试误差分析"""
        st.markdown("### 📊 训练误差 vs 测试误差")
        st.markdown("""
        **核心公式**（来自线性回归理论）：
        
        训练误差期望：
        $$\\mathbb{E}[E_{\\text{in}}] = \\sigma^2 \\left(1 - \\frac{d+1}{N}\\right)$$
        
        测试误差期望：
        $$\\mathbb{E}[E_{\\text{out}}] = \\sigma^2 \\left(1 + \\frac{d+1}{N}\\right)$$
        
        其中：
        - $\\sigma^2$: 噪声方差
        - $d$: 模型维度（参数数量-1）
        - $N$: 训练样本数量
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            noise_var = st.slider("噪声方差 σ²", 0.1, 5.0, 1.0, 0.1)
            model_dim = st.slider("模型维度 d", 1, 20, 5, 1)
            max_samples = st.slider("最大样本数", 50, 500, 200, 50)
        
        # 样本数量范围
        N_range = np.arange(model_dim + 2, max_samples, 2)
        
        # 计算理论值
        E_in_theory = noise_var * (1 - (model_dim + 1) / N_range)
        E_out_theory = noise_var * (1 + (model_dim + 1) / N_range)
        
        # 创建可视化
        fig = go.Figure()
        
        # 噪声水平基线
        fig.add_hline(y=noise_var, 
                     line_dash="dash", 
                     line_color="gray",
                     annotation_text=f"噪声方差 σ² = {noise_var}",
                     annotation_position="right")
        
        # 训练误差
        fig.add_trace(go.Scatter(
            x=N_range, y=E_in_theory,
            mode='lines',
            name='训练误差 E_in',
            line=dict(color='blue', width=3),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        # 测试误差
        fig.add_trace(go.Scatter(
            x=N_range, y=E_out_theory,
            mode='lines',
            name='测试误差 E_out',
            line=dict(color='red', width=3),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        
        # 标注关键点
        critical_n = model_dim * 10  # 一般认为 N = 10d 是比较好的点
        if critical_n < max_samples:
            idx = np.argmin(np.abs(N_range - critical_n))
            fig.add_vline(x=N_range[idx],
                         line_dash="dot",
                         line_color="green",
                         annotation_text=f"N ≈ 10d = {critical_n}",
                         annotation_position="top")
        
        fig.update_layout(
            title=f"训练误差 vs 测试误差随样本数变化 (d={model_dim}, σ²={noise_var})",
            xaxis_title="训练样本数量 N",
            yaxis_title="误差 (MSE)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        # 找到 N=10d 附近的值
        target_n = model_dim * 10
        if target_n < max_samples:
            idx = np.argmin(np.abs(N_range - target_n))
            with col1:
                st.metric("当 N=10d 时", f"N={N_range[idx]}")
            with col2:
                st.metric("训练误差", f"{E_in_theory[idx]:.3f}")
            with col3:
                st.metric("测试误差", f"{E_out_theory[idx]:.3f}")
            with col4:
                gap = E_out_theory[idx] - E_in_theory[idx]
                st.metric("泛化间隙", f"{gap:.3f}")
        
        st.markdown("""
        **关键观察**:
        
        1. **训练误差 < 噪声方差**: 
           - 模型对训练集噪声进行了拟合
           - $E_{\\text{in}}$ 从下方逼近 $\\sigma^2$
        
        2. **测试误差 > 噪声方差**: 
           - 训练集噪声的拟合无法泛化
           - $E_{\\text{out}}$ 从上方逼近 $\\sigma^2$
        
        3. **随着 N 增大**:
           - 两者都收敛到 $\\sigma^2$
           - 泛化间隙 $\\propto \\frac{d+1}{N}$
        
        4. **当 N ≈ 10d 时**:
           - 通常认为是较好的样本量
           - 模型能较好地学习规律而非噪声
        """)
    
    @staticmethod
    def _render_learning_curves():
        """学习曲线分析"""
        st.markdown("### 📈 学习曲线：模型如何学习")
        st.markdown("""
        **学习曲线**展示了随着训练样本增加，模型性能的变化
        
        **典型形态**:
        - 训练误差 ↗ 逐渐上升
        - 测试误差 ↘ 逐渐下降
        - 最终都收敛到噪声水平 $\\sigma^2$
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            true_function = st.selectbox("真实函数", ["线性", "二次", "正弦"])
            noise_level = st.slider("噪声水平", 0.1, 2.0, 0.5, 0.1)
            model_complexity = st.slider("模型复杂度", 1, 10, 3, 1)
        
        # 生成真实函数
        def get_true_function(X, func_type):
            if func_type == "线性":
                return 2 * X + 1
            elif func_type == "二次":
                return 0.5 * X**2 - X + 1
            else:  # 正弦
                return 2 * np.sin(X)
        
        # 不同样本数量下的误差
        sample_sizes = np.arange(10, 201, 10)
        train_errors = []
        test_errors = []
        
        # 固定测试集
        np.random.seed(42)
        X_test = np.linspace(0, 10, 200)
        y_test = get_true_function(X_test, true_function)
        
        for n in sample_sizes:
            # 生成训练数据
            X_train = np.sort(np.random.uniform(0, 10, n))
            y_train_true = get_true_function(X_train, true_function)
            y_train = y_train_true + np.random.normal(0, noise_level, n)
            
            # 训练模型
            model = make_pipeline(
                PolynomialFeatures(model_complexity), 
                Ridge(alpha=0.1)
            )
            model.fit(X_train.reshape(-1, 1), y_train)
            
            # 计算误差
            y_pred_train = model.predict(X_train.reshape(-1, 1))
            y_pred_test = model.predict(X_test.reshape(-1, 1))
            
            train_errors.append(np.mean((y_train - y_pred_train)**2))
            test_errors.append(np.mean((y_test - y_pred_test)**2))
        
        # 可视化
        fig = go.Figure()
        
        # 噪声水平
        fig.add_hline(y=noise_level**2,
                     line_dash="dash",
                     line_color="gray",
                     annotation_text=f"噪声方差 σ² = {noise_level**2:.2f}",
                     annotation_position="right")
        
        # 训练误差曲线
        fig.add_trace(go.Scatter(
            x=sample_sizes, y=train_errors,
            mode='lines+markers',
            name='训练误差',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # 测试误差曲线
        fig.add_trace(go.Scatter(
            x=sample_sizes, y=test_errors,
            mode='lines+markers',
            name='测试误差',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
        
        # 填充间隙
        fig.add_trace(go.Scatter(
            x=list(sample_sizes) + list(sample_sizes[::-1]),
            y=list(train_errors) + list(test_errors[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='泛化间隙',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"学习曲线 (模型复杂度={model_complexity}, 噪声={noise_level})",
            xaxis_title="训练样本数量",
            yaxis_title="均方误差 (MSE)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        final_train_error = train_errors[-1]
        final_test_error = test_errors[-1]
        final_gap = final_test_error - final_train_error
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最终训练误差", f"{final_train_error:.3f}")
        with col2:
            st.metric("最终测试误差", f"{final_test_error:.3f}")
        with col3:
            st.metric("最终泛化间隙", f"{final_gap:.3f}")
        
        # 诊断
        if final_gap > noise_level**2 * 0.5:
            st.warning("⚠️ 泛化间隙较大，可能存在过拟合")
        elif final_train_error > noise_level**2 * 2:
            st.warning("⚠️ 训练误差较大，可能存在欠拟合")
        else:
            st.success("✅ 模型训练良好，泛化性能较优")
        
        st.markdown("""
        **学习曲线解读**:
        
        1. **初期** (样本少):
           - 训练误差很低（模型记住了所有数据）
           - 测试误差很高（严重过拟合）
           - 泛化间隙很大
        
        2. **中期** (样本增多):
           - 训练误差上升（无法完美拟合所有数据）
           - 测试误差下降（学到了真实规律）
           - 泛化间隙缩小
        
        3. **后期** (样本充足):
           - 两条曲线收敛
           - 都接近噪声方差 $\\sigma^2$
           - 继续增加数据收益递减
        """)
    
    @staticmethod
    def _render_triangle_balance():
        """模型复杂度、数据量、噪声的三角平衡"""
        st.markdown("### ⚖️ 学习的三角平衡")
        st.markdown("""
        **泛化误差分解**（简化版）:
        
        $$R \\approx R_{\\text{emp}} + \\sqrt{\\frac{d_{VC}}{N}} + \\sigma^2$$
        
        | 要素 | 影响 | 控制手段 |
        |------|------|----------|
        | 模型复杂度 $d$ | 容量越大，越容易过拟合 | 正则化、剪枝、早停 |
        | 数据量 $N$ | 数据越多，泛化越好 | 数据增强、主动学习 |
        | 噪声 $\\sigma^2$ | 性能下界 | 数据清洗、鲁棒损失 |
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            vc_dim = st.slider("VC维 d", 1, 50, 10, 1)
            sample_size = st.slider("样本数 N", 10, 1000, 100, 10)
            noise_var = st.slider("噪声方差 σ²", 0.1, 5.0, 1.0, 0.1)
        
        # 计算泛化误差的各个组成部分
        empirical_risk = 0.1  # 假设经验风险很小
        complexity_penalty = np.sqrt(vc_dim / sample_size)
        noise_bound = noise_var
        total_risk = empirical_risk + complexity_penalty + noise_bound
        
        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=['经验风险', '复杂度惩罚', '噪声下界'],
            values=[empirical_risk, complexity_penalty, noise_bound],
            marker=dict(colors=['#3B82F6', '#F59E0B', '#EF4444']),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=f"泛化误差分解 (d={vc_dim}, N={sample_size}, σ²={noise_var})",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 指标展示
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("经验风险", f"{empirical_risk:.3f}")
        with col2:
            st.metric("复杂度惩罚", f"{complexity_penalty:.3f}")
        with col3:
            st.metric("噪声下界", f"{noise_bound:.3f}")
        with col4:
            st.metric("总泛化误差", f"{total_risk:.3f}")
        
        # 比例分析
        st.markdown("#### 📊 误差来源分析")
        
        complexity_ratio = complexity_penalty / total_risk
        noise_ratio = noise_bound / total_risk
        
        if noise_ratio > 0.6:
            st.info(f"🔊 **噪声主导** ({noise_ratio*100:.1f}%): 数据质量是主要瓶颈，考虑数据清洗或鲁棒方法")
        elif complexity_ratio > 0.4:
            st.warning(f"📐 **复杂度主导** ({complexity_ratio*100:.1f}%): 模型过于复杂或数据不足，考虑正则化或增加数据")
        else:
            st.success("✅ **平衡状态**: 模型、数据、噪声三者较为平衡")
        
        # 建议
        st.markdown("#### 💡 优化建议")
        
        suggestions = []
        
        if vc_dim / sample_size > 0.1:
            suggestions.append("⚠️ **样本不足**: $N/d$ 比例较低，建议增加数据量或降低模型复杂度")
        
        if noise_var > 2.0:
            suggestions.append("🔊 **噪声过大**: 考虑数据清洗、特征工程或使用鲁棒损失函数")
        
        if sample_size < vc_dim * 10:
            suggestions.append("📊 **经验法则**: 通常建议 $N \\geq 10d$，当前 $N/d = {:.1f}$".format(sample_size/vc_dim))
        
        if len(suggestions) == 0:
            st.success("✅ 当前配置合理，继续保持！")
        else:
            for sugg in suggestions:
                st.markdown(sugg)
    
    @staticmethod
    def _render_robustness():
        """噪声鲁棒性策略"""
        st.markdown("### 🛡️ 噪声鲁棒性策略")
        st.markdown("""
        **目标**: 让模型"学会跳过"噪声，只学习真实规律
        
        **核心策略**:
        1. **正则化** (L1/L2): 限制模型复杂度
        2. **早停** (Early Stopping): 防止过度训练
        3. **Dropout**: 随机失活，增强泛化
        4. **数据增强**: 增加样本多样性
        5. **鲁棒损失**: 对异常值不敏感
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            strategy = st.selectbox("选择策略", [
                "正则化效果",
                "早停演示",
                "鲁棒损失对比"
            ])
        
        if strategy == "正则化效果":
            InteractiveNoise._render_regularization_effect()
        elif strategy == "早停演示":
            InteractiveNoise._render_early_stopping()
        else:
            InteractiveNoise._render_robust_loss()
    
    @staticmethod
    def _render_regularization_effect():
        """正则化效果演示"""
        st.markdown("#### 🎯 正则化：限制模型复杂度")
        
        with st.sidebar:
            noise_level = st.slider("噪声水平", 0.1, 2.0, 0.8, 0.1)
            alpha = st.slider("正则化强度 α", 0.0, 10.0, 1.0, 0.1)
            n_samples = st.slider("训练样本", 20, 100, 30, 10)
        
        # 生成数据
        np.random.seed(42)
        X_train = np.sort(np.random.uniform(0, 10, n_samples))
        true_y = np.sin(X_train) * 2
        y_train = true_y + np.random.normal(0, noise_level, n_samples)
        
        X_test = np.linspace(0, 10, 200)
        y_test_true = np.sin(X_test) * 2
        
        # 训练三个模型：无正则化、弱正则化、强正则化
        models = {
            '无正则化 (α=0)': Ridge(alpha=0.001),
            f'适度正则化 (α={alpha})': Ridge(alpha=alpha),
            '强正则化 (α=10)': Ridge(alpha=10.0)
        }
        
        fig = go.Figure()
        
        # 真实函数
        fig.add_trace(go.Scatter(
            x=X_test, y=y_test_true,
            mode='lines',
            name='真实函数',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # 训练数据
        fig.add_trace(go.Scatter(
            x=X_train, y=y_train,
            mode='markers',
            name='训练数据',
            marker=dict(color='gray', size=8, opacity=0.5)
        ))
        
        colors = ['red', 'blue', 'orange']
        train_errors = []
        test_errors = []
        
        for (name, model), color in zip(models.items(), colors):
            # 使用高阶多项式
            poly_model = make_pipeline(PolynomialFeatures(10), model)
            poly_model.fit(X_train.reshape(-1, 1), y_train)
            
            y_pred = poly_model.predict(X_test.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(
                x=X_test, y=y_pred,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
            
            # 计算误差
            train_pred = poly_model.predict(X_train.reshape(-1, 1))
            train_errors.append(np.mean((y_train - train_pred)**2))
            test_errors.append(np.mean((y_test_true - y_pred)**2))
        
        fig.update_layout(
            title="正则化效果对比",
            xaxis_title="X",
            yaxis_title="Y",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 误差对比
        st.markdown("#### 📊 误差对比")
        
        col1, col2, col3 = st.columns(3)
        for i, (name, _) in enumerate(models.items()):
            with [col1, col2, col3][i]:
                st.metric(name, f"测试误差: {test_errors[i]:.3f}")
                st.caption(f"训练误差: {train_errors[i]:.3f}")
        
        st.markdown("""
        **观察**:
        - 🔴 **无正则化**: 严重过拟合，曲线剧烈震荡
        - 🔵 **适度正则化**: 平滑曲线，泛化最好
        - 🟠 **强正则化**: 曲线过于平滑，可能欠拟合
        
        **结论**: 正则化通过惩罚模型复杂度，迫使模型忽略噪声
        """)
    
    @staticmethod
    def _render_early_stopping():
        """早停演示"""
        st.markdown("#### ⏱️ 早停：在合适的时机停止训练")
        
        st.markdown("""
        **原理**: 监控验证集误差，当其不再下降时停止训练
        
        **类比**: 学生做题，做到一定程度就够了，继续做反而会"背题"
        """)
        
        with st.sidebar:
            noise_level = st.slider("噪声水平", 0.1, 2.0, 0.5, 0.1)
            max_epochs = st.slider("最大训练轮数", 50, 500, 200, 50)
        
        # 生成数据
        np.random.seed(42)
        n_train = 50
        X_train = np.sort(np.random.uniform(0, 10, n_train))
        true_y_train = np.sin(X_train) * 2
        y_train = true_y_train + np.random.normal(0, noise_level, n_train)
        
        X_val = np.sort(np.random.uniform(0, 10, 30))
        true_y_val = np.sin(X_val) * 2
        y_val = true_y_val + np.random.normal(0, noise_level, 30)
        
        # 模拟训练过程（通过增加模型复杂度模拟训练）
        epochs = np.arange(1, max_epochs + 1)
        train_errors = []
        val_errors = []
        
        for epoch in epochs:
            # 复杂度随epoch增加
            degree = min(1 + epoch // 20, 15)
            model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.01))
            model.fit(X_train.reshape(-1, 1), y_train)
            
            train_pred = model.predict(X_train.reshape(-1, 1))
            val_pred = model.predict(X_val.reshape(-1, 1))
            
            train_errors.append(np.mean((y_train - train_pred)**2))
            val_errors.append(np.mean((y_val - val_pred)**2))
        
        # 找到最佳停止点（验证误差最小）
        best_epoch = np.argmin(val_errors) + 1
        
        # 可视化
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs, y=train_errors,
            mode='lines',
            name='训练误差',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=val_errors,
            mode='lines',
            name='验证误差',
            line=dict(color='red', width=2)
        ))
        
        # 标注最佳停止点
        fig.add_vline(x=best_epoch,
                     line_dash="dash",
                     line_color="green",
                     annotation_text=f"最佳停止点 (epoch={best_epoch})",
                     annotation_position="top")
        
        # 标注过拟合区域
        fig.add_vrect(x0=best_epoch, x1=max_epochs,
                     fillcolor="red", opacity=0.1,
                     annotation_text="过拟合区域",
                     annotation_position="top right")
        
        fig.update_layout(
            title="早停演示：训练误差 vs 验证误差",
            xaxis_title="训练轮数 (Epoch)",
            yaxis_title="误差 (MSE)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最佳停止点", f"Epoch {best_epoch}")
        with col2:
            st.metric("最佳验证误差", f"{val_errors[best_epoch-1]:.4f}")
        with col3:
            final_val_error = val_errors[-1]
            improvement = final_val_error - val_errors[best_epoch-1]
            st.metric("过拟合损失", f"+{improvement:.4f}", 
                     delta=f"{improvement:.4f}", delta_color="inverse")
        
        st.markdown("""
        **关键观察**:
        
        1. **初期**: 训练误差和验证误差都下降（学习规律）
        2. **最佳点**: 验证误差达到最小值
        3. **过拟合区**: 训练误差继续下降，验证误差上升（学习噪声）
        
        **策略**: 在验证误差不再改善时（通常观察5-10个epoch），停止训练
        """)
    
    @staticmethod
    def _render_robust_loss():
        """鲁棒损失函数对比"""
        st.markdown("#### 📐 鲁棒损失：对异常值不敏感")
        
        st.markdown("""
        **问题**: 平方损失 (MSE) 对异常值非常敏感
        
        **解决方案**: 使用鲁棒损失函数
        - **Huber Loss**: 在小误差时用L2，大误差时用L1
        - **MAE (L1)**: 对所有误差线性惩罚
        """)
        
        with st.sidebar:
            delta = st.slider("Huber δ 参数", 0.5, 5.0, 1.0, 0.5)
            outlier_ratio = st.slider("异常值比例", 0.0, 0.3, 0.1, 0.05)
        
        # 生成误差范围
        errors = np.linspace(-5, 5, 200)
        
        # 不同损失函数
        mse_loss = errors ** 2
        mae_loss = np.abs(errors)
        huber_loss = np.where(
            np.abs(errors) <= delta,
            0.5 * errors ** 2,
            delta * np.abs(errors) - 0.5 * delta ** 2
        )
        
        # 可视化损失函数
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("损失函数对比", "对异常值的影响"))
        
        # 左图：损失函数曲线
        fig.add_trace(go.Scatter(
            x=errors, y=mse_loss,
            mode='lines',
            name='MSE (L2)',
            line=dict(color='red', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=errors, y=mae_loss,
            mode='lines',
            name='MAE (L1)',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=errors, y=huber_loss,
            mode='lines',
            name=f'Huber (δ={delta})',
            line=dict(color='green', width=2)
        ), row=1, col=1)
        
        # 右图：实际数据示例
        np.random.seed(42)
        n_samples = 50
        X = np.linspace(0, 10, n_samples)
        y_true = 2 * X + 1
        y_noisy = y_true + np.random.normal(0, 0.5, n_samples)
        
        # 添加异常值
        n_outliers = int(n_samples * outlier_ratio)
        outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
        y_noisy[outlier_idx] += np.random.normal(0, 5, n_outliers)
        
        fig.add_trace(go.Scatter(
            x=X, y=y_true,
            mode='lines',
            name='真实函数',
            line=dict(color='green', width=3, dash='dash'),
            showlegend=False
        ), row=1, col=2)
        
        # 正常点
        normal_mask = np.ones(n_samples, dtype=bool)
        normal_mask[outlier_idx] = False
        
        fig.add_trace(go.Scatter(
            x=X[normal_mask], y=y_noisy[normal_mask],
            mode='markers',
            name='正常数据',
            marker=dict(color='blue', size=6),
            showlegend=False
        ), row=1, col=2)
        
        # 异常点
        fig.add_trace(go.Scatter(
            x=X[outlier_idx], y=y_noisy[outlier_idx],
            mode='markers',
            name='异常值',
            marker=dict(color='red', size=10, symbol='x'),
            showlegend=False
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="误差", row=1, col=1)
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="损失", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 计算不同损失下的总损失
        residuals = y_noisy - y_true
        total_mse = np.mean(residuals ** 2)
        total_mae = np.mean(np.abs(residuals))
        total_huber = np.mean(np.where(
            np.abs(residuals) <= delta,
            0.5 * residuals ** 2,
            delta * np.abs(residuals) - 0.5 * delta ** 2
        ))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MSE损失", f"{total_mse:.3f}", 
                     help="对异常值非常敏感")
        with col2:
            st.metric("MAE损失", f"{total_mae:.3f}",
                     help="对异常值鲁棒，但优化困难")
        with col3:
            st.metric("Huber损失", f"{total_huber:.3f}",
                     help="平衡了MSE和MAE的优点")
        
        st.markdown("""
        **损失函数特点**:
        
        | 损失函数 | 优点 | 缺点 | 适用场景 |
        |---------|------|------|----------|
        | **MSE (L2)** | 可微、优化容易 | 对异常值敏感 | 数据干净时 |
        | **MAE (L1)** | 对异常值鲁棒 | 不可微、优化困难 | 大量异常值 |
        | **Huber** | 兼具两者优点 | 需要调参δ | 少量异常值 |
        
        **结论**: 
        - 当数据包含异常值或标签噪声时，使用鲁棒损失
        - Huber损失是实践中常用的折中方案
        - 根据异常值的严重程度调整δ参数
        """)
