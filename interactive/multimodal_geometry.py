"""
多模态几何交互式可视化
严格按照 24.MultimodalGeometry.md 中的理论实现

核心内容：
1. 超球面上的流形对齐 (CLIP)
2. InfoNCE与互信息
3. 温度系数的物理意义
4. 格拉斯曼流形
5. 张量融合
6. 条件SDE与Cross-Attention
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

class InteractiveMultimodalGeometry:
    """交互式多模态几何可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🌐 多模态几何：异构空间的对齐")
        
        st.markdown(r"""
        **核心挑战**: 如何让图像和文本这两种完全不同的模态"交流"？
        
        **关键技术**:
        1. **超球面对齐**: 将特征归一化到单位球面 $\|z\|_2 = 1$
        2. **InfoNCE损失**: 最大化互信息 $I(X;Y)$ 的下界
        3. **对比学习**: 拉近正样本，推开负样本
        
        **应用**: CLIP、ALIGN、Stable Diffusion、GPT-4V
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "超球面对齐 (CLIP)",
                    "InfoNCE与互信息",
                    "温度系数的作用",
                    "对比学习动态过程",
                    "格拉斯曼流形",
                    "张量融合",
                    "Cross-Attention几何"
                ]
            )
        
        if demo_type == "超球面对齐 (CLIP)":
            InteractiveMultimodalGeometry._render_hypersphere_alignment()
        elif demo_type == "InfoNCE与互信息":
            InteractiveMultimodalGeometry._render_info_nce()
        elif demo_type == "温度系数的作用":
            InteractiveMultimodalGeometry._render_temperature()
        elif demo_type == "对比学习动态过程":
            InteractiveMultimodalGeometry._render_contrastive_dynamics()
        elif demo_type == "格拉斯曼流形":
            InteractiveMultimodalGeometry._render_grassmannian()
        elif demo_type == "张量融合":
            InteractiveMultimodalGeometry._render_tensor_fusion()
        elif demo_type == "Cross-Attention几何":
            InteractiveMultimodalGeometry._render_cross_attention()
    

        # 添加交互式测验
        quiz_system = QuizSystem("multimodal_geometry")
        quizzes = QuizTemplates.get_multimodal_geometry_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_hypersphere_alignment():
        """超球面对齐可视化"""
        st.markdown("### 🌍 超球面对齐：CLIP的几何原理")
        
        st.markdown(r"""
        **核心思想**: 在单位超球面上，消除模长影响，只关注方向（语义）
        
        **数学原理**:
        """)
        
        st.latex(r"""
        \|z_I - z_T\|^2 = 2 - 2\cos(\theta) = 2(1 - \langle z_I, z_T \rangle)
        """)
        
        st.markdown(r"""
        **关键洞察**:
        - 球面上的欧氏距离 ↔ 余弦相似度
        - 模长被归一化，优化集中在角度
        - 高维球面体积集中在表面
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_samples = st.slider("样本对数量", 5, 30, 12, 1)
            alignment_quality = st.slider("对齐质量", 0.0, 1.0, 0.0, 0.1)
            show_connections = st.checkbox("显示配对连线", value=True)
            projection = st.selectbox("投影方式", ["2D圆", "3D球面"])
        
        # 生成数据
        np.random.seed(42)
        
        if projection == "2D圆":
            # 2D可视化
            dim = 2
            
            # 图像特征：集中在右上
            feat_image = np.random.randn(n_samples, dim) + np.array([2, 2])
            feat_image = feat_image / np.linalg.norm(feat_image, axis=1, keepdims=True)
            
            # 文本特征：根据对齐质量调整位置
            if alignment_quality < 0.1:
                # 初始状态：完全不对齐，在左下
                feat_text = np.random.randn(n_samples, dim) + np.array([-2, -2])
            else:
                # 逐渐对齐
                target = feat_image.copy()
                noise = np.random.randn(n_samples, dim) * (1 - alignment_quality)
                feat_text = alignment_quality * target + (1 - alignment_quality) * noise
            
            feat_text = feat_text / np.linalg.norm(feat_text, axis=1, keepdims=True)
            
            # 可视化
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "单位圆上的特征分布",
                    "余弦相似度矩阵"
                ),
                specs=[[{"type": "xy"}, {"type": "xy"}]]
            )
            
            # 1. 圆形边界
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            
            fig.add_trace(
                go.Scatter(
                    x=circle_x, y=circle_y,
                    mode='lines',
                    line=dict(color='lightgray', dash='dash'),
                    name='单位圆',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 2. 图像特征点
            fig.add_trace(
                go.Scatter(
                    x=feat_image[:, 0],
                    y=feat_image[:, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='circle',
                        color=list(range(n_samples)),
                        colorscale='Rainbow',
                        line=dict(width=2, color='black')
                    ),
                    name='图像特征',
                    text=[f'Image {i}' for i in range(n_samples)],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            
            # 3. 文本特征点
            fig.add_trace(
                go.Scatter(
                    x=feat_text[:, 0],
                    y=feat_text[:, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='x',
                        color=list(range(n_samples)),
                        colorscale='Rainbow',
                        line=dict(width=2)
                    ),
                    name='文本特征',
                    text=[f'Text {i}' for i in range(n_samples)],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            
            # 4. 配对连线
            if show_connections:
                for i in range(n_samples):
                    fig.add_trace(
                        go.Scatter(
                            x=[feat_image[i, 0], feat_text[i, 0]],
                            y=[feat_image[i, 1], feat_text[i, 1]],
                            mode='lines',
                            line=dict(
                                color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                                width=1
                            ),
                            opacity=0.4,
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1, col=1
                    )
            
            # 5. 余弦相似度矩阵
            similarity_matrix = np.dot(feat_image, feat_text.T)
            
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    x=[f'T{i}' for i in range(n_samples)],
                    y=[f'I{i}' for i in range(n_samples)],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(similarity_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 8},
                    colorbar=dict(title="余弦相似度")
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="X", scaleanchor="y", scaleratio=1, row=1, col=1)
            fig.update_yaxes(title_text="Y", row=1, col=1)
            fig.update_xaxes(title_text="文本", row=1, col=2)
            fig.update_yaxes(title_text="图像", row=1, col=2)
            
            fig.update_layout(
                height=500,
                showlegend=True,
                title_text=f"超球面对齐 (对齐质量={alignment_quality:.1f})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # 3D球面可视化
            dim = 3
            
            # 生成3D特征
            feat_image = np.random.randn(n_samples, dim) + np.array([1, 1, 1])
            feat_image = feat_image / np.linalg.norm(feat_image, axis=1, keepdims=True)
            
            if alignment_quality < 0.1:
                feat_text = np.random.randn(n_samples, dim) + np.array([-1, -1, -1])
            else:
                target = feat_image.copy()
                noise = np.random.randn(n_samples, dim) * (1 - alignment_quality)
                feat_text = alignment_quality * target + (1 - alignment_quality) * noise
            
            feat_text = feat_text / np.linalg.norm(feat_text, axis=1, keepdims=True)
            
            # 创建3D图
            fig = go.Figure()
            
            # 球面网格
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.1,
                colorscale='Blues',
                showscale=False,
                name='单位球面'
            ))
            
            # 图像特征点
            fig.add_trace(go.Scatter3d(
                x=feat_image[:, 0],
                y=feat_image[:, 1],
                z=feat_image[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    symbol='circle',
                    color=list(range(n_samples)),
                    colorscale='Rainbow',
                    line=dict(width=2, color='black')
                ),
                name='图像特征',
                text=[f'Image {i}' for i in range(n_samples)]
            ))
            
            # 文本特征点
            fig.add_trace(go.Scatter3d(
                x=feat_text[:, 0],
                y=feat_text[:, 1],
                z=feat_text[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    symbol='diamond',
                    color=list(range(n_samples)),
                    colorscale='Rainbow'
                ),
                name='文本特征',
                text=[f'Text {i}' for i in range(n_samples)]
            ))
            
            # 连线
            if show_connections:
                for i in range(n_samples):
                    fig.add_trace(go.Scatter3d(
                        x=[feat_image[i, 0], feat_text[i, 0]],
                        y=[feat_image[i, 1], feat_text[i, 1]],
                        z=[feat_image[i, 2], feat_text[i, 2]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        opacity=0.3,
                        showlegend=False
                    ))
            
            fig.update_layout(
                title=f"3D球面对齐 (对齐质量={alignment_quality:.1f})",
                scene=dict(
                    xaxis=dict(range=[-1.5, 1.5]),
                    yaxis=dict(range=[-1.5, 1.5]),
                    zaxis=dict(range=[-1.5, 1.5]),
                    aspectmode='cube'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 对齐分析")
        
        # 计算平均距离和相似度
        distances = np.linalg.norm(feat_image - feat_text, axis=1)
        similarities = np.sum(feat_image * feat_text, axis=1)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_dist = np.mean(distances)
            st.metric("平均距离", f"{avg_dist:.3f}")
        
        with col2:
            avg_sim = np.mean(similarities)
            st.metric("平均相似度", f"{avg_sim:.3f}")
        
        with col3:
            # 对角线相似度（正样本）
            diag_sim = np.mean(similarities)
            st.metric("正样本相似度", f"{diag_sim:.3f}")
        
        with col4:
            # 非对角线相似度（负样本）
            if n_samples > 1:
                sim_matrix = np.dot(feat_image, feat_text.T)
                mask = ~np.eye(n_samples, dtype=bool)
                off_diag_sim = np.mean(sim_matrix[mask])
                st.metric("负样本相似度", f"{off_diag_sim:.3f}")
        
        # 理论解释
        st.markdown("### 🎓 理论要点")
        
        st.success(r"""
        **球面几何的优势**:
        
        1. **消除模长**: $\|z\| = 1$ 使得优化只关注方向（语义）
        2. **度量等价**: 欧氏距离 $\leftrightarrow$ 余弦相似度
        3. **体积集中**: 高维球面体积集中在表面
        
        **关系式**:
        $$\|z_I - z_T\|^2 = 2(1 - \langle z_I, z_T \rangle)$$
        
        因此最小化距离 = 最大化余弦相似度
        """)
        
        if alignment_quality < 0.2:
            st.warning("""
            **当前状态: 未对齐**
            - 图像和文本特征分布在球面的不同区域
            - 余弦相似度接近0或负值
            - 需要通过对比学习训练
            """)
        elif alignment_quality < 0.7:
            st.info("""
            **当前状态: 部分对齐**
            - 特征开始向同一区域移动
            - 正样本相似度增加
            - 需要继续训练
            """)
        else:
            st.success("""
            **当前状态: 高度对齐**
            - 图像和文本特征重合
            - 正样本相似度接近1
            - 可以进行零样本推理
            """)
    
    @staticmethod
    def _render_info_nce():
        """InfoNCE与互信息可视化"""
        st.markdown("### 📐 InfoNCE：互信息的变分下界")
        
        st.markdown(r"""
        **核心定理**: 最小化InfoNCE Loss等价于最大化互信息的下界
        """)
        
        st.latex(r"""
        I(X; Y) \geq \log N - \mathcal{L}_{\text{NCE}}
        """)
        
        st.markdown(r"""
        **推导关键步骤**:
        
        1. 互信息定义: $I(X;Y) = \mathbb{E}\left[\log\frac{p(y|x)}{p(y)}\right]$
        
        2. InfoNCE损失:
        """)
        
        st.latex(r"""
        \mathcal{L}_{\text{NCE}} = -\mathbb{E}\left[\log\frac{e^{f(x,y)}}{e^{f(x,y)} + \sum_{j=1}^{N-1} e^{f(x,y_j)}}\right]
        """)
        
        st.markdown(r"""
        3. 当$N$很大时: $\mathcal{L}_{\text{NCE}} \approx \log N - I(X;Y)$
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_negatives = st.slider("负样本数量 N", 2, 128, 32, 2)
            temperature = st.slider("温度 τ", 0.01, 1.0, 0.1, 0.01)
            mutual_info = st.slider("真实互信息 I(X;Y)", 0.0, 5.0, 2.0, 0.1)
        
        # 模拟不同负样本数量下的下界紧密度
        N_range = np.logspace(0, 3, 50)  # 1 到 1000
        
        # 理论互信息
        true_MI = mutual_info
        
        # InfoNCE下界
        # 假设最优情况下 L_NCE 接近 0（完美分类）
        optimal_loss = 0.1  # 实际中很难为0
        lower_bound = np.log(N_range) - optimal_loss
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "互信息下界 vs 负样本数",
                "InfoNCE Loss分解",
                "下界紧密度",
                "温度对loss的影响"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. 互信息下界
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=[true_MI] * len(N_range),
                mode='lines',
                name='真实互信息 I(X;Y)',
                line=dict(color='green', width=3, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=lower_bound,
                mode='lines',
                name='InfoNCE下界',
                line=dict(color='blue', width=3),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        # 标注当前N
        current_bound = np.log(n_negatives) - optimal_loss
        fig.add_trace(
            go.Scatter(
                x=[n_negatives],
                y=[current_bound],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name=f'当前N={n_negatives}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. InfoNCE Loss分解
        components = ['log N', '-I(X;Y)', '误差项', 'Total Loss']
        values = [np.log(n_negatives), -true_MI, 0.5, np.log(n_negatives) - true_MI + 0.5]
        colors = ['blue', 'red', 'orange', 'purple']
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=values,
                marker_color=colors,
                text=[f'{v:.2f}' for v in values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. 下界紧密度（Gap）
        gap = true_MI - lower_bound
        
        fig.add_trace(
            go.Scatter(
                x=N_range,
                y=gap,
                mode='lines',
                name='下界间隙',
                line=dict(color='red', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[n_negatives],
                y=[true_MI - current_bound],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 温度对loss的影响
        tau_range = np.linspace(0.01, 1.0, 50)
        
        # 模拟：温度越低，loss越敏感（梯度越大）
        # 假设正样本相似度=0.8，负样本相似度=0.2
        pos_sim = 0.8
        neg_sim = 0.2
        
        losses = []
        for tau in tau_range:
            logits_pos = pos_sim / tau
            logits_neg = neg_sim / tau
            
            # InfoNCE loss（简化）
            exp_pos = np.exp(logits_pos)
            exp_neg_sum = (n_negatives - 1) * np.exp(logits_neg)
            loss = -np.log(exp_pos / (exp_pos + exp_neg_sum))
            losses.append(loss)
        
        fig.add_trace(
            go.Scatter(
                x=tau_range,
                y=losses,
                mode='lines',
                name='InfoNCE Loss',
                line=dict(color='purple', width=3)
            ),
            row=2, col=2
        )
        
        # 标注当前温度
        idx = np.argmin(np.abs(tau_range - temperature))
        fig.add_trace(
            go.Scatter(
                x=[temperature],
                y=[losses[idx]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name=f'当前τ={temperature}',
                showlegend=True
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="负样本数量 N", type="log", row=1, col=1)
        fig.update_yaxes(title_text="互信息", row=1, col=1)
        fig.update_xaxes(title_text="组件", row=1, col=2)
        fig.update_yaxes(title_text="值", row=1, col=2)
        fig.update_xaxes(title_text="负样本数量 N", type="log", row=2, col=1)
        fig.update_yaxes(title_text="间隙 (真实MI - 下界)", row=2, col=1)
        fig.update_xaxes(title_text="温度 τ", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="InfoNCE与互信息"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 理论分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("负样本数 N", f"{n_negatives}")
        
        with col2:
            st.metric("理论上界 log N", f"{np.log(n_negatives):.2f}")
        
        with col3:
            current_gap = true_MI - current_bound
            st.metric("下界间隙", f"{current_gap:.2f}")
        
        with col4:
            tightness = (current_bound / true_MI * 100) if true_MI > 0 else 0
            st.metric("下界紧密度", f"{tightness:.1f}%")
        
        # 建议
        st.markdown("### 💡 关键洞察")
        
        st.success(r"""
        **InfoNCE的深层数学**:
        
        1. **下界性质**: 
           - InfoNCE提供了互信息的下界
           - 最小化loss = 最大化互信息下界
        
        2. **负样本的作用**:
           - $N$越大，下界越紧
           - CLIP使用32k的batch size不是偶然！
        
        3. **理论保证**:
           $$I(X;Y) \geq \log N - \mathcal{L}_{\text{NCE}}$$
           
           当loss接近0时，学到的互信息接近 $\log N$
        """)
        
        if n_negatives < 16:
            st.warning("""
            **负样本太少**:
            - 下界很松，学到的互信息有限
            - 建议: 增大batch size或使用memory bank
            """)
        elif n_negatives > 64:
            st.success("""
            **负样本充足**:
            - 下界较紧，能学到丰富的互信息
            - 这是大规模对比学习的关键
            """)
    
    @staticmethod
    def _render_temperature():
        """温度系数的作用可视化"""
        st.markdown("### 🌡️ 温度系数：精确匹配 vs 最大熵")
        
        st.markdown(r"""
        **物理意义**: 温度 $\tau$ 控制softmax分布的尖锐程度
        
        **数学表达**:
        """)
        
        st.latex(r"""
        p(y|x) = \frac{\exp(z_x \cdot z_y / \tau)}{\sum_{j} \exp(z_x \cdot z_{y_j} / \tau)}
        """)
        
        st.markdown(r"""
        **效果**:
        - **小 $\tau$ (低温)**: 分布尖锐，只关注最难的负样本
        - **大 $\tau$ (高温)**: 分布平滑，所有负样本均匀贡献
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            temperature = st.slider("温度 τ", 0.01, 2.0, 0.1, 0.01)
            n_samples = st.slider("样本数", 5, 20, 10, 1)
        
        # 模拟相似度分布
        np.random.seed(42)
        
        # 正样本相似度（高）
        pos_similarity = 0.9
        
        # 负样本相似度（分布在低相似度区域）
        neg_similarities = np.random.beta(2, 5, n_samples - 1) * 0.8
        
        all_similarities = np.concatenate([[pos_similarity], neg_similarities])
        
        # 计算不同温度下的softmax概率
        temperatures = [0.01, 0.05, 0.1, 0.5, 1.0]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"τ = {t}" for t in temperatures] + ["温度对梯度的影响"],
            specs=[[{"type": "xy"}] * 3,
                   [{"type": "xy"}] * 2 + [{"type": "xy"}]]
        )
        
        # 绘制不同温度下的概率分布
        position_labels = ['正样本'] + [f'负{i}' for i in range(1, n_samples)]
        
        for idx, temp in enumerate(temperatures):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            # 计算softmax
            logits = all_similarities / temp
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # 柱状图
            colors = ['green'] + ['red'] * (n_samples - 1)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(n_samples)),
                    y=probs,
                    marker_color=colors,
                    showlegend=False,
                    text=[f'{p:.3f}' for p in probs],
                    textposition='outside'
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(title_text="概率", range=[0, 1], row=row, col=col)
            fig.update_xaxes(title_text="样本", row=row, col=col)
        
        # 温度对梯度的影响
        temp_range = np.logspace(-2, 0.5, 50)
        
        gradient_norms = []
        entropies = []
        
        for temp in temp_range:
            logits = all_similarities / temp
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # 梯度范数（简化：正样本的梯度）
            grad_norm = (probs[0] - 1) ** 2
            gradient_norms.append(grad_norm)
            
            # 熵
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        fig.add_trace(
            go.Scatter(
                x=temp_range,
                y=gradient_norms,
                mode='lines',
                name='梯度范数',
                line=dict(color='blue', width=3),
                yaxis='y6'
            ),
            row=2, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=temp_range,
                y=entropies,
                mode='lines',
                name='分布熵',
                line=dict(color='red', width=3),
                yaxis='y7'
            ),
            row=2, col=3
        )
        
        # 标注当前温度
        idx_current = np.argmin(np.abs(temp_range - temperature))
        fig.add_trace(
            go.Scatter(
                x=[temperature, temperature],
                y=[0, max(gradient_norms)],
                mode='lines',
                line=dict(color='green', dash='dash', width=2),
                name=f'当前τ={temperature}',
                showlegend=True
            ),
            row=2, col=3
        )
        
        fig.update_xaxes(title_text="温度 τ", type="log", row=2, col=3)
        fig.update_yaxes(title_text="梯度/熵", row=2, col=3)
        
        # 创建双y轴效果（通过调整范围）
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="温度系数的作用"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 当前温度分析
        st.markdown("### 📊 当前温度分析")
        
        logits = all_similarities / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("温度 τ", f"{temperature:.3f}")
        
        with col2:
            st.metric("正样本概率", f"{probs[0]:.3f}")
        
        with col3:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            st.metric("分布熵", f"{entropy:.3f}")
        
        with col4:
            hardness = 1 - probs[0]
            st.metric("训练难度", f"{hardness:.3f}")
        
        # 理论解释
        st.markdown("### 🎓 理论要点")
        
        if temperature < 0.05:
            st.warning(r"""
            **极低温 (τ < 0.05)**:
            - 概率分布极度尖锐
            - 只有最相似的样本有贡献
            - **Hard Negative Mining**: 只关注最难区分的负样本
            - **风险**: 训练不稳定，容易陷入局部最优
            """)
        elif temperature < 0.2:
            st.success(r"""
            **低温 (0.05 < τ < 0.2)**:
            - CLIP的默认选择 (τ ≈ 0.07)
            - 平衡精确匹配和稳定训练
            - 对难负样本敏感，对简单负样本不敏感
            - **最佳实践区域**
            """)
        elif temperature < 0.5:
            st.info(r"""
            **中温 (0.2 < τ < 0.5)**:
            - 分布较平滑
            - 所有负样本都有一定贡献
            - 训练更稳定但收敛慢
            """)
        else:
            st.warning(r"""
            **高温 (τ > 0.5)**:
            - 分布接近均匀
            - **最大熵**: 所有样本贡献相同
            - 训练信号弱，学习缓慢
            - 类似最优传输中的高熵正则化
            """)
        
        st.info(r"""
        **与最优传输的联系** (见Ch 22):
        
        温度 $\tau$ 在InfoNCE中的作用与Sinkhorn算法中的熵正则化 $\epsilon$ 完全相同：
        
        - **$\tau \to 0$**: 精确匹配（Hard Assignment）
        - **$\tau \to \infty$**: 最大熵（Uniform Distribution）
        
        两者都在平衡"精确性"与"稳定性"。
        """)
    
    @staticmethod
    def _render_contrastive_dynamics():
        """对比学习动态过程可视化"""
        st.markdown("### 🔄 对比学习的动态过程")
        
        st.markdown(r"""
        **学习过程**: 在超球面上，通过梯度下降将配对特征拉近
        
        **梯度方向**:
        - 正样本: 拉近 (attractive force)
        - 负样本: 推开 (repulsive force)
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            n_pairs = st.slider("样本对数量", 5, 15, 8, 1)
            learning_rate = st.slider("学习率", 0.01, 1.0, 0.3, 0.01)
            temperature = st.slider("温度", 0.01, 0.5, 0.1, 0.01)
            n_steps = st.slider("训练步数", 10, 100, 50, 10)
            show_animation = st.checkbox("显示动画", value=False)
        
        # 生成初始数据
        np.random.seed(42)
        dim = 2
        
        # 图像特征：右上
        feat_image = np.random.randn(n_pairs, dim) + np.array([1.5, 1.5])
        feat_image = feat_image / np.linalg.norm(feat_image, axis=1, keepdims=True)
        
        # 文本特征：左下（初始不对齐）
        feat_text = np.random.randn(n_pairs, dim) + np.array([-1.5, -1.5])
        feat_text = feat_text / np.linalg.norm(feat_text, axis=1, keepdims=True)
        
        # 记录训练历史
        history_text = [feat_text.copy()]
        loss_history = []
        
        # 训练循环
        for step in range(n_steps):
            # 计算相似度矩阵
            sim_matrix = np.dot(feat_image, feat_text.T) / temperature
            
            # Softmax
            exp_sim = np.exp(sim_matrix - np.max(sim_matrix, axis=1, keepdims=True))
            probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
            
            # InfoNCE loss
            loss = -np.mean(np.log(np.diag(probs) + 1e-10))
            loss_history.append(loss)
            
            # 梯度计算（简化）
            targets = np.eye(n_pairs)
            grad = np.dot((probs - targets).T, feat_image) / temperature
            
            # 更新
            feat_text = feat_text - learning_rate * grad
            feat_text = feat_text / np.linalg.norm(feat_text, axis=1, keepdims=True)
            
            history_text.append(feat_text.copy())
        
        # 可视化
        if show_animation:
            # 创建动画帧
            frames = []
            for step_idx in range(0, len(history_text), max(1, len(history_text) // 20)):
                frame_data = []
                
                # 圆
                theta = np.linspace(0, 2*np.pi, 100)
                frame_data.append(go.Scatter(
                    x=np.cos(theta), y=np.sin(theta),
                    mode='lines', line=dict(color='lightgray', dash='dash'),
                    showlegend=False
                ))
                
                # 图像特征
                frame_data.append(go.Scatter(
                    x=feat_image[:, 0], y=feat_image[:, 1],
                    mode='markers',
                    marker=dict(size=10, symbol='circle', color=list(range(n_pairs)),
                               colorscale='Rainbow', line=dict(width=2, color='black')),
                    showlegend=False
                ))
                
                # 文本特征
                current_text = history_text[step_idx]
                frame_data.append(go.Scatter(
                    x=current_text[:, 0], y=current_text[:, 1],
                    mode='markers',
                    marker=dict(size=10, symbol='x', color=list(range(n_pairs)),
                               colorscale='Rainbow'),
                    showlegend=False
                ))
                
                # 连线
                for i in range(n_pairs):
                    frame_data.append(go.Scatter(
                        x=[feat_image[i, 0], current_text[i, 0]],
                        y=[feat_image[i, 1], current_text[i, 1]],
                        mode='lines',
                        line=dict(width=1, color='gray'),
                        opacity=0.3,
                        showlegend=False
                    ))
                
                frames.append(go.Frame(data=frame_data, name=str(step_idx)))
            
            fig = go.Figure(
                data=frames[0].data,
                frames=frames,
                layout=go.Layout(
                    title="对比学习动态过程",
                    updatemenus=[{
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {"label": "播放", "method": "animate",
                             "args": [None, {"frame": {"duration": 100}}]},
                            {"label": "暂停", "method": "animate",
                             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
                        ]
                    }],
                    xaxis=dict(range=[-1.5, 1.5], scaleanchor="y", scaleratio=1),
                    yaxis=dict(range=[-1.5, 1.5]),
                    height=600
                )
            )
        else:
            # 静态对比图
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("初始状态", "最终状态"),
                specs=[[{"type": "xy"}, {"type": "xy"}]]
            )
            
            # 初始状态
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta),
                                    mode='lines', line=dict(color='lightgray', dash='dash'),
                                    showlegend=False), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=feat_image[:, 0], y=feat_image[:, 1],
                mode='markers',
                marker=dict(size=10, symbol='circle', color=list(range(n_pairs)),
                           colorscale='Rainbow', showscale=False),
                name='Image', showlegend=True
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=history_text[0][:, 0], y=history_text[0][:, 1],
                mode='markers',
                marker=dict(size=10, symbol='x', color=list(range(n_pairs)),
                           colorscale='Rainbow', showscale=False),
                name='Text (初始)', showlegend=True
            ), row=1, col=1)
            
            # 最终状态
            fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta),
                                    mode='lines', line=dict(color='lightgray', dash='dash'),
                                    showlegend=False), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=feat_image[:, 0], y=feat_image[:, 1],
                mode='markers',
                marker=dict(size=10, symbol='circle', color=list(range(n_pairs)),
                           colorscale='Rainbow', showscale=False),
                showlegend=False
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=history_text[-1][:, 0], y=history_text[-1][:, 1],
                mode='markers',
                marker=dict(size=10, symbol='x', color=list(range(n_pairs)),
                           colorscale='Rainbow', showscale=False),
                name='Text (最终)', showlegend=True
            ), row=1, col=2)
            
            # 轨迹
            for i in range(n_pairs):
                trajectory = np.array([h[i] for h in history_text])
                fig.add_trace(go.Scatter(
                    x=trajectory[:, 0], y=trajectory[:, 1],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    opacity=0.5,
                    showlegend=False
                ), row=1, col=2)
            
            fig.update_xaxes(scaleanchor="y", scaleratio=1, range=[-1.5, 1.5], row=1, col=1)
            fig.update_yaxes(range=[-1.5, 1.5], row=1, col=1)
            fig.update_xaxes(scaleanchor="y2", scaleratio=1, range=[-1.5, 1.5], row=1, col=2)
            fig.update_yaxes(range=[-1.5, 1.5], row=1, col=2)
            
            fig.update_layout(height=500, title_text="对比学习训练过程")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Loss曲线
        st.markdown("### 📉 训练曲线")
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(loss_history))),
            y=loss_history,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig_loss.update_layout(
            title="InfoNCE Loss",
            xaxis_title="Step",
            yaxis_title="Loss",
            height=300
        )
        
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # 统计
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_dist = np.mean(np.linalg.norm(feat_image - history_text[0], axis=1))
            st.metric("初始距离", f"{initial_dist:.3f}")
        
        with col2:
            final_dist = np.mean(np.linalg.norm(feat_image - history_text[-1], axis=1))
            st.metric("最终距离", f"{final_dist:.3f}")
        
        with col3:
            improvement = (initial_dist - final_dist) / initial_dist * 100
            st.metric("改善程度", f"{improvement:.1f}%")
        
        st.success("""
        **对比学习的几何直观**:
        
        1. **吸引力**: 正样本对之间的相似度增加
        2. **排斥力**: 负样本对之间保持距离
        3. **球面约束**: 所有特征都在单位球面上
        4. **收敛**: 最终配对特征重合，形成对齐
        """)

    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _render_grassmannian():
        """格拉斯曼流形（简化版）"""
        st.markdown("### 📐 格拉斯曼流形：子空间的几何")
        
        st.markdown(r"""
        **应用**: 视频理解、Few-shot学习
        
        **核心思想**: 将一组向量（如视频帧）表示为一个子空间
        """)
        
        st.info("""
        格拉斯曼流形 Gr(k,n) 是 R^n 中所有 k 维线性子空间的集合。
        
        **距离度量**: 主角度 (Principal Angles)
        
        **应用场景**:
        - 视频分类：每个视频 → 一个子空间
        - Few-shot学习：每个类 → 一个子空间
        - 多模态对齐：子空间之间的距离
        """)
        
        st.warning("完整的格拉斯曼流形可视化需要高维数学，这里展示概念性理解。")
    
    @staticmethod
    def _render_tensor_fusion():
        """张量融合可视化"""
        st.markdown("### 🧮 张量融合：捕捉高阶交互")
        
        st.markdown(r"""
        **问题**: 简单拼接 [v_I; v_T] 只是线性操作，无法捕捉乘法交互
        
        **解决方案**: 外积 (Outer Product)
        """)
        
        st.latex(r"""
        Z = v_I \otimes v_T \in \mathbb{R}^{D_I \times D_T}
        """)
        
        st.markdown(r"""
        $Z_{ij} = v_{I,i} \cdot v_{T,j}$ 捕捉了所有特征对之间的交互
        """)
        
        with st.sidebar:
            dim_image = st.slider("图像特征维度", 4, 16, 8, 2)
            dim_text = st.slider("文本特征维度", 4, 16, 8, 2)
        
        # 生成特征
        np.random.seed(42)
        v_image = np.random.randn(dim_image)
        v_text = np.random.randn(dim_text)
        
        # 外积
        tensor_fusion = np.outer(v_image, v_text)
        
        # 可视化
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("图像特征", "文本特征", "张量融合 (外积)"),
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
        )
        
        # 图像特征
        fig.add_trace(go.Bar(
            y=list(range(dim_image)),
            x=v_image,
            orientation='h',
            marker_color='blue',
            name='Image'
        ), row=1, col=1)
        
        # 文本特征
        fig.add_trace(go.Bar(
            x=list(range(dim_text)),
            y=v_text,
            marker_color='red',
            name='Text'
        ), row=1, col=2)
        
        # 张量融合
        fig.add_trace(go.Heatmap(
            z=tensor_fusion,
            colorscale='RdBu',
            zmid=0,
            showscale=True
        ), row=1, col=3)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(r"""
        **张量融合的优势**:
        - 捕捉特征间的二阶相关性
        - 逻辑与/或/非操作
        - VQA任务的核心技术
        
        **维度爆炸问题**: $D_I \times D_T$ 太大
        
        **解决方案**: 低秩分解
        $$Z \approx (v_I W_I) \odot (v_T W_T)$$
        """)
    
    @cache_numpy_computation(ttl=1800)
    @staticmethod
    def _render_cross_attention():
        """Cross-Attention几何可视化"""
        st.markdown("### 🎯 Cross-Attention：跨模态的传送门")
        
        st.markdown(r"""
        **在扩散模型中的作用**: 文本引导图像生成
        
        **数学形式**:
        """)
        
        st.latex(r"""
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
        """)
        
        st.markdown(r"""
        **Cross-Attention**: 
        - Q: 图像特征（查询）
        - K, V: 文本特征（键值对）
        
        **几何直观**: 文本作为"注意力引导"，将图像拉向特定的语义方向
        """)
        
        st.info("""
        **在Stable Diffusion中**:
        
        1. 文本编码器 → 文本嵌入 K, V
        2. 图像去噪网络 → 查询 Q
        3. Cross-Attention → 文本引导图像生成
        
        **条件SDE**:
        ∇_x log p_t(x|y) = ∇_x log p_t(x) + λ ∇_x log p_t(y|x)
        
        Cross-Attention本质上在估计这个条件引导项。
        """)
        
        st.success("""
        **多模态几何的三大技术**:
        
        1. **球面几何 (CLIP)**: 解决"是什么"的问题
           - 建立模态间的字典
        
        2. **张量几何**: 解决"怎么样"的问题
           - 捕捉复杂的逻辑交互
        
        3. **微分几何 (Cross-Attention/Diffusion)**: 解决"创造"的问题
           - 让语义能够引导生成
        """)

        # 添加交互式测验
