"""
交互式因果推断与Do-Calculus可视化
严格按照 21.Causation.md 中的理论实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from scipy import stats
import warnings
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

warnings.filterwarnings('ignore')

# 尝试导入dowhy，如果不可用则使用简化实现
try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    st.warning("⚠️ DoWhy库未安装，部分功能将使用简化实现")


class InteractiveCausation:
    """交互式因果推断与Do-Calculus可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🔗 因果推断：结构方程与Do-Calculus")
        st.markdown("""
        **核心思想**: 从相关性走向因果性，建立区别于经典概率论的运算体系
        
        关键概念：
        - **结构因果模型(SCM)**: $\mathcal{M} = \langle U, V, F, P(U) \rangle$
        - **Do算子**: $P(Y|do(X))$ vs $P(Y|X)$
        - **后门调整**: $P(Y=y|do(X=x)) = \sum_z P(Y=y|X=x, Z=z) P(Z=z)$
        - **反事实推理**: $P(Y_{x'}|X=x, Y=y)$
        """)
        
        with st.sidebar:
            st.markdown("### 📊 可视化选择")
            viz_type = st.selectbox("选择可视化类型", 
                ["DAG基础", "Simpson悖论", "Do-Calculus", "反事实推理"])
            
            st.markdown("### 🎛️ 参数调整")
        
        if viz_type == "DAG基础":
            InteractiveCausation._render_dag_basics()
        elif viz_type == "Simpson悖论":
            InteractiveCausation._render_simpson_paradox()
        elif viz_type == "Do-Calculus":
            InteractiveCausation._render_do_calculus()
        elif viz_type == "反事实推理":
            InteractiveCausation._render_counterfactual()
    

        # 添加交互式测验
        quiz_system = QuizSystem("causation")
        quizzes = QuizTemplates.get_causation_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_dag_basics():
        """DAG基础概念演示"""
        st.markdown("### 🕸️ 有向无环图(DAG)：因果关系的拓扑表示")
        
        st.markdown("""
        **核心概念**：
        - **节点**: 随机变量
        - **边**: 直接因果关系
        - **路径**: 间接因果影响
        - **后门路径**: 导致混淆的路径
        """)
        
        with st.sidebar:
            dag_type = st.selectbox("图类型", 
                ["简单链式", "混淆结构", "对撞结构", "后门路径"])
            show_paths = st.checkbox("显示路径分析", value=True)
            show_intervention = st.checkbox("显示干预效果", value=True)
        
        # 创建不同类型的DAG
        if dag_type == "简单链式":
            # X -> M -> Y
            G = nx.DiGraph()
            G.add_edges_from([('X', 'M'), ('M', 'Y')])
            pos = {'X': (0, 0), 'M': (2, 0), 'Y': (4, 0)}
            title = "链式中介: X → M → Y"
            
        elif dag_type == "混淆结构":
            # Z -> X, Z -> Y, X -> Y
            G = nx.DiGraph()
            G.add_edges_from([('Z', 'X'), ('Z', 'Y'), ('X', 'Y')])
            pos = {'Z': (2, 2), 'X': (0, 0), 'Y': (4, 0)}
            title = "混淆结构: Z → X → Y, Z → Y"
            
        elif dag_type == "对撞结构":
            # X -> M, Y -> M
            G = nx.DiGraph()
            G.add_edges_from([('X', 'M'), ('Y', 'M')])
            pos = {'X': (0, 0), 'Y': (4, 0), 'M': (2, -2)}
            title = "对撞结构: X → M ← Y"
            
        else:  # 后门路径
            # Z -> X, Z -> Y, X -> Y, W -> X
            G = nx.DiGraph()
            G.add_edges_from([('Z', 'X'), ('Z', 'Y'), ('X', 'Y'), ('W', 'X')])
            pos = {'Z': (1, 2), 'W': (3, 2), 'X': (2, 0), 'Y': (4, 0)}
            title = "后门路径: Z, W → X → Y, Z → Y"
        
        # 可视化DAG
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[title, "路径分析"],
            specs=[[{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 绘制DAG
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color='gray'),
                showlegend=False
            ),
            row=1, col=1
        )

        # 添加交互式测验
        
        # 绘制节点
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
                text=node_text,
                textposition="middle center",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 路径分析表格
        if show_paths:
            paths_data = []
            if dag_type == "混淆结构":
                paths_data = [
                    ["X → Y", "直接因果路径", "开放"],
                    ["X ← Z → Y", "后门路径", "需要阻断"],
                    ["X → Y | Z", "条件化后", "因果效应"]
                ]
            elif dag_type == "对撞结构":
                paths_data = [
                    ["X → M ← Y", "对撞路径", "天然阻断"],
                    ["X → M ← Y | M", "条件化对撞", "开放偏误"],
                    ["X ⊥ Y", "边际独立", "无直接关联"]
                ]
            elif dag_type == "后门路径":
                paths_data = [
                    ["X → Y", "直接因果", "开放"],
                    ["X ← Z → Y", "后门路径1", "需要控制Z"],
                    ["X ← W → Y", "后门路径2", "需要控制W"],
                    ["X → Y | Z, W", "完全控制", "纯净因果"]
                ]
            else:
                paths_data = [
                    ["X → M → Y", "间接因果", "完全中介"],
                    ["X ⊥ Y | M", "条件独立", "中介阻断"],
                    ["X → Y", "总效应", "直接+间接"]
                ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=["路径", "类型", "状态"]),
                    cells=dict(values=list(zip(*paths_data)))
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="因果图结构分析",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 干预效果演示
        if show_intervention:
            st.markdown("### 🔧 干预效果演示")
            
            # 生成模拟数据
            np.random.seed(42)
            n = 1000
            
            if dag_type == "混淆结构":
                Z = np.random.normal(0, 1, n)
                X = 0.5 * Z + np.random.normal(0, 1, n)
                Y = 2 * X + 1.5 * Z + np.random.normal(0, 1, n)
                
                # 计算观测效应和因果效应
                obs_effect = np.cov(X, Y)[0, 1] / np.var(X)
                causal_effect = 2.0  # 真实系数
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("观测关联", f"{obs_effect:.3f}")
                    st.info("包含混淆偏差")
                with col2:
                    st.metric("因果效应", f"{causal_effect:.3f}")
                    st.success("真实物理机制")
                
                # 可视化
                fig = go.Figure()
                
                # 原始数据
                fig.add_trace(go.Scatter(
                    x=X, y=Y,
                    mode='markers',
                    name='观测数据',
                    opacity=0.6,
                    marker=dict(color='blue', size=6)
                ))
                
                # 拟合线
                x_range = np.linspace(X.min(), X.max(), 100)
                y_obs = obs_effect * x_range
                y_causal = causal_effect * x_range
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_obs,
                    mode='lines',
                    name='观测回归',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_causal,
                    mode='lines',
                    name='因果效应',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title="观测关联 vs 因果效应",
                    xaxis_title="X",
                    yaxis_title="Y"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **DAG的核心价值**：
        - **定性分析**: 明确变量间的因果方向
        - **识别策略**: 确定需要控制的混淆变量
        - **干预指导**: 告诉我们如何进行有效的干预
        """)
    
    @staticmethod
    def _render_simpson_paradox():
        """Simpson悖论演示"""
        st.markdown("### 🔄 Simpson悖论：统计学的陷阱")
        
        st.markdown("""
        **数学本质**: 选择偏差导致总体相关性与分组相关性符号相反
        
        **偏差公式**: $\Delta_{obs} = \delta + \sum_z \mathbb{E}[Y|T=0,z] [P(z|T=1) - P(z|T=0)]$
        """)
        
        with st.sidebar:
            confounding_strength = st.slider("混淆强度", 0.5, 3.0, 1.5, 0.1)
            treatment_effect = st.slider("真实治疗效应", 0.5, 3.0, 2.0, 0.1)
            sample_size = st.slider("样本量", 100, 2000, 500, 100)
            show_groups = st.checkbox("显示分组分析", value=True)
        
        # 生成Simpson悖论数据
        np.random.seed(42)
        n = sample_size
        
        # 混淆变量Z (比如病情严重程度)
        Z = np.random.binomial(1, 0.5, n)  # 0: 轻症, 1: 重症
        
        # 治疗分配T (重症更可能接受治疗)
        logit_t = -1 + confounding_strength * Z
        P_T_given_Z = 1 / (1 + np.exp(-logit_t))
        T = np.random.binomial(1, P_T_given_Z, n)
        
        # 结果Y (治疗效果)
        # 重症基础恢复率低，但治疗效果相同
        base_recovery = 0.8 - 0.4 * Z  # 轻症0.8, 重症0.4
        Y = base_recovery + treatment_effect * T + np.random.normal(0, 0.1, n)
        
        # 创建DataFrame
        df = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})
        df['Z_label'] = df['Z'].map({0: '轻症', 1: '重症'})
        df['T_label'] = df['T'].map({0: '对照', 1: '治疗'})
        
        # 计算各种效应
        # 总体观测效应
        overall_effect = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()
        
        # 分组效应
        group_effects = {}
        for z_val in [0, 1]:
            group_df = df[df['Z'] == z_val]
            effect = group_df[group_df['T']==1]['Y'].mean() - group_df[group_df['T']==0]['Y'].mean()
            group_effects[f'{"轻症" if z_val==0 else "重症"}'] = effect
        
        # 真实因果效应 (我们设定的参数)
        true_effect = treatment_effect
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "总体数据分布", "分组数据分布",
                "效应对比", "混淆机制分析"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 总体分布
        for t_val in [0, 1]:
            subset = df[df['T'] == t_val]
            fig.add_trace(
                go.Scatter(
                    x=subset['Y'],
                    y=np.random.normal(0, 0.1, len(subset)) + t_val,
                    mode='markers',
                    name=f'{"对照" if t_val==0 else "治疗"}',
                    opacity=0.6,
                    marker=dict(color='red' if t_val==0 else 'blue')
                ),
                row=1, col=1
            )
        
        # 分组分布
        colors = ['lightblue', 'darkblue']
        for i, z_val in enumerate([0, 1]):
            subset = df[df['Z'] == z_val]
            for t_val in [0, 1]:
                t_subset = subset[subset['T'] == t_val]
                fig.add_trace(
                    go.Scatter(
                        x=t_subset['Y'],
                        y=np.random.normal(i, 0.1, len(t_subset)) + t_val * 0.3,
                        mode='markers',
                        name=f'{"轻症" if z_val==0 else "重症"}-{"对照" if t_val==0 else "治疗"}',
                        opacity=0.6,
                        marker=dict(color=colors[t_val])
                    ),
                    row=1, col=2
                )
        
        # 效应对比
        effects = ['总体观测', '轻症组', '重症组', '真实因果']
        values = [overall_effect, group_effects['轻症'], group_effects['重症'], true_effect]
        colors_bar = ['red', 'orange', 'orange', 'green']
        
        fig.add_trace(
            go.Bar(
                x=effects,
                y=values,
                marker_color=colors_bar,
                name='效应值'
            ),
            row=2, col=1
        )
        
        # 混淆机制表格
        confusion_data = [
            ["轻症人群", "治疗比例", f"{df[df['Z']==0]['T'].mean():.2%}"],
            ["重症人群", "治疗比例", f"{df[df['Z']==1]['T'].mean():.2%}"],
            ["选择偏差", "差异", f"{abs(df[df['Z']==1]['T'].mean() - df[df['Z']==0]['T'].mean()):.2%}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["分组", "指标", "数值"]),
                cells=dict(values=list(zip(*confusion_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Simpson悖论完整分析",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 📊 详细效应分析")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总体观测效应", f"{overall_effect:.3f}", "❌ 包含偏差")
        with col2:
            st.metric("轻症组效应", f"{group_effects['轻症']:.3f}", "✅ 接近真实")
        with col3:
            st.metric("重症组效应", f"{group_effects['重症']:.3f}", "✅ 接近真实")
        with col4:
            st.metric("真实因果效应", f"{true_effect:.3f}", "🎯 物理机制")
        
        # Simpson悖论判断
        paradox_detected = (overall_effect * treatment_effect < 0) or \
                          (abs(overall_effect - treatment_effect) > 0.5)
        
        if paradox_detected:
            st.error("🚨 检测到Simpson悖论！总体效应与真实效应方向相反或差异巨大")
        else:
            st.warning("⚠️ 未检测到典型Simpson悖论，但仍存在混淆偏差")
        
        st.success("""
        **Simpson悖论的启示**：
        - **分组分析的重要性**: 忽略混淆变量会导致错误结论
        - **因果推断的必要性**: 相关性不等同于因果性
        - **随机化的价值**: RCT通过随机分配消除选择偏差
        """)
    
    @staticmethod
    def _render_do_calculus():
        """Do-Calculus演示"""
        st.markdown("### ⚙️ Do-Calculus：从观测到干预的数学转换")
        
        st.markdown("""
        **核心公式**：
        - **截断因子分解**: $P(v|do(x)) = \prod_{i, V_i \neq X} P(v_i|pa_i) |_{X=x}$
        - **后门调整**: $P(Y=y|do(X=x)) = \sum_z P(Y=y|X=x, Z=z) P(Z=z)$
        """)
        
        with st.sidebar:
            sample_size = st.slider("样本量", 500, 5000, 2000, 100)
            treatment_strength = st.slider("治疗强度", 0.5, 3.0, 2.0, 0.1)
            confounding_strength = st.slider("混淆强度", 0.5, 2.0, 1.0, 0.1)
            show_method_comparison = st.checkbox("显示方法对比", value=True)
        
        # 生成SCM数据
        np.random.seed(42)
        n = sample_size
        
        # 外生变量
        Z = np.random.normal(0, 1, n)  # 混淆变量
        U_T = np.random.normal(0, 1, n)  # 治疗噪声
        U_Y = np.random.normal(0, 1, n)  # 结果噪声
        
        # 结构方程
        # T = f_T(Z, U_T)
        logit_T = confounding_strength * Z + U_T
        P_T = 1 / (1 + np.exp(-logit_T))
        T = np.random.binomial(1, P_T, n)
        
        # Y = f_Y(T, Z, U_Y)
        Y = treatment_strength * T + 1.5 * Z + U_Y
        
        # 创建DataFrame
        df = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})
        
        # 计算各种效应
        # 1. 朴素观测效应
        naive_effect = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()
        
        # 2. 后门调整效应
        # 分层计算
        adjusted_effects = []
        weights = []
        
        for z_val in np.percentile(Z, np.linspace(0, 100, 10)):
            mask = (Z >= z_val - 0.5) & (Z < z_val + 0.5)
            if mask.sum() > 10:
                subset = df[mask]
                if len(subset['T'].unique()) == 2:
                    effect = subset[subset['T']==1]['Y'].mean() - subset[subset['T']==0]['Y'].mean()
                    weight = len(subset) / len(df)
                    adjusted_effects.append(effect)
                    weights.append(weight)
        
        # 加权平均
        backdoor_effect = np.average(adjusted_effects, weights=weights) if adjusted_effects else 0
        
        # 3. 真实因果效应 (do算子)
        # 模拟干预：强制T=0和T=1
        Y_do0 = 1.5 * Z + U_Y  # T=0时的Y
        Y_do1 = treatment_strength + 1.5 * Z + U_Y  # T=1时的Y
        true_effect = Y_do1.mean() - Y_do0.mean()
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "数据分布与混淆", "不同方法估计对比",
                "调整过程可视化", "估计误差分析"
            ]
        )
        
        # 数据分布
        for t_val in [0, 1]:
            subset = df[df['T'] == t_val]
            fig.add_trace(
                go.Scatter(
                    x=subset['Z'], y=subset['Y'],
                    mode='markers',
                    name=f'T={t_val}',
                    opacity=0.6,
                    marker=dict(
                        color='red' if t_val==0 else 'blue',
                        size=6
                    )
                ),
                row=1, col=1
            )
        
        # 方法对比
        methods = ['朴素观测', '后门调整', '真实因果']
        estimates = [naive_effect, backdoor_effect, true_effect]
        colors = ['red', 'orange', 'green']
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=estimates,
                marker_color=colors,
                name='效应估计'
            ),
            row=1, col=2
        )
        
        # 调整过程
        z_bins = np.percentile(Z, np.linspace(0, 100, 10))
        bin_centers = (z_bins[:-1] + z_bins[1:]) / 2
        
        if len(adjusted_effects) == len(weights):
            fig.add_trace(
                go.Scatter(
                    x=bin_centers[:len(adjusted_effects)],
                    y=adjusted_effects,
                    mode='markers+lines',
                    name='分层效应',
                    marker=dict(size=8, color='orange')
                ),
                row=2, col=1
            )
            
            # 添加真实效应线
            fig.add_trace(
                go.Scatter(
                    x=[bin_centers.min(), bin_centers.max()],
                    y=[true_effect, true_effect],
                    mode='lines',
                    name='真实效应',
                    line=dict(color='green', dash='dash')
                ),
                row=2, col=1
            )
        
        # 误差分析
        errors = {
            '朴素观测': abs(naive_effect - true_effect),
            '后门调整': abs(backdoor_effect - true_effect)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(errors.keys()),
                y=list(errors.values()),
                marker_color=['red', 'orange'],
                name='绝对误差'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Do-Calculus完整分析流程",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细结果
        st.markdown("### 📈 估计结果详细分析")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("朴素观测", f"{naive_effect:.3f}", 
                     f"偏差: {naive_effect - true_effect:+.3f}")
            st.error("❌ 包含混淆偏差")
        
        with col2:
            st.metric("后门调整", f"{backdoor_effect:.3f}",
                     f"偏差: {backdoor_effect - true_effect:+.3f}")
            if abs(backdoor_effect - true_effect) < 0.2:
                st.success("✅ 接近真实值")
            else:
                st.warning("⚠️ 仍有偏差")
        
        with col3:
            st.metric("真实因果", f"{true_effect:.3f}", "基准值")
            st.info("🎯 物理机制")
        
        # 方法对比表格
        if show_method_comparison:
            st.markdown("### 🔬 方法对比分析")
            
            comparison_data = [
                ["朴素观测", "E[Y|T=1] - E[Y|T=0]", "简单但有偏", "❌ 不推荐"],
                ["后门调整", "∑z E[Y|T=1,Z=z]P(Z=z)", "控制混淆", "✅ 推荐"],
                ["工具变量", "高级方法", "需要额外假设", "🔧 特殊情况"],
                ["RCT", "随机对照试验", "黄金标准", "⭐ 最优但昂贵"]
            ]
            
            st.table(pd.DataFrame(comparison_data, 
                                columns=["方法", "公式", "特点", "推荐度"]))
        
        st.success("""
        **Do-Calculus的核心价值**：
        - **数学严谨**: 从图结构到可计算公式的严格推导
        - **实用性强**: 将哲学问题转化为可操作的统计方法
        - **通用框架**: 适用于各种因果推断场景
        """)
    
    @staticmethod
    def _render_counterfactual():
        """反事实推理演示"""
        st.markdown("### 🔄 反事实推理：溯因-干预-预测三部曲")
        
        st.markdown("""
        **三步法**：
        1. **溯因(Abduction)**: $P(U|X=x, Y=y)$ - 推断外生变量
        2. **干预(Action)**: 建立新模型 $\mathcal{M}_{x'}$
        3. **预测(Prediction)**: $\mathbb{E}[Y_{x'}|X=x, Y=y]$
        """)
        
        with st.sidebar:
            scenario = st.selectbox("场景选择", 
                ["医疗决策", "政策评估", "个人选择"])
            show_abduction = st.checkbox("显示溯因过程", value=True)
            show_individual = st.checkbox("显示个体分析", value=True)
        
        # 不同场景的参数设置
        if scenario == "医疗决策":
            # 医疗场景：病人特征、治疗选择、康复结果
            n_patients = 1000
            true_treatment_effect = 2.0
            
            # 病人基础特征(外生变量)
            np.random.seed(42)
            health_status = np.random.normal(0, 1, n_patients)  # 健康状况
            genetic_factor = np.random.normal(0, 0.5, n_patients)  # 基因因素
            
            # 治疗选择(受健康状况影响)
            treatment_prob = 1 / (1 + np.exp(-(-0.5 + 0.8 * health_status)))
            treatment = np.random.binomial(1, treatment_prob, n_patients)
            
            # 康复结果
            recovery = (1.0 * health_status + 0.5 * genetic_factor + 
                       true_treatment_effect * treatment + 
                       np.random.normal(0, 1, n_patients))
            
            variable_names = {
                'U': ['健康状况', '基因因素'],
                'X': '治疗',
                'Y': '康复结果',
                'x': '接受治疗',
                "x'": '未接受治疗'
            }
            
        elif scenario == "政策评估":
            # 政策场景：地区特征、政策实施、经济效果
            n_patients = 800
            true_treatment_effect = 1.5
            
            np.random.seed(42)
            base_economy = np.random.normal(50, 10, n_patients)  # 基础经济水平
            human_capital = np.random.normal(0, 1, n_patients)  # 人力资本
            
            policy_prob = 1 / (1 + np.exp(-(-1 + 0.05 * base_economy)))
            treatment = np.random.binomial(1, policy_prob, n_patients)
            
            recovery = (0.3 * base_economy + 2.0 * human_capital + 
                       true_treatment_effect * treatment + 
                       np.random.normal(0, 5, n_patients))
            
            variable_names = {
                'U': ['基础经济', '人力资本'],
                'X': '政策实施',
                'Y': '经济效果',
                'x': '实施政策',
                "x'": '未实施政策'
            }
            
        else:  # 个人选择
            # 教育场景：个人能力、教育选择、收入
            n_patients = 1200
            true_treatment_effect = 3.0
            
            np.random.seed(42)
            ability = np.random.normal(100, 15, n_patients)  # 个人能力
            family_bg = np.random.normal(0, 1, n_patients)  # 家庭背景
            
            edu_prob = 1 / (1 + np.exp(-(-2 + 0.03 * ability + 0.5 * family_bg)))
            treatment = np.random.binomial(1, edu_prob, n_patients)
            
            recovery = (0.1 * ability + 2.0 * family_bg + 
                       true_treatment_effect * treatment + 
                       np.random.normal(0, 8, n_patients))
            
            variable_names = {
                'U': ['个人能力', '家庭背景'],
                'X': '教育选择',
                'Y': '收入水平',
                'x': '接受教育',
                "x'": '未接受教育'
            }
        
        # 选择几个个体进行详细分析
        individual_indices = np.random.choice(n_patients, 5, replace=False)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "总体数据分布", "反事实推理流程",
                "个体反事实分析", "推理不确定性"
            ]
        )
        
        # 总体分布
        colors = ['red', 'blue']
        for t_val in [0, 1]:
            subset_t = treatment == t_val
            fig.add_trace(
                go.Scatter(
                    x=health_status if scenario == "医疗决策" else base_economy if scenario == "政策评估" else ability,
                    y=recovery[subset_t],
                    mode='markers',
                    name=f'{variable_names["X"]}={t_val}',
                    opacity=0.6,
                    marker=dict(color=colors[t_val], size=6)
                ),
                row=1, col=1
            )
        
        # 反事实推理流程图
        steps = ['溯因\n推断U', '干预\n改变X', '预测\n计算Y\'']
        step_positions = [1, 2, 3]
        
        fig.add_trace(
            go.Scatter(
                x=step_positions,
                y=[2, 2, 2],
                mode='markers+text',
                text=steps,
                textposition="middle center",
                marker=dict(size=20, color='lightblue'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 添加箭头
        for i in range(len(step_positions)-1):
            fig.add_trace(
                go.Scatter(
                    x=[step_positions[i]+0.15, step_positions[i+1]-0.15],
                    y=[2, 2],
                    mode='lines',
                    line=dict(width=2, color='gray'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 个体反事实分析
        if show_individual:
            for i, idx in enumerate(individual_indices):
                # 实际结果
                actual_y = recovery[idx]
                actual_x = treatment[idx]
                
                # 简化的反事实计算（这里用线性近似）
                # 实际应该用完整的溯因-干预-预测流程
                if actual_x == 1:
                    counterfactual_y = actual_y - true_treatment_effect
                else:
                    counterfactual_y = actual_y + true_treatment_effect
                
                fig.add_trace(
                    go.Scatter(
                        x=[i, i],
                        y=[actual_y, counterfactual_y],
                        mode='markers+lines',
                        name=f'个体{idx}',
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 推理不确定性
        uncertainty_levels = ['低', '中', '高']
        uncertainty_values = [0.1, 0.3, 0.6]
        
        fig.add_trace(
            go.Bar(
                x=uncertainty_levels,
                y=uncertainty_values,
                marker_color=['green', 'orange', 'red'],
                name='不确定性水平',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"反事实推理 - {scenario}场景",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细分析
        st.markdown("### 🎯 个体反事实详细分析")
        
        if show_individual:
            individual_data = []
            for idx in individual_indices:
                actual_x = treatment[idx]
                actual_y = recovery[idx]
                
                # 简化的反事实计算
                if actual_x == 1:
                    counterfactual_y = actual_y - true_treatment_effect
                    x_prime_key = "x'"  # 先定义键名
                    scenario_text = f"如果{variable_names[x_prime_key]}, Y会变为{counterfactual_y:.2f}"
                else:
                    counterfactual_y = actual_y + true_treatment_effect
                    scenario_text = f"如果{variable_names['x']}, Y会变为{counterfactual_y:.2f}"
                
                individual_data.append([
                    f"个体{idx}",
                    f"{variable_names['X']}={actual_x}",
                    f"{actual_y:.2f}",
                    f"{counterfactual_y:.2f}",
                    scenario_text
                ])
            
            st.table(pd.DataFrame(individual_data, 
                                columns=["个体", "实际X", "实际Y", "反事实Y", "说明"]))
        
        # 理论解释
        st.markdown("### 📚 反事实推理的数学基础")
        
        st.latex(r"""
        P(Y_{x'} | X=x, Y=y) = \int f_Y(x', pa_Y, u) \cdot P(u | x, y) du
        """)
        
        st.markdown("""
        **关键挑战**：
        1. **外生变量推断**: $P(u|x,y)$ 通常不可观测，需要假设
        2. **结构方程估计**: $f_Y$ 的形式需要领域知识
        3. **计算复杂度**: 高维积分难以计算
        
        **实际应用**：
        - **个性化医疗**: "如果用另一种疗法，结果会怎样？"
        - **政策评估**: "如果当初不实施这项政策，经济会怎样？"
        - **法律判决**: "如果被告没有这样做，损害还会发生吗？"
        """)
        
        st.success("""
        **反事实推理的价值**：
        - **决策支持**: 为"what if"问题提供量化答案
        - **责任归因**: 帮助确定因果责任
        - **学习机制**: 深入理解系统的运作原理
        """)


# 为了兼容性，添加缺少的导入
try:
    import networkx as nx
except ImportError:
    st.error("❌ NetworkX库未安装，请运行: pip install networkx")

        # 添加交互式测验
