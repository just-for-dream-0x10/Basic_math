"""
交互式机器学习重要曲线可视化
严格按照 10.Important_Curves.md 中的公式实现
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates


class InteractiveMLCurves:
    """交互式机器学习曲线可视化"""
    
    @staticmethod
    @safe_render
    def render():
        st.subheader("📈 机器学习重要曲线")
        st.markdown("""
        **评估曲线**: 可视化模型性能的关键工具
        
        **核心曲线**:
        - **ROC曲线**: 真阳性率 vs 假阳性率
        - **PR曲线**: 精确率 vs 召回率
        - **学习曲线**: 训练/验证误差 vs 样本数
        - **验证曲线**: 模型性能 vs 超参数
        - **混淆矩阵**: 分类结果的完整视图
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择曲线类型")
            curve_type = st.selectbox("曲线类型", [
                "ROC曲线",
                "PR曲线 (Precision-Recall)",
                "学习曲线",
                "验证曲线",
                "混淆矩阵",
                "校准曲线"
            ])
        
        if curve_type == "ROC曲线":
            InteractiveMLCurves._render_roc()
        elif curve_type == "PR曲线 (Precision-Recall)":
            InteractiveMLCurves._render_pr()
        elif curve_type == "学习曲线":
            InteractiveMLCurves._render_learning_curve()
        elif curve_type == "验证曲线":
            InteractiveMLCurves._render_validation_curve()
        elif curve_type == "混淆矩阵":
            InteractiveMLCurves._render_confusion_matrix()
        elif curve_type == "校准曲线":
            InteractiveMLCurves._render_calibration_curve()
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("ml_curves")
        quizzes = QuizTemplates.get_ml_curves_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_roc():
        """ROC曲线可视化"""
        st.markdown("### 📊 ROC曲线 (Receiver Operating Characteristic)")
        
        st.latex(r"""
        \text{TPR} = \frac{TP}{TP + FN} = \text{Recall} = \text{Sensitivity}
        """)
        st.latex(r"""
        \text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
        """)
        
        st.markdown("""
        **ROC曲线**:
        - X轴: 假阳性率 (FPR) - 误报率
        - Y轴: 真阳性率 (TPR) - 召回率
        - AUC (Area Under Curve): 曲线下面积，越大越好
        
        **解读**:
        - AUC = 1.0: 完美分类器
        - AUC = 0.5: 随机猜测
        - AUC < 0.5: 比随机还差
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 数据设置")
            n_samples = st.slider("样本数量", 100, 2000, 500, 100)
            class_separation = st.slider("类别分离度", 0.5, 3.0, 1.5, 0.1,
                                        help="值越大，分类越容易")
            class_imbalance = st.slider("类别不平衡", 0.1, 0.9, 0.5, 0.05,
                                       help="正类占比")
        
        # 生成模拟数据
        np.random.seed(42)
        n_positive = int(n_samples * class_imbalance)
        n_negative = n_samples - n_positive
        
        # 正类得分（较高）
        y_score_pos = np.random.normal(class_separation, 1.0, n_positive)
        # 负类得分（较低）
        y_score_neg = np.random.normal(0, 1.0, n_negative)
        
        y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
        y_score = np.concatenate([y_score_pos, y_score_neg])
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # 创建图表
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("ROC曲线", "得分分布")
        )
        
        # ROC曲线
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines',
                      line=dict(color='blue', width=3),
                      fill='tozeroy',
                      fillcolor='rgba(0, 100, 255, 0.2)',
                      name=f'ROC (AUC={roc_auc:.3f})'),
            row=1, col=1
        )
        
        # 对角线（随机猜测）
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(color='red', width=2, dash='dash'),
                      name='随机猜测 (AUC=0.5)'),
            row=1, col=1
        )
        
        # 得分分布
        fig.add_trace(
            go.Histogram(x=y_score_neg, name='负类', opacity=0.7,
                        marker_color='red', nbinsx=30),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=y_score_pos, name='正类', opacity=0.7,
                        marker_color='blue', nbinsx=30),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="假阳性率 (FPR)", row=1, col=1)
        fig.update_yaxes(title_text="真阳性率 (TPR)", row=1, col=1)
        fig.update_xaxes(title_text="预测得分", row=1, col=2)
        fig.update_yaxes(title_text="样本数", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC", f"{roc_auc:.4f}")
        with col2:
            st.metric("正类数", n_positive)
        with col3:
            st.metric("负类数", n_negative)
        with col4:
            quality = "优秀" if roc_auc > 0.9 else "良好" if roc_auc > 0.8 else "一般" if roc_auc > 0.7 else "较差"
            st.metric("评价", quality)
        
        # 不同阈值下的性能
        st.markdown("### 🎚️ 阈值选择的影响")
        
        # 选择几个关键阈值
        threshold_idx = [len(thresholds)//4, len(thresholds)//2, 3*len(thresholds)//4]
        
        threshold_data = []
        for idx in threshold_idx:
            if idx < len(thresholds):
                thresh = thresholds[idx]
                y_pred = (y_score >= thresh).astype(int)
                
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tp = np.sum((y_true == 1) & (y_pred == 1))
                
                tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                threshold_data.append({
                    '阈值': f'{thresh:.2f}',
                    'TPR': f'{tpr_val:.3f}',
                    'FPR': f'{fpr_val:.3f}',
                    'Precision': f'{precision:.3f}'
                })
        
        import pandas as pd
        df = pd.DataFrame(threshold_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("""
        ### 📚 ROC曲线的应用场景
        
        **适用于**:
        - ✅ 类别平衡或轻微不平衡的数据
        - ✅ 关注整体分类性能
        - ✅ 需要在TPR和FPR之间权衡
        
        **不适用于**:
        - ❌ 严重类别不平衡的数据（用PR曲线）
        - ❌ 更关注精确率而非召回率
        
        **实际应用**:
        - 医疗诊断: 权衡漏诊(FN)和误诊(FP)
        - 垃圾邮件过滤: 避免误判正常邮件
        - 欺诈检测: 在检测率和误报率间平衡
        """)
    
    @staticmethod
    def _render_pr():
        """PR曲线可视化"""
        st.markdown("### 🎯 PR曲线 (Precision-Recall Curve)")
        
        st.latex(r"""
        \text{Precision} = \frac{TP}{TP + FP}
        """)
        st.latex(r"""
        \text{Recall} = \frac{TP}{TP + FN} = \text{TPR}
        """)
        
        st.markdown("""
        **PR曲线**:
        - X轴: 召回率 (Recall) - 找到了多少正样本
        - Y轴: 精确率 (Precision) - 预测为正的有多少是对的
        - AP (Average Precision): 平均精确率，越大越好
        
        **PR vs ROC**:
        - 类别严重不平衡时，PR曲线更有意义
        - PR曲线更关注正类的预测质量
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 数据设置")
            n_samples = st.slider("样本数量", 100, 2000, 1000, 100)
            positive_ratio = st.slider("正类占比", 0.01, 0.5, 0.1, 0.01,
                                      help="模拟不平衡数据")
            model_quality = st.slider("模型质量", 0.5, 3.0, 1.5, 0.1)
        
        # 生成不平衡数据
        np.random.seed(42)
        n_positive = int(n_samples * positive_ratio)
        n_negative = n_samples - n_positive
        
        # 生成得分
        y_score_pos = np.random.normal(model_quality, 1.0, n_positive)
        y_score_neg = np.random.normal(0, 1.0, n_negative)
        
        y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
        y_score = np.concatenate([y_score_pos, y_score_neg])
        
        # 计算PR曲线
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        # 计算ROC用于对比
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # 基线（随机分类器）
        baseline_precision = n_positive / n_samples
        
        # 创建对比图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"PR曲线 (AP={pr_auc:.3f})", f"ROC曲线 (AUC={roc_auc:.3f})")
        )
        
        # PR曲线
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines',
                      line=dict(color='blue', width=3),
                      fill='tozeroy',
                      fillcolor='rgba(0, 100, 255, 0.2)',
                      name=f'PR (AP={pr_auc:.3f})'),
            row=1, col=1
        )
        
        # 基线
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[baseline_precision, baseline_precision],
                      mode='lines',
                      line=dict(color='red', width=2, dash='dash'),
                      name=f'基线 (随机={baseline_precision:.3f})'),
            row=1, col=1
        )
        
        # ROC曲线（对比）
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines',
                      line=dict(color='green', width=3),
                      name=f'ROC (AUC={roc_auc:.3f})'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(color='red', width=2, dash='dash'),
                      name='随机 (0.5)'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="召回率 (Recall)", row=1, col=1)
        fig.update_yaxes(title_text="精确率 (Precision)", row=1, col=1)
        fig.update_xaxes(title_text="假阳性率 (FPR)", row=1, col=2)
        fig.update_yaxes(title_text="真阳性率 (TPR)", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AP (PR)", f"{pr_auc:.4f}")
        with col2:
            st.metric("AUC (ROC)", f"{roc_auc:.4f}")
        with col3:
            st.metric("正类占比", f"{positive_ratio:.1%}")
        with col4:
            st.metric("类别不平衡", f"{n_negative/n_positive:.1f}:1")
        
        st.markdown(f"""
        ### 🔍 PR vs ROC 对比分析
        
        **当前数据特征**:
        - 正类: {n_positive} ({positive_ratio:.1%})
        - 负类: {n_negative} ({1-positive_ratio:.1%})
        - 不平衡比: {n_negative/n_positive:.1f}:1
        
        **观察**:
        - PR曲线更关注正类的预测质量
        - 在不平衡数据下，ROC曲线可能过于"乐观"
        - PR曲线的基线是正类占比 = {baseline_precision:.3f}
        - ROC曲线的基线是 0.5（对角线）
        
        **何时使用PR曲线**:
        - ✅ 严重类别不平衡（如欺诈检测 1:1000）
        - ✅ 更关心精确率（避免误报）
        - ✅ 正类是关注重点
        
        **典型应用场景**:
        - 信息检索: 搜索结果的相关性
        - 推荐系统: 推荐的准确性
        - 异常检测: 检测稀有事件
        - 医学诊断: 罕见疾病筛查
        """)
    
    @staticmethod
    def _render_learning_curve():
        """学习曲线可视化"""
        st.markdown("### 📚 学习曲线 (Learning Curve)")
        
        st.markdown("""
        **学习曲线**: 展示模型性能随训练样本数量的变化
        
        $$\\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Noise}$$
        
        **用途**:
        - 诊断过拟合/欠拟合
        - 判断是否需要更多数据
        - 理解偏差-方差权衡
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 模型设置")
            model_complexity = st.selectbox("模型复杂度", [
                "过简单(欠拟合)", "适中(理想)", "过复杂(过拟合)"
            ])
            
            n_samples_max = st.slider("最大样本数", 100, 1000, 500, 50)
            noise_level = st.slider("数据噪声", 0.0, 2.0, 0.5, 0.1)
        
        # 模拟学习曲线
        train_sizes = np.linspace(10, n_samples_max, 20).astype(int)
        
        if model_complexity == "过简单(欠拟合)":
            # 高偏差：训练和验证误差都高且接近
            train_scores = 0.8 - 0.3 * np.log(train_sizes / 10) + np.random.rand(len(train_sizes)) * 0.05
            val_scores = 0.75 - 0.25 * np.log(train_sizes / 10) + np.random.rand(len(train_sizes)) * 0.05
            diagnosis = "欠拟合 (High Bias)"
            color_train = 'red'
            color_val = 'orange'
        elif model_complexity == "适中(理想)":
            # 理想情况：都收敛到较低误差
            train_scores = 0.95 - 0.5 * np.exp(-train_sizes / 100) + np.random.rand(len(train_sizes)) * 0.02
            val_scores = 0.85 - 0.4 * np.exp(-train_sizes / 100) + np.random.rand(len(train_sizes)) * 0.02
            diagnosis = "理想拟合 (Good Fit)"
            color_train = 'green'
            color_val = 'lightgreen'
        else:  # 过拟合
            # 高方差：训练误差低，验证误差高
            train_scores = 0.98 - 0.6 * np.exp(-train_sizes / 50) + np.random.rand(len(train_sizes)) * 0.01
            val_scores = 0.7 - 0.3 * np.exp(-train_sizes / 200) + np.random.rand(len(train_sizes)) * 0.05
            diagnosis = "过拟合 (High Variance)"
            color_train = 'blue'
            color_val = 'purple'
        
        # 绘制学习曲线
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_scores,
            mode='lines+markers',
            line=dict(color=color_train, width=3),
            marker=dict(size=8),
            name='训练得分'
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_scores,
            mode='lines+markers',
            line=dict(color=color_val, width=3),
            marker=dict(size=8),
            name='验证得分'
        ))
        
        # 填充gap
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_scores, val_scores[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f"学习曲线 - {diagnosis}",
            xaxis_title="训练样本数量",
            yaxis_title="模型得分",
            height=500,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示诊断
        gap = train_scores[-1] - val_scores[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练得分", f"{train_scores[-1]:.3f}")
        with col2:
            st.metric("验证得分", f"{val_scores[-1]:.3f}")
        with col3:
            st.metric("Gap", f"{gap:.3f}")
        
        # 诊断建议
        st.markdown(f"### 🔍 诊断: {diagnosis}")
        
        if model_complexity == "过简单(欠拟合)":
            st.markdown("""
            **特征**:
            - ❌ 训练得分和验证得分都较低
            - ❌ 两条曲线接近但都不理想
            - ❌ 增加数据帮助不大
            
            **原因**: 模型容量不足，无法捕捉数据的复杂模式
            
            **解决方案**:
            - ✅ 增加模型复杂度（更深的网络、更多特征）
            - ✅ 减少正则化强度
            - ✅ 增加多项式特征
            - ✅ 使用更强大的模型
            """)
        elif model_complexity == "适中(理想)":
            st.markdown("""
            **特征**:
            - ✅ 训练得分和验证得分都较高
            - ✅ Gap较小且稳定
            - ✅ 曲线趋于收敛
            
            **状态**: 模型达到良好平衡
            
            **建议**:
            - ✅ 当前模型已经很好
            - ✅ 可以尝试微调超参数进一步优化
            - ✅ 如果需要更高性能，考虑集成方法
            """)
        else:  # 过拟合
            st.markdown("""
            **特征**:
            - ❌ 训练得分很高，验证得分较低
            - ❌ Gap很大（高方差）
            - ❌ 增加数据会有帮助
            
            **原因**: 模型过于复杂，记住了训练数据的噪声
            
            **解决方案**:
            - ✅ 收集更多训练数据
            - ✅ 增加正则化 (L1/L2/Dropout)
            - ✅ 减少模型复杂度
            - ✅ 早停 (Early Stopping)
            - ✅ 数据增强
            """)
    
    @staticmethod
    def _render_validation_curve():
        """验证曲线可视化"""
        st.markdown("### 🎚️ 验证曲线 (Validation Curve)")
        
        st.markdown("""
        **验证曲线**: 展示模型性能随超参数变化的趋势
        
        **用途**:
        - 选择最优超参数
        - 理解超参数的影响
        - 避免过拟合/欠拟合
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 超参数类型")
            param_type = st.selectbox("参数类型", [
                "正则化强度 (C/λ)",
                "树的深度",
                "学习率",
                "隐藏层大小"
            ])
        
        # 根据参数类型生成曲线
        if param_type == "正则化强度 (C/λ)":
            param_range = np.logspace(-4, 4, 20)
            param_name = "正则化参数"
            xaxis_type = "log"
            
            # 模拟：λ太小→过拟合，λ太大→欠拟合
            train_scores = 1.0 - 0.5 / (1 + param_range)
            val_scores = 0.85 - 0.3 / (1 + param_range) - 0.2 * np.log10(param_range + 0.1)**2 / 10
            
        elif param_type == "树的深度":
            param_range = np.arange(1, 21)
            param_name = "树的深度"
            xaxis_type = "linear"
            
            # 深度太小→欠拟合，深度太大→过拟合
            train_scores = 1.0 - 0.8 * np.exp(-param_range / 3)
            val_scores = 0.9 - 0.5 * np.exp(-param_range / 3) - 0.3 * (param_range / 20)**2
            
        elif param_type == "学习率":
            param_range = np.logspace(-4, 0, 20)
            param_name = "学习率"
            xaxis_type = "log"
            
            # 学习率太小→收敛慢，太大→不稳定
            optimal_lr = 0.01
            train_scores = 0.95 - 0.5 * np.abs(np.log10(param_range / optimal_lr))
            val_scores = 0.85 - 0.6 * np.abs(np.log10(param_range / optimal_lr))
            
        else:  # 隐藏层大小
            param_range = np.arange(10, 210, 10)
            param_name = "隐藏层神经元数"
            xaxis_type = "linear"
            
            # 太小→欠拟合，太大→过拟合
            train_scores = 1.0 - 0.6 * np.exp(-param_range / 30)
            val_scores = 0.9 - 0.4 * np.exp(-param_range / 30) - 0.2 * (param_range / 200)**2
        
        # 找到最优参数
        best_idx = np.argmax(val_scores)
        best_param = param_range[best_idx]
        
        # 绘制验证曲线
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=param_range, y=train_scores,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
            name='训练得分'
        ))
        
        fig.add_trace(go.Scatter(
            x=param_range, y=val_scores,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6),
            name='验证得分'
        ))
        
        # 标注最优点
        fig.add_vline(x=best_param, line_dash="dash", line_color="green",
                     annotation_text=f"最优: {best_param:.4f}")
        
        fig.update_layout(
            title=f"验证曲线 - {param_type}",
            xaxis_title=param_name,
            yaxis_title="模型得分",
            xaxis_type=xaxis_type,
            height=500,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示最优参数
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最优参数", f"{best_param:.4f}")
        with col2:
            st.metric("训练得分", f"{train_scores[best_idx]:.3f}")
        with col3:
            st.metric("验证得分", f"{val_scores[best_idx]:.3f}")
        
        st.markdown("""
        ### 🎯 如何使用验证曲线
        
        **观察要点**:
        1. **欠拟合区域**（左侧）: 训练和验证得分都低
        2. **理想区域**（中间）: 验证得分达到峰值
        3. **过拟合区域**（右侧）: 训练得分高，验证得分下降
        
        **调参策略**:
        - 找到验证得分最高的参数值
        - 观察训练-验证gap的变化
        - 如果gap很大，考虑增加正则化
        - 如果两条曲线都低，考虑增加模型容量
        
        **常见参数**:
        - 正则化: λ/C (SVM, 逻辑回归)
        - 树模型: max_depth, min_samples_split
        - 神经网络: learning_rate, hidden_size, dropout
        - kNN: n_neighbors
        """)
    
    @staticmethod
    def _render_confusion_matrix():
        """混淆矩阵可视化"""
        st.markdown("### 📊 混淆矩阵 (Confusion Matrix)")
        
        st.markdown("""
        **混淆矩阵**: 分类结果的完整展示
        """)
        
        st.latex(r"""
        \begin{bmatrix}
        TN & FP \\
        FN & TP
        \end{bmatrix}
        """)
        
        st.markdown("""
        **派生指标**:
        - Accuracy = (TP + TN) / Total
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 Score = 2 × Precision × Recall / (Precision + Recall)
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 混淆矩阵设置")
            n_samples = st.slider("样本数", 100, 1000, 500)
            accuracy = st.slider("模型准确率", 0.5, 0.99, 0.85, 0.01)
            
            st.markdown("### 偏向设置")
            bias_type = st.radio("模型偏向", [
                "无偏向", "偏向预测正类", "偏向预测负类"
            ])
        
        # 生成混淆矩阵
        np.random.seed(42)
        n_positive = n_samples // 2
        n_negative = n_samples - n_positive
        
        if bias_type == "无偏向":
            tp = int(n_positive * accuracy)
            tn = int(n_negative * accuracy)
            fn = n_positive - tp
            fp = n_negative - tn
        elif bias_type == "偏向预测正类":
            # 高召回率，低精确率
            tp = int(n_positive * accuracy * 1.1)
            tp = min(tp, n_positive)
            fn = n_positive - tp
            fp = int(n_negative * (1 - accuracy) * 1.5)
            tn = n_negative - fp
        else:  # 偏向预测负类
            # 低召回率，高精确率
            tp = int(n_positive * accuracy * 0.7)
            fn = n_positive - tp
            fp = int(n_negative * (1 - accuracy) * 0.5)
            tn = n_negative - fp
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # 计算指标
        total = tp + tn + fp + fn
        acc = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 绘制混淆矩阵
        labels = ['负类 (0)', '正类 (1)']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['预测: 负类', '预测: 正类'],
            y=['实际: 负类', '实际: 正类'],
            colorscale='Blues',
            text=[[f'TN<br>{tn}', f'FP<br>{fp}'],
                  [f'FN<br>{fn}', f'TP<br>{tp}']],
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=True
        ))
        
        fig.update_layout(
            title="混淆矩阵",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示指标
        st.markdown("### 📊 性能指标")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1 Score", f"{f1:.3f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("TP", tp, help="真阳性")
        with col2:
            st.metric("TN", tn, help="真阴性")
        with col3:
            st.metric("FP", fp, help="假阳性", delta_color="inverse")
        with col4:
            st.metric("FN", fn, help="假阴性", delta_color="inverse")
        
        st.markdown(f"""
        ### 🔍 错误分析
        
        **错误类型**:
        - **假阳性 (FP = {fp})**: 将负类错判为正类
          - 影响: 降低精确率
          - 后果: 误报，浪费资源
          
        - **假阴性 (FN = {fn})**: 将正类错判为负类
          - 影响: 降低召回率
          - 后果: 漏报，错失目标
        
        **权衡考虑**:
        - 医疗诊断: FN代价高（漏诊），宁可FP高（过度检查）
        - 垃圾邮件: FP代价高（误删重要邮件），可容忍FN
        - 欺诈检测: 需要平衡，使用F1 Score
        
        **当前状态**: {bias_type}
        - Precision: {precision:.3f} - {"高" if precision > 0.8 else "中" if precision > 0.6 else "低"}精确率
        - Recall: {recall:.3f} - {"高" if recall > 0.8 else "中" if recall > 0.6 else "低"}召回率
        - F1: {f1:.3f} - 综合表现{"良好" if f1 > 0.8 else "一般" if f1 > 0.6 else "较差"}
        """)
    
    @staticmethod
    def _render_calibration_curve():
        """校准曲线可视化"""
        st.markdown("### 🎲 校准曲线 (Calibration Curve)")
        
        st.markdown("""
        **校准**: 模型预测的概率是否可靠
        
        **理想情况**: 如果模型预测某事件概率为70%，那么在所有这样预测的情况中，
        实际发生的比例也应该是70%
        
        **应用**: 当你需要可解释的概率输出时（医疗、金融决策）
        """)
        
        with st.sidebar:
            st.markdown("### 🎛️ 模型类型")
            model_type = st.selectbox("模型校准程度", [
                "良好校准",
                "过度自信",
                "不够自信"
            ])
        
        # 生成预测概率和真实标签
        np.random.seed(42)
        n_samples = 1000
        
        predicted_probs = np.random.beta(2, 2, n_samples)  # 生成[0,1]的概率
        
        if model_type == "良好校准":
            # 真实概率接近预测概率
            true_probs = predicted_probs + np.random.normal(0, 0.1, n_samples)
            true_probs = np.clip(true_probs, 0, 1)
        elif model_type == "过度自信":
            # 预测概率更极端
            true_probs = 0.5 + 0.3 * (predicted_probs - 0.5)
        else:  # 不够自信
            # 预测概率更保守
            true_probs = 0.5 + 1.5 * (predicted_probs - 0.5)
            true_probs = np.clip(true_probs, 0, 1)
        
        # 生成真实标签
        y_true = (np.random.rand(n_samples) < true_probs).astype(int)
        
        # 计算校准曲线
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        fraction_of_positives = []
        mean_predicted_value = []
        
        for i in range(n_bins):
            mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i+1])
            if np.sum(mask) > 0:
                fraction_of_positives.append(np.mean(y_true[mask]))
                mean_predicted_value.append(np.mean(predicted_probs[mask]))
            else:
                fraction_of_positives.append(0)
                mean_predicted_value.append(bin_centers[i])
        
        # 绘制校准曲线
        fig = go.Figure()
        
        # 完美校准线
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name='完美校准'
        ))
        
        # 实际校准曲线
        fig.add_trace(go.Scatter(
            x=mean_predicted_value, y=fraction_of_positives,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10),
            name='模型校准'
        ))
        
        fig.update_layout(
            title=f"校准曲线 - {model_type}",
            xaxis_title="预测概率",
            yaxis_title="实际发生比例",
            height=500,
            xaxis_range=[0, 1],
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 计算校准误差
        calibration_error = np.mean(np.abs(np.array(fraction_of_positives) - np.array(mean_predicted_value)))
        
        st.metric("平均校准误差", f"{calibration_error:.4f}")
        
        st.markdown(f"""
        ### 🔍 校准分析
        
        **{model_type}**:
        
        {"✅ 模型输出的概率值可靠，可以直接使用" if model_type == "良好校准" else
         "⚠️ 模型过于自信，实际概率低于预测" if model_type == "过度自信" else
         "⚠️ 模型过于保守，实际概率高于预测"}
        
        **校准方法**:
        - **Platt Scaling**: 在输出上训练逻辑回归
        - **Isotonic Regression**: 非参数校准方法
        - **Temperature Scaling**: 神经网络常用
        
        **何时需要校准**:
        - 决策树/随机森林: 通常需要校准
        - 朴素贝叶斯: 通常过度自信
        - 神经网络: 可能需要温度缩放
        - SVM: 需要Platt Scaling
        
        **应用场景**:
        - 医疗诊断: 需要准确的风险评估
        - 保险定价: 基于概率计算保费
        - 天气预报: "70%降雨概率"的可信度
        """)
