"""
交互式SVM分类器可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveSVM:
    """交互式SVM可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🎯 交互式SVM分类器")
        st.markdown("实时调整参数，观察决策边界变化")
        
        with st.sidebar:
            st.markdown("### 📊 SVM参数")
            C = st.slider("C (正则化参数)", 0.01, 10.0, 1.0, 0.1,
                         help="C越大，对误分类的惩罚越大")
            kernel = st.selectbox("核函数", ["linear", "rbf", "poly"])
            
            if kernel == "rbf":
                gamma = st.slider("Gamma", 0.01, 10.0, 1.0, 0.1,
                                 help="RBF核的参数，控制单个样本的影响范围")
            elif kernel == "poly":
                degree = st.slider("多项式度数", 2, 5, 3, 1)
            
            st.markdown("### 🎲 数据设置")
            n_samples = st.slider("样本数量", 20, 200, 100, 10)
            noise = st.slider("噪声水平", 0.0, 1.0, 0.2, 0.05)
            separation = st.slider("类别分离度", 0.5, 3.0, 1.5, 0.1)
        
        # 生成数据
        np.random.seed(42)
        X, y = InteractiveSVM._generate_data(n_samples, noise, separation)
        
        # 训练SVM
        try:
            from sklearn import svm
            
            if kernel == "rbf":
                clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
            elif kernel == "poly":
                clf = svm.SVC(kernel=kernel, C=C, degree=degree)
            else:
                clf = svm.SVC(kernel=kernel, C=C)
            
            clf.fit(X, y)
            
            # 可视化
            fig = InteractiveSVM._visualize_svm(X, y, clf)
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示统计信息
            st.markdown("### 📊 模型信息")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("支持向量数", len(clf.support_))
            with col2:
                accuracy = clf.score(X, y)
                st.metric("训练准确率", f"{accuracy*100:.1f}%")
            with col3:
                st.metric("类别0样本", f"{np.sum(y==0)}")
                st.metric("类别1样本", f"{np.sum(y==1)}")
        
        except ImportError:
            st.error("需要安装 scikit-learn: pip install scikit-learn")
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("svm")
        quizzes = QuizTemplates.get_svm_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _generate_data(n_samples, noise, separation):
        """生成二分类数据"""
        n_half = n_samples // 2
        class_0 = np.random.randn(n_half, 2) * (1 + noise) - separation
        class_1 = np.random.randn(n_half, 2) * (1 + noise) + separation
        X = np.vstack([class_0, class_1])
        y = np.hstack([np.zeros(n_half), np.ones(n_half)])
        return X, y
    
    @staticmethod
    def _visualize_svm(X, y, clf):
        """可视化SVM决策边界"""
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # 预测
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        
        # 决策边界等高线
        fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale='RdBu',
            showscale=False,
            contours=dict(
                start=-1, end=1, size=0.5,
                showlabels=True
            ),
            opacity=0.3
        ))
        
        # 数据点
        fig.add_trace(go.Scatter(
            x=X[y==0, 0], y=X[y==0, 1],
            mode='markers',
            marker=dict(color='blue', size=8, line=dict(color='black', width=1)),
            name='类别 0'
        ))
        
        fig.add_trace(go.Scatter(
            x=X[y==1, 0], y=X[y==1, 1],
            mode='markers',
            marker=dict(color='red', size=8, line=dict(color='black', width=1)),
            name='类别 1'
        ))
        
        # 支持向量
        fig.add_trace(go.Scatter(
            x=X[clf.support_, 0], y=X[clf.support_, 1],
            mode='markers',
            marker=dict(size=15, color='yellow', 
                       line=dict(color='black', width=2)),
            name='支持向量'
        ))
        
        fig.update_layout(
            title="SVM决策边界",
            xaxis_title="特征 1",
            yaxis_title="特征 2",
            height=500
        )
        
        return fig
