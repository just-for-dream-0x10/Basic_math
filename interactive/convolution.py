"""
交互式卷积操作可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import convolve2d


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveConvolution:
    """交互式卷积可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🔲 交互式卷积操作")
        st.markdown("实时调整卷积核，观察卷积结果")
        
        with st.sidebar:
            st.markdown("### 🎛️ 卷积核设置")
            kernel_type = st.selectbox("预设卷积核", 
                ["自定义", "边缘检测(垂直)", "边缘检测(水平)", "边缘检测(全方向)",
                 "锐化", "高斯模糊", "均值模糊", "浮雕", "Sobel X", "Sobel Y"])
            
            kernel_size = st.selectbox("卷积核大小", [3, 5], index=0)
            
            st.markdown("### 🖼️ 图像设置")
            image_type = st.radio("选择图像", ["示例图案", "上传图片"])
            
            if image_type == "示例图案":
                pattern = st.selectbox("图案类型", 
                    ["棋盘格", "渐变", "圆形", "条纹", "噪声"])
            else:
                uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])
        
        # 创建卷积核
        kernel = InteractiveConvolution._get_kernel(kernel_type, kernel_size)
        
        # 显示卷积核
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 当前卷积核")
            st.dataframe(kernel, use_container_width=True)
        
        with col2:
            st.markdown("#### 卷积核热力图")
            fig_kernel = go.Figure(data=go.Heatmap(
                z=kernel,
                colorscale='RdBu',
                zmid=0,
                text=np.round(kernel, 2),
                texttemplate='%{text}',
                textfont={"size": 14}
            ))
            fig_kernel.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_kernel, use_container_width=True)
        
        st.markdown("---")
        
        # 生成或加载图像
        if image_type == "示例图案":
            image = InteractiveConvolution._generate_pattern(pattern, size=200)
        else:
            if 'uploaded_file' in locals() and uploaded_file is not None:
                from PIL import Image
                pil_image = Image.open(uploaded_file).convert('L')
                image = np.array(pil_image) / 255.0
            else:
                image = InteractiveConvolution._generate_pattern("棋盘格", size=200)
        
        # 执行卷积
        conv_result = convolve2d(image, kernel, mode='same', boundary='symm')
        conv_result = (conv_result - conv_result.min()) / (conv_result.max() - conv_result.min() + 1e-8)
        
        # 显示结果
        st.markdown("### 📊 卷积结果对比")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 原始图像")
            fig1 = go.Figure(data=go.Heatmap(z=image, colorscale='gray', showscale=False))
            fig1.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            fig1.update_xaxes(showticklabels=False)
            fig1.update_yaxes(showticklabels=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("#### 卷积结果")
            fig2 = go.Figure(data=go.Heatmap(z=conv_result, colorscale='gray', showscale=False))
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            fig2.update_xaxes(showticklabels=False)
            fig2.update_yaxes(showticklabels=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.markdown("#### 差异图")
            diff = np.abs(conv_result - image)
            fig3 = go.Figure(data=go.Heatmap(z=diff, colorscale='Hot', showscale=False))
            fig3.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            fig3.update_xaxes(showticklabels=False)
            fig3.update_yaxes(showticklabels=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        # 统计信息
        st.markdown("### 📈 统计信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("原始均值", f"{np.mean(image):.3f}")
        with col2:
            st.metric("卷积后均值", f"{np.mean(conv_result):.3f}")
        with col3:
            st.metric("最大响应", f"{np.max(np.abs(conv_result)):.3f}")
        with col4:
            st.metric("标准差变化", f"{np.std(conv_result)/np.std(image):.2f}x")
    

        # 添加交互式测验

        # 添加交互式测验
        quiz_system = QuizSystem("convolution")
        quizzes = QuizTemplates.get_convolution_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _get_kernel(kernel_type, size):
        """获取预设卷积核"""
        if size == 3:
            kernels = {
                "自定义": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                "边缘检测(垂直)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                "边缘检测(水平)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                "边缘检测(全方向)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                "锐化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
                "高斯模糊": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
                "均值模糊": np.ones((3, 3)) / 9,
                "浮雕": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
                "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            }
        else:  # size == 5
            kernels = {
                "自定义": np.eye(5),
                "边缘检测(垂直)": np.array([[-1, -2, 0, 2, 1],
                                          [-2, -3, 0, 3, 2],
                                          [-3, -4, 0, 4, 3],
                                          [-2, -3, 0, 3, 2],
                                          [-1, -2, 0, 2, 1]]) / 32,
                "边缘检测(水平)": np.array([[-1, -2, -3, -2, -1],
                                          [-2, -3, -4, -3, -2],
                                          [0, 0, 0, 0, 0],
                                          [2, 3, 4, 3, 2],
                                          [1, 2, 3, 2, 1]]) / 32,
                "边缘检测(全方向)": np.array([[-1, -1, -1, -1, -1],
                                            [-1, 2, 2, 2, -1],
                                            [-1, 2, 8, 2, -1],
                                            [-1, 2, 2, 2, -1],
                                            [-1, -1, -1, -1, -1]]),
                "锐化": np.array([[0, 0, -1, 0, 0],
                                [0, -1, -2, -1, 0],
                                [-1, -2, 17, -2, -1],
                                [0, -1, -2, -1, 0],
                                [0, 0, -1, 0, 0]]),
                "高斯模糊": np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]]) / 256,
                "均值模糊": np.ones((5, 5)) / 25,
                "浮雕": np.array([[-2, -1, -1, -1, 0],
                                 [-1, -1, -1, 0, 1],
                                 [-1, -1, 0, 1, 1],
                                 [-1, 0, 1, 1, 1],
                                 [0, 1, 1, 1, 2]]),
                "Sobel X": np.array([[-1, -2, 0, 2, 1],
                                    [-2, -3, 0, 3, 2],
                                    [-3, -4, 0, 4, 3],
                                    [-2, -3, 0, 3, 2],
                                    [-1, -2, 0, 2, 1]]) / 32,
                "Sobel Y": np.array([[-1, -2, -3, -2, -1],
                                    [-2, -3, -4, -3, -2],
                                    [0, 0, 0, 0, 0],
                                    [2, 3, 4, 3, 2],
                                    [1, 2, 3, 2, 1]]) / 32
            }
        return kernels.get(kernel_type, kernels["自定义"]).astype(float)
    
    @staticmethod
    def _generate_pattern(pattern_type, size=200):
        """生成测试图案"""
        if pattern_type == "棋盘格":
            pattern = np.zeros((size, size))
            block_size = size // 8
            for i in range(8):
                for j in range(8):
                    if (i + j) % 2 == 0:
                        pattern[i*block_size:(i+1)*block_size, 
                               j*block_size:(j+1)*block_size] = 1
            return pattern
        
        elif pattern_type == "渐变":
            x = np.linspace(0, 1, size)
            y = np.linspace(0, 1, size)
            X, Y = np.meshgrid(x, y)
            return (X + Y) / 2
        
        elif pattern_type == "圆形":
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            pattern = np.zeros((size, size))
            pattern[R < 0.3] = 1
            pattern[(R >= 0.3) & (R < 0.6)] = 0.5
            return pattern
        
        elif pattern_type == "条纹":
            pattern = np.zeros((size, size))
            stripe_width = size // 10
            for i in range(0, size, stripe_width * 2):
                pattern[:, i:i+stripe_width] = 1
            return pattern
        
        elif pattern_type == "噪声":
            return np.random.rand(size, size)
        
        else:
            return np.zeros((size, size))
