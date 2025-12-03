"""
信号处理与时频分析交互式可视化
严格按照 25.Singal_processing.md 中的理论实现

核心内容：
1. 傅里叶变换的局限性
2. 短时傅里叶变换(STFT)与声谱图
3. 海森堡不确定性原理
4. 小波变换与多分辨率分析
5. 状态空间模型(SSM/Mamba)
6. MFCC与复数神经网络
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal
from scipy.fft import fft, fftfreq


import sys
import os

# 添加父目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.error_handler import safe_render
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveSignalProcessing:
    """交互式信号处理可视化"""
    
    @staticmethod
    @safe_render

    def render():
        st.subheader("🌊 信号处理：从时域到频域的数学之旅")
        
        st.markdown(r"""
        **核心思想**: 机器学习的数据源（声音、脑电波、股市）本质上是**波**
        
        **关键问题**:
        - 傅里叶变换告诉你"有什么频率"，但不知道"何时发生"
        - 如何同时获得时间和频率信息？→ **海森堡不确定性**
        - 如何处理非平稳信号？→ **小波变换**
        - 如何建模长序列？→ **状态空间模型(SSM/Mamba)**
        
        **应用**: 语音识别、音乐分类、脑机接口、时间序列预测
        """)
        
        with st.sidebar:
            st.markdown("### 📊 选择演示")
            demo_type = st.selectbox(
                "演示类型",
                [
                    "傅里叶变换的局限",
                    "STFT与声谱图",
                    "海森堡不确定性原理",
                    "小波变换",
                    "状态空间模型(SSM)",
                    "MFCC与梅尔刻度",
                    "完整流程对比"
                ]
            )
        
        if demo_type == "傅里叶变换的局限":
            InteractiveSignalProcessing._render_fourier_limits()
        elif demo_type == "STFT与声谱图":
            InteractiveSignalProcessing._render_stft()
        elif demo_type == "海森堡不确定性原理":
            InteractiveSignalProcessing._render_heisenberg()
        elif demo_type == "小波变换":
            InteractiveSignalProcessing._render_wavelet()
        elif demo_type == "状态空间模型(SSM)":
            InteractiveSignalProcessing._render_ssm()
        elif demo_type == "MFCC与梅尔刻度":
            InteractiveSignalProcessing._render_mfcc()
        elif demo_type == "完整流程对比":
            InteractiveSignalProcessing._render_comparison()
    

        # 添加交互式测验
        quiz_system = QuizSystem("signal_processing")
        quizzes = QuizTemplates.get_signal_processing_quizzes()
        quiz_system.render_quiz(quizzes)
    @staticmethod
    def _render_fourier_limits():
        """傅里叶变换的局限性可视化"""
        st.markdown("### 🎵 傅里叶变换的致命缺陷：时间信息丢失")
        
        st.markdown(r"""
        **标准傅里叶变换**:
        """)
        
        st.latex(r"""
        X(\omega) = \int_{-\infty}^{\infty} x(t) e^{-j\omega t} dt
        """)
        
        st.markdown(r"""
        **问题**: 积分区间是 $(-\infty, \infty)$ → 全局分析
        
        **例子**: 
        - "先钢琴后小提琴" vs "钢琴小提琴同时演奏"
        - 全局频谱可能完全相同！
        - **无法区分时序**
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            signal_type = st.selectbox(
                "信号类型",
                ["线性调频(Chirp)", "分段频率", "钢琴+小提琴"]
            )
            duration = st.slider("信号时长(秒)", 1.0, 5.0, 2.0, 0.5)
            fs = st.slider("采样率(Hz)", 500, 2000, 1000, 100)
        
        # 生成时间轴
        t = np.linspace(0, duration, int(duration * fs))
        
        # 生成不同类型的信号
        if signal_type == "线性调频(Chirp)":
            # Chirp信号：频率从50Hz线性增加到200Hz
            x = signal.chirp(t, f0=50, f1=200, t1=duration, method='linear')
            title_suffix = "频率从50Hz→200Hz"
            
        elif signal_type == "分段频率":
            # 前半段50Hz，后半段150Hz
            split = len(t) // 2
            x = np.concatenate([
                np.sin(2 * np.pi * 50 * t[:split]),
                np.sin(2 * np.pi * 150 * t[split:])
            ])
            title_suffix = "前50Hz后150Hz"
            
        else:  # 钢琴+小提琴
            # 模拟：先钢琴(C4=262Hz)再小提琴(A4=440Hz)
            split = len(t) // 2
            piano = np.sin(2 * np.pi * 262 * t[:split])
            violin = np.sin(2 * np.pi * 440 * t[split:])
            x = np.concatenate([piano, violin])
            title_suffix = "先钢琴(262Hz)后小提琴(440Hz)"
        
        # 计算FFT
        X = fft(x)
        freqs = fftfreq(len(t), 1/fs)
        
        # 只取正频率
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        X_pos = np.abs(X[pos_mask])
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"时域信号: {title_suffix}",
                "频域(FFT): 时间信息完全丢失！"
            ),
            vertical_spacing=0.15,
            specs=[[{"type": "xy"}], [{"type": "xy"}]]
        )
        
        # 1. 时域波形
        fig.add_trace(
            go.Scatter(
                x=t,
                y=x,
                mode='lines',
                name='时域信号',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # 标注时间段（如果是分段信号）
        if signal_type in ["分段频率", "钢琴+小提琴"]:
            mid_time = duration / 2
            fig.add_vline(x=mid_time, line_dash="dash", line_color="red",
                         annotation_text="频率变化点",
                         row=1, col=1)
        
        # 2. 频域
        fig.add_trace(
            go.Scatter(
                x=freqs_pos,
                y=X_pos,
                mode='lines',
                name='频谱',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="时间 (秒)", row=1, col=1)
        fig.update_yaxes(title_text="幅度", row=1, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", range=[0, 500], row=2, col=1)
        fig.update_yaxes(title_text="幅度", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="傅里叶变换的局限性"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 问题分析")
        
        # 找到主要频率成分
        peak_indices = signal.find_peaks(X_pos, height=np.max(X_pos)*0.1)[0]
        main_freqs = freqs_pos[peak_indices]
        
        st.info(f"""
        **FFT检测到的主要频率**: {', '.join([f'{f:.1f} Hz' for f in main_freqs[:5]])}
        
        **问题**:
        - ❌ FFT只告诉你"存在哪些频率"
        - ❌ 无法知道"何时发生"
        - ❌ 无法区分"先后顺序"还是"同时发生"
        
        **解决方案**: 短时傅里叶变换(STFT) → 下一个演示
        """)
        
        st.success(r"""
        **数学本质**:
        
        傅里叶变换是**内积**: $X(\omega) = \langle x(t), e^{j\omega t} \rangle$
        
        - 它测量信号与**无限长正弦波**的相似度
        - 这些基函数 $e^{j\omega t}$ 从 $-\infty$ 延伸到 $+\infty$
        - 因此天然丢失了时间局部化信息
        
        **要点**: 不是傅里叶变换"不好"，而是它**设计目的**就是全局频域分析
        """)
    
    @staticmethod
    def _render_stft():
        """STFT与声谱图可视化"""
        st.markdown("### 📸 短时傅里叶变换：给信号拍照")
        
        st.markdown(r"""
        **STFT定义**: 加窗口的傅里叶变换
        """)
        
        st.latex(r"""
        STFT(t, \omega) = \int_{-\infty}^{\infty} x(\tau) w(\tau - t) e^{-j\omega \tau} d\tau
        """)
        
        st.markdown(r"""
        **核心思想**:
        - 窗函数 $w(t)$ 滑动扫描信号
        - 每个时刻做局部傅里叶变换
        - 结果: 二维矩阵 (时间 × 频率)
        
        **声谱图**: $|STFT(t, \omega)|^2$ → 这是一张**图像**！
        
        **这就是为什么CNN可以处理音频分类**
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            signal_choice = st.selectbox(
                "信号类型",
                ["线性调频", "音乐音符序列", "语音模拟"]
            )
            window_size = st.slider("窗口大小(样本数)", 32, 512, 128, 32)
            overlap = st.slider("重叠比例", 0.0, 0.9, 0.75, 0.05)
        
        # 生成信号
        fs = 1000
        duration = 2.0
        t = np.linspace(0, duration, int(duration * fs))
        
        if signal_choice == "线性调频":
            x = signal.chirp(t, f0=50, f1=300, t1=duration, method='linear')
            title = "Chirp: 50Hz→300Hz"
        elif signal_choice == "音乐音符序列":
            # C-D-E-F-G (Do Re Mi Fa Sol)
            notes = [262, 294, 330, 349, 392]  # Hz
            x = np.zeros_like(t)
            segment_len = len(t) // 5
            for i, freq in enumerate(notes):
                start = i * segment_len
                end = (i + 1) * segment_len if i < 4 else len(t)
                x[start:end] = np.sin(2 * np.pi * freq * t[start:end])
            title = "音乐音符: C-D-E-F-G"
        else:  # 语音模拟
            # 模拟：基频+共振峰
            fundamental = 150  # 基频
            formants = [800, 1200, 2500]  # 共振峰
            x = np.sin(2 * np.pi * fundamental * t)
            for f in formants:
                x += 0.3 * np.sin(2 * np.pi * f * t)
            title = "语音模拟: 基频+共振峰"
        
        # 计算STFT
        noverlap = int(window_size * overlap)
        f, t_stft, Zxx = signal.stft(x, fs=fs, nperseg=window_size, noverlap=noverlap)
        
        # 声谱图（取模的平方，转为dB）
        spectrogram = np.abs(Zxx)
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # 可视化
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"时域波形: {title}",
                f"声谱图(窗口={window_size}, 重叠={overlap:.0%})",
                "频率随时间演化(切片)"
            ),
            vertical_spacing=0.1,
            specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]]
        )
        
        # 1. 时域
        fig.add_trace(
            go.Scatter(
                x=t,
                y=x,
                mode='lines',
                name='时域信号',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # 2. 声谱图（热力图）
        fig.add_trace(
            go.Heatmap(
                x=t_stft,
                y=f,
                z=spectrogram_db,
                colorscale='Jet',
                colorbar=dict(title="dB", y=0.5, len=0.6),
                showscale=True
            ),
            row=2, col=1
        )
        
        # 3. 频率切片（选几个时刻）
        time_slices = [0.25, 0.5, 0.75, 1.0, 1.5]
        colors = px.colors.qualitative.Set1
        
        for i, time_point in enumerate(time_slices):
            # 找最近的时间索引
            idx = np.argmin(np.abs(t_stft - time_point))
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=spectrogram[:, idx],
                    mode='lines',
                    name=f't={time_point:.2f}s',
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=3, col=1
            )
        
        fig.update_xaxes(title_text="时间 (s)", row=1, col=1)
        fig.update_yaxes(title_text="幅度", row=1, col=1)
        fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
        fig.update_yaxes(title_text="频率 (Hz)", range=[0, 500], row=2, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", range=[0, 500], row=3, col=1)
        fig.update_yaxes(title_text="幅度", row=3, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="短时傅里叶变换(STFT)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 声谱图解读")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("时间分辨率", f"{1000/fs * window_size:.1f} ms")
        
        with col2:
            freq_resolution = fs / window_size
            st.metric("频率分辨率", f"{freq_resolution:.2f} Hz")
        
        with col3:
            st.metric("声谱图尺寸", f"{len(t_stft)} × {len(f)}")
        
        st.success("""
        **声谱图 = 图像**:
        
        - 横轴: 时间
        - 纵轴: 频率
        - 颜色: 能量强度
        
        **应用**:
        - 语音识别: CNN直接处理声谱图
        - 音乐分类: ResNet on Spectrogram
        - 声音事件检测: YOLO for Audio
        
        **关键洞察**: 声谱图将1D时间序列转为2D图像，解锁了CV的全部工具箱！
        """)
    
    @staticmethod
    def _render_heisenberg():
        """海森堡不确定性原理可视化"""
        st.markdown("### ⚛️ 海森堡不确定性原理：时频权衡")
        
        st.markdown(r"""
        **不确定性原理** (信号处理版):
        """)
        
        st.latex(r"""
        \sigma_t \cdot \sigma_\omega \geq \frac{1}{2}
        """)
        
        st.markdown(r"""
        **物理意义**:
        - $\sigma_t$: 时间展宽（信号在时域的"宽度"）
        - $\sigma_\omega$: 频率展宽（信号在频域的"宽度"）
        - 两者的乘积有下界！
        
        **权衡**:
        - **窄窗口** → 时间精确，频率模糊
        - **宽窗口** → 频率精确，时间模糊
        
        **Gabor变换**: 当窗函数是高斯时，等号成立（理论最优）
        """)
        
        with st.sidebar:
            st.markdown("#### 参数设置")
            window_size_small = st.slider("窄窗口大小", 16, 128, 32, 16)
            window_size_large = st.slider("宽窗口大小", 128, 512, 256, 32)
        
        # 生成Chirp信号
        fs = 1000
        duration = 2.0
        t = np.linspace(0, duration, int(duration * fs))
        x = signal.chirp(t, f0=50, f1=300, t1=duration, method='linear')
        
        # 计算不同窗口大小的STFT
        f_small, t_small, Zxx_small = signal.stft(x, fs=fs, nperseg=window_size_small)
        f_large, t_large, Zxx_large = signal.stft(x, fs=fs, nperseg=window_size_large)
        
        spec_small = 20 * np.log10(np.abs(Zxx_small) + 1e-10)
        spec_large = 20 * np.log10(np.abs(Zxx_large) + 1e-10)
        
        # 可视化对比
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"窄窗口 (n={window_size_small}): 高时间分辨率",
                f"宽窗口 (n={window_size_large}): 高频率分辨率",
                "时间切片对比",
                "频率切片对比"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. 窄窗口声谱图
        fig.add_trace(
            go.Heatmap(
                x=t_small, y=f_small, z=spec_small,
                colorscale='Jet',
                showscale=False
            ),
            row=1, col=1
        )
        
        # 2. 宽窗口声谱图
        fig.add_trace(
            go.Heatmap(
                x=t_large, y=f_large, z=spec_large,
                colorscale='Jet',
                showscale=False
            ),
            row=1, col=2
        )
        
        # 3. 时间切片（固定频率，看时间分辨率）
        freq_idx_small = np.argmin(np.abs(f_small - 150))
        freq_idx_large = np.argmin(np.abs(f_large - 150))
        
        fig.add_trace(
            go.Scatter(
                x=t_small,
                y=np.abs(Zxx_small[freq_idx_small, :]),
                mode='lines',
                name='窄窗口',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=t_large,
                y=np.abs(Zxx_large[freq_idx_large, :]),
                mode='lines',
                name='宽窗口',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # 4. 频率切片（固定时间，看频率分辨率）
        time_idx_small = len(t_small) // 2
        time_idx_large = len(t_large) // 2
        
        fig.add_trace(
            go.Scatter(
                x=f_small,
                y=np.abs(Zxx_small[:, time_idx_small]),
                mode='lines',
                name='窄窗口',
                line=dict(color='blue', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=f_large,
                y=np.abs(Zxx_large[:, time_idx_large]),
                mode='lines',
                name='宽窗口',
                line=dict(color='red', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="频率 (Hz)", range=[0, 400], row=1, col=1)
        fig.update_yaxes(title_text="频率 (Hz)", range=[0, 400], row=1, col=2)
        fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
        fig.update_yaxes(title_text="幅度", row=2, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", range=[0, 400], row=2, col=2)
        fig.update_yaxes(title_text="幅度", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="海森堡不确定性原理演示"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析
        st.markdown("### 📊 量化对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**窄窗口**")
            time_res_small = 1000 / fs * window_size_small
            freq_res_small = fs / window_size_small
            st.metric("时间分辨率", f"{time_res_small:.1f} ms")
            st.metric("频率分辨率", f"{freq_res_small:.2f} Hz")
            product_small = time_res_small * freq_res_small / 1000
            st.metric("Δt·Δf", f"{product_small:.3f}")
        
        with col2:
            st.markdown("**宽窗口**")
            time_res_large = 1000 / fs * window_size_large
            freq_res_large = fs / window_size_large
            st.metric("时间分辨率", f"{time_res_large:.1f} ms")
            st.metric("频率分辨率", f"{freq_res_large:.2f} Hz")
            product_large = time_res_large * freq_res_large / 1000
            st.metric("Δt·Δf", f"{product_large:.3f}")
        
        st.info(r"""
        **观察**:
        
        1. **窄窗口** (左图):
           - 时间轴上的线很清晰（高时间分辨率）
           - 但频率轴上模糊、宽（低频率分辨率）
        
        2. **宽窗口** (右图):
           - 频率轴上的线很细（高频率分辨率）
           - 但时间轴上模糊（低时间分辨率）
        
        3. **Δt·Δf 约束**:
           - 两者的乘积都 ≥ 0.5
           - 这不是算法限制，是数学真理！
        
        **应用选择**:
        - 语音: 需要时间精度 → 窄窗口
        - 音乐: 需要音高精度 → 宽窗口
        """)
        
        st.success(r"""
        **与量子力学的联系**:
        """)
        
        st.markdown("**海森堡的原始公式**:")
        st.latex(r"""
        \Delta x \cdot \Delta p \geq \frac{\hbar}{2}
        """)
        
        st.markdown("**信号处理版本**:")
        st.latex(r"""
        \Delta t \cdot \Delta \omega \geq \frac{1}{2}
        """)
        
        st.markdown(r"""
        **本质相同**: 共轭变量之间的不确定性关系
        - 量子力学: 位置 ↔ 动量
        - 信号处理: 时间 ↔ 频率
        
        这是傅里叶变换的数学性质，与物理测量无关！
        """)
    
    @staticmethod
    def _render_wavelet():
        """小波变换可视化"""
        st.markdown("### 🔬 小波变换：数学显微镜")
        
        st.markdown(r"""
        **小波变换**: 多分辨率分析
        """)
        
        st.latex(r"""
        W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt
        """)
        
        st.markdown(r"""
        **参数**:
        - $a$: 尺度 (scale) ≈ 频率倒数
          - 小 $a$ → 压缩波形 → 捕捉高频
          - 大 $a$ → 拉伸波形 → 捕捉低频
        - $b$: 平移 (shift) → 时间位置
        
        **优势**: 
        - 高频 → 窄窗口（高时间分辨率）
        - 低频 → 宽窗口（高频率分辨率）
        - **自适应时频权衡**
        """)
        
        st.info("""
        **小波 vs STFT**:
        
        | 特性 | STFT | 小波变换 |
        |------|------|----------|
        | 窗口 | 固定大小 | 自适应大小 |
        | 时频分辨率 | 固定权衡 | 频率依赖 |
        | 适用场景 | 平稳信号 | 非平稳信号 |
        | 应用 | 语音识别 | ECG、地震波 |
        
        **直观理解**: 小波变换像变焦显微镜，根据观察对象自动调整放大倍率
        """)
        
        st.success("""
        **小波家族**:
        
        - **Haar小波**: 最简单，阶跃函数
        - **Daubechies小波**: 紧支撑，正交
        - **Morlet小波**: 高斯调制正弦波，常用于时频分析
        - **Mexican Hat**: Ricker小波，用于峰值检测
        
        **应用**:
        - JPEG 2000: 小波压缩
        - ECG分析: 心率变异性
        - 地震预警: 震动信号分析
        """)
    
    @staticmethod
    def _render_ssm():
        """状态空间模型(SSM/Mamba)可视化"""
        st.markdown("### 🐍 状态空间模型：Mamba背后的数学")
        
        st.markdown(r"""
        **SSM核心思想**: 将深度学习模型视为连续时间系统的离散化
        
        **连续系统 (ODE)**:
        """)
        
        st.latex(r"""
        \begin{cases}
        h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t) & \text{(状态方程)} \\
        y(t) = \mathbf{C}h(t) & \text{(输出方程)}
        \end{cases}
        """)
        
        st.markdown(r"""
        **离散化** (Zero-Order Hold):
        """)
        
        st.latex(r"""
        h_k = \bar{\mathbf{A}} h_{k-1} + \bar{\mathbf{B}} x_k
        """)
        
        st.markdown("这看起来就是一个**RNN**！")
        
        st.markdown("### 🔄 卷积-递归对偶性")
        
        st.info(r"""
        **SSM的魔法**: 同一个模型，两种计算方式
        
        **训练时** (并行):
        $$y = x * \mathbf{K}$$
        其中 $\mathbf{K} = (\mathbf{CB}, \mathbf{CAB}, \mathbf{CA}^2\mathbf{B}, ...)$
        
        → 像CNN一样并行训练！
        
        **推理时** (串行):
        $$h_k = \bar{\mathbf{A}} h_{k-1} + \bar{\mathbf{B}} x_k$$
        
        → 像RNN一样 $O(1)$ 推理！
        
        **结论**: SSM打通了CNN（并行训练）和RNN（快速推理）的任督二脉
        """)
        
        st.success("""
        **HiPPO矩阵**: 记忆的数学
        
        如果 $\mathbf{A}$ 随机初始化 → 模型会遗忘历史
        
        **HiPPO理论**: 当 $\mathbf{A}$ 取特定形式时，隐状态 $h(t)$ 存储了历史输入在勒让德多项式基底上的投影
        
        **效果**: 解决长期依赖问题
        
        **Mamba = S4 + 选择性SSM**:
        - S4: 结构化状态空间
        - 选择性: $\mathbf{A}, \mathbf{B}, \mathbf{C}$ 依赖于输入
        - 结果: 超越Transformer的长序列建模能力
        """)
    
    @staticmethod
    def _render_mfcc():
        """MFCC与梅尔刻度可视化"""
        st.markdown("### 🎤 MFCC：模拟人耳的感知")
        
        st.markdown(r"""
        **问题**: 人耳对频率的感知不是线性的
        
        **Mel刻度**:
        """)
        
        st.latex(r"""
        M(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
        """)
        
        st.markdown("""
        **特点**:
        - 低频: 更敏感（音高差异明显）
        - 高频: 不敏感（音高差异不明显）
        """)
        
        # 绘制Mel刻度
        f_hz = np.linspace(0, 8000, 1000)
        f_mel = 2595 * np.log10(1 + f_hz / 700)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=f_hz,
            y=f_mel,
            mode='lines',
            name='Mel刻度',
            line=dict(color='blue', width=3)
        ))
        
        # 添加线性参考
        fig.add_trace(go.Scatter(
            x=f_hz,
            y=f_hz * f_mel[-1] / f_hz[-1],
            mode='lines',
            name='线性刻度',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Mel刻度 vs 线性刻度",
            xaxis_title="频率 (Hz)",
            yaxis_title="Mel / 归一化频率",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 MFCC计算流程")
        
        st.info("""
        **MFCC提取步骤**:
        
        1. **预加重** (Pre-emphasis): 增强高频
        2. **分帧** (Framing): 20-40ms窗口
        3. **加窗** (Windowing): Hamming窗
        4. **FFT**: 时域 → 频域
        5. **Mel滤波器组**: 线性频率 → Mel频率
        6. **对数**: $\log(E)$ 模拟人耳
        7. **DCT**: 去相关，压缩特征
        
        **输出**: 通常取前13个MFCC系数
        
        **应用**: 语音识别、说话人识别、情感识别
        """)
        
        st.success("""
        **为什么要DCT？**
        
        频谱相邻频带高度相关 → DCT去相关
        
        类似于PCA的作用：
        - 将相关特征变换为独立特征
        - 能量集中在前几个系数
        - 后续可以用GMM或神经网络
        
        **深度学习时代**: 
        - 传统: MFCC + GMM-HMM
        - 现代: 原始波形 / 声谱图 + CNN
        - 但MFCC仍然是baseline
        """)
    
    @staticmethod
    def _render_comparison():
        """完整流程对比"""
        st.markdown("### 🔄 完整流程：FFT vs STFT vs 小波")
        
        st.markdown("""
        这里展示同一个信号在不同分析方法下的表现
        """)
        
        # 生成复杂信号：低频背景 + 高频瞬态
        fs = 1000
        t = np.linspace(0, 2, 2000)
        
        # 低频背景 (50Hz)
        low_freq = np.sin(2 * np.pi * 50 * t)
        
        # 高频瞬态 (300Hz, 只在0.5-0.7秒)
        high_freq = np.zeros_like(t)
        mask = (t >= 0.5) & (t <= 0.7)
        high_freq[mask] = np.sin(2 * np.pi * 300 * t[mask])
        
        x = low_freq + high_freq
        
        # FFT
        X = fft(x)
        freqs = fftfreq(len(t), 1/fs)
        pos_mask = freqs >= 0
        
        # STFT
        f_stft, t_stft, Zxx = signal.stft(x, fs=fs, nperseg=128)
        
        # 可视化对比
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "时域信号",
                "FFT: 看到50Hz和300Hz，但不知何时",
                "STFT: 清楚看到瞬态",
                "分析对比"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 时域
        fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='信号',
                                line=dict(color='blue', width=1)),
                     row=1, col=1)
        
        # FFT
        fig.add_trace(go.Scatter(x=freqs[pos_mask], y=np.abs(X[pos_mask]),
                                mode='lines', name='FFT',
                                line=dict(color='red', width=2)),
                     row=1, col=2)
        
        # STFT
        fig.add_trace(go.Heatmap(x=t_stft, y=f_stft, z=20*np.log10(np.abs(Zxx)+1e-10),
                                colorscale='Jet', showscale=False),
                     row=2, col=1)
        
        # 对比表格（使用文本）
        fig.add_annotation(
            text="<b>方法对比</b><br><br>" +
                 "FFT: 全局，无时间<br>" +
                 "STFT: 时频平衡<br>" +
                 "小波: 自适应<br><br>" +
                 "<b>最佳选择</b>:<br>" +
                 "平稳→FFT<br>" +
                 "语音→STFT<br>" +
                 "瞬态→小波",
            xref="x4", yref="y4",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14),
            align="left"
        )
        
        fig.update_xaxes(title_text="时间(s)", row=1, col=1)
        fig.update_xaxes(title_text="频率(Hz)", range=[0, 400], row=1, col=2)
        fig.update_yaxes(title_text="频率(Hz)", range=[0, 400], row=2, col=1)
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="信号分析方法对比")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **总结**:
        
        1. **傅里叶变换**: 
           - 全局频域分析
           - 丢失时间信息
           - 适合平稳信号
        
        2. **STFT**: 
           - 时频局部化
           - 固定窗口
           - 适合语音、音乐
        
        3. **小波变换**: 
           - 多分辨率
           - 自适应窗口
           - 适合非平稳信号
        
        4. **SSM/Mamba**: 
           - 连续系统视角
           - 卷积-递归对偶
           - 适合超长序列
        
        **深度学习应用**: 
        - 音频分类: CNN on Spectrogram
        - 语音识别: Transformer + MFCC
        - 音乐生成: Diffusion on Mel-Spectrogram
        - 长序列: Mamba (SSM)
        """)

        # 添加交互式测验
