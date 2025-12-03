# 数学可视化交互式学习系统

基于 Streamlit 和 Manim 的机器学习数学可视化平台，通过交互式参数调整和动画演示帮助理解深度学习核心数学概念。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Manim](https://img.shields.io/badge/Manim-0.18+-green.svg)](https://docs.manim.community/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**32个交互模块 | 150+个动画场景 | 8条学习路径**

[©️ Just For Dream Lab, 团队开源笔记库](https://github.com/just-for-dream-0x10/beginML)


---


## 核心特性

### 交互式可视化
- **32个交互模块** - 涵盖线性代数、优化理论、深度学习架构、前沿算法
- **150+个动画场景** - 基于 Manim 的数学动画，直观展示抽象概念
- **参数实时调整** - 修改超参数，观察算法行为和收敛过程
- **数学公式渲染** - LaTeX 公式实时渲染，支持复杂数学表达

### 系统化学习
- **8条学习路径** - 针对不同背景和目标的结构化学习方案
- **133个核心概念** - 20层知识依赖结构，从基础到前沿
- **知识图谱可视化** - 3种视图模式展示概念关系和依赖
- **进度跟踪系统** - 学习成就、进度统计、智能推荐

### 实用工具
- **参数计算器** - Transformer参数量、FLOPs、显存估算
- **算法对比** - SGD vs Adam、L1 vs L2等优化器和正则化对比
- **缩放定律分析** - 模型性能预测、计算最优配置
- **性能优化** - 智能缓存系统，快速响应

## 📁 项目结构

```
vision/
├── app.py                       # Streamlit主应用入口
├── config.py                    # 32个模块配置和主题设置
├── learning_paths.py            # 8条学习路径定义和依赖关系
├── learning_path_ui.py          # 学习路径UI组件和进度跟踪
├── run_manim.py                 # Manim视频生成脚本
├── utils.py                     # 数学工具函数和辅助方法
├── organize_videos.py           # 视频文件组织工具
├── post_process_video.py        # 视频后处理脚本
├── requirements.txt             # Python依赖包列表
├── manim.cfg                    # Manim配置文件
├── interactive/                 # 32个交互式模块实现
│   ├── __init__.py             # 模块注册和导入中心
│   ├── base.py                 # 基础交互类和通用组件
│   ├── calculus.py             # 微积分基础：导数、链式法则
│   ├── matrix.py               # 矩阵论：线性变换、特征值
│   ├── probability.py          # 概率与信息论：熵、KL散度
│   ├── neural_geometry.py      # 神经几何：参数效率、LoRA
│   ├── convolution.py          # 卷积神经网络：卷积操作
│   ├── loss.py                 # 损失函数：MSE、交叉熵
│   ├── optimizer.py            # 优化器：SGD、Momentum、Adam
│   ├── regularization.py       # 正则化：L1/L2、权重衰减
│   ├── lagrange.py             # 拉格朗日乘子法：约束优化
│   ├── svm.py                  # 支持向量机：最大间隔、核方法
│   ├── vcdim.py                # VC维理论：泛化能力分析
│   ├── vcdim_derivation.py     # VC维推导：Hoeffding不等式
│   ├── classification_optimization.py  # 分类优化逻辑
│   ├── noise.py                # 噪声理论：鲁棒性、平滑
│   ├── ml_curves.py            # 机器学习曲线：ROC、学习曲线
│   ├── cnn_math_foundations.py # CNN数学基础：群论、频域分析
│   ├── hilbert_space.py        # 希尔伯特空间：内积空间、核技巧
│   ├── kernel_regression.py    # 核回归与注意力机制
│   ├── neuroevolution.py       # 神经进化：进化策略、遗传算法
│   ├── diffusion_model.py      # 扩散模型：DDPM、得分匹配
│   ├── mdp.py                  # 马尔可夫决策过程：强化学习基础
│   ├── probabilistic_programming.py  # 概率编程：贝叶斯推断
│   ├── training_dynamics.py    # 训练动力学：初始化、归一化
│   ├── information_geometry.py # 信息几何：费雪信息、自然梯度
│   ├── gcn.py                  # 图神经网络：谱图卷积
│   ├── causation.py            # 因果推断：Do-Calculus、反事实
│   ├── optimal_transport.py    # 最优传输：Wasserstein距离
│   ├── game_theory.py          # 博弈论：纳什均衡、Stackelberg
│   ├── multimodal_geometry.py  # 多模态几何：CLIP、对比学习
│   ├── signal_processing.py    # 信号处理：傅里叶变换、小波
│   ├── scaling_laws.py         # 缩放定律：幂律拟合、Chinchilla
│   └── dimensions_parameters.py # 维度与参数：计算复杂度分析
├── scenes/                      # Manim动画场景定义
│   ├── convolution/            # 卷积相关场景
│   ├── loss/                   # 损失函数场景
│   ├── matrix/                 # 矩阵变换场景
│   ├── optimizer/              # 优化器场景
│   └── svm/                    # SVM场景
├── common/                      # 通用组件和工具
│   ├── base_scene.py           # Manim场景基类
│   ├── quiz_system.py          # 测验系统
│   ├── smart_cache.py          # 智能缓存系统
│   └── performance.py          # 性能监控
├── assets/                      # 静态资源文件
│   ├── calculus/               # 微积分相关资源
│   ├── convolution/            # 卷积相关资源
│   └── ...                     # 其他模块资源
├── media/                       # Manim生成的媒体文件
├── output/                      # 输出文件目录
└── __pycache__/                 # Python缓存目录
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 推荐使用虚拟环境
- FFmpeg（用于Manim视频生成）

### 安装与运行

```bash

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动Streamlit应用
streamlit run app.py
```

访问 `http://localhost:8501` 开始学习。

### Manim视频生成（可选）

```bash
# 生成所有模块的视频
python run_manim.py --all --quality high

# 生成特定模块视频
python run_manim.py --module matrix --scene matrix_transform

# 列出所有可用场景
python run_manim.py --list
```

### 核心依赖
- `streamlit` - 交互式Web应用框架
- `manim` - 数学动画引擎
- `numpy` - 数值计算基础
- `plotly` - 交互式图表库
- `scipy` - 科学计算工具
- `matplotlib` - 静态图表绘制
- `pandas` - 数据处理
- `scikit-learn` - 机器学习算法
- `Pillow` - 图像处理

### 使用流程

1. **选择学习模式**
   - 📖 **模块浏览**：按难度查看所有32个模块
   - 🎓 **学习路径**：遵循8条结构化学习方案
   - 🕸️ **概念地图**：查看133个概念的依赖关系

2. **学习单个模块**
   - 阅读数学原理和公式推导
   - 观看Manim动画演示
   - 调整参数观察算法行为
   - 完成测验检验理解程度

3. **跟踪进度**
   - 标记已完成模块
   - 查看学习统计和成就
   - 获取下一步智能推荐

4. **生成教学视频**（可选）
   - 使用Manim生成高质量动画
   - 导出为MP4用于教学或分享

## 📖 完整模块列表（32个）

按学习难度从低到高排序（⭐难度1-5）：

### 第一阶段：数学基础 (⭐⭐)
- **矩阵论** - 线性变换、特征值分解、SVD
- **微积分** - 梯度、链式法则、反向传播
- **概率与信息论** - 熵、KL散度、互信息

### 第二阶段：机器学习入门 (⭐⭐)
- **损失函数** - MSE、交叉熵、损失曲面几何
- **优化器** - SGD、Momentum、Adam算法对比
- **机器学习曲线** - 学习曲线、验证曲线、ROC/AUC
- **噪声理论** - 噪声鲁棒性、标签平滑、数据增强

### 第三阶段：模型与正则化 (⭐⭐⭐)
- **L1/L2正则化** - 范数约束几何、稀疏性、权重衰减
- **卷积神经网络** - 卷积算子、感受野、特征层次
- **分类优化** - 最小二乘分类、逻辑回归、SVM对比

### 第四阶段：高级理论 (⭐⭐⭐⭐)
- **拉格朗日乘子法** - 约束优化、KKT条件、Slater条件
- **神经几何维度** - 参数效率、LoRA、低秩分解
- **训练动力学** - 批归一化、初始化策略、学习率调度
- **CNN数学基础** - 平移等变性、群论、频域卷积
- **希尔伯特空间** - 内积空间、RKHS、核技巧

### 第五阶段：深度学习进阶 (⭐⭐⭐⭐)
- **支持向量机** - 最大间隔原理、核方法、SMO算法
- **核回归与注意力** - Nadaraya-Watson、自注意力、Q-K-V分解
- **扩散模型** - DDPM、得分匹配、SDE/ODE求解
- **信息几何** - 费雪信息矩阵、自然梯度、KL几何
- **信号处理** - 快速傅里叶变换、STFT、小波分析

### 第六阶段：专业领域 (⭐⭐⭐⭐⭐)
- **VC维理论** - PAC学习框架、泛化界、样本复杂度
- **VC维推导** - Hoeffding不等式、增长函数、Sauer引理
- **神经进化** - 遗传算法、NEAT算法、进化策略
- **概率编程** - 贝叶斯网络、变分推断、MCMC采样
- **马尔可夫决策过程** - 动态规划、价值迭代、策略迭代

### 第七阶段：前沿研究 (⭐⭐⭐⭐⭐)
- **图神经网络** - GCN、图注意力网络、消息传递
- **因果推断** - 结构因果模型、Do-Calculus、工具变量
- **最优传输** - Wasserstein距离、Sinkhorn算法、Gromov-Wasserstein
- **博弈论** - 纳什均衡、双层优化、演化稳定策略
- **多模态几何** - CLIP对比学习、流形对齐、模态间隙
- **缩放定律** - 幂律拟合、计算最优、Chinchilla定律
- **维度与参数** - 参数量计算、FLOPs估算、内存占用分析

## 🎯 核心功能

### 📊 交互式可视化
- **实时参数调整**：滑块、输入框实时修改算法参数
- **多维度可视化**：2D/3D图表、Manim动画、热力图
- **算法对比**：并排对比不同算法的性能和行为
- **案例演示**：经典数据集（MNIST、Iris等）实战

### 🎬 Manim动画系统
- **150+个动画场景**：基于Manim的数学动画
- **高质量渲染**：支持4K分辨率和自定义帧率
- **批量生成**：一键生成所有模块的教学视频
- **场景管理**：模块化的场景定义和配置

### 🧪 知识图谱与测验
- **133个核心概念**：涵盖数学基础到前沿研究
- **20层依赖结构**：清晰展示学习路径和前置知识
- **3种可视化模式**：
  - 层次视图：按学习顺序逐层展开
  - 网络视图：交互式概念关系图
  - 列表视图：搜索功能，快速查找概念
- **进度跟踪**：实时显示已掌握/可学习/待解锁的概念

### 📐 实用计算器
- **Transformer参数计算**：根据层数、维度计算参数量
- **CNN FLOPs估算**：卷积层计算复杂度分析
- **显存估算**：训练/推理内存占用预测
- **缩放定律拟合**：Loss vs 参数量的幂律关系

### 🚀 性能优化
- **智能缓存**：优化加载速度，流畅体验
- **批量渲染**：提高大规模数据可视化性能
- **错误处理**：完善的边界检查和异常捕获
- **响应式设计**：适配各种屏幕尺寸

## 🎓 学习路径（8条）

针对不同背景和目标提供结构化学习方案：

### 1. 🚀 快速入门 (1-2周)
**适合**：完全新手，快速了解机器学习基础  
**模块**：矩阵论 → 概率论 → 损失函数 → 优化器 → 机器学习曲线  
**目标**：理解基本概念，能看懂简单论文

### 2. 🧠 深度学习 (4-5周)
**适合**：已有基础，系统学习深度学习  
**模块**：卷积神经网络 → 注意力机制 → Transformer → 训练动力学 → 扩散模型  
**目标**：掌握现代深度学习架构和训练技巧

### 3. 📈 优化理论 (3-4周)
**适合**：想深入理解优化算法  
**模块**：梯度下降 → 动量方法 → Adam → 自然梯度 → 拉格朗日乘子法  
**目标**：理解优化算法的数学原理和收敛性分析

### 4. 🎲 概率与生成模型 (4-5周)
**适合**：对生成模型感兴趣  
**模块**：概率论 → 变分推断 → VAE → GAN → 扩散模型 → 概率编程  
**目标**：掌握概率建模和生成模型设计

### 5. 🔬 理论基础 (5-6周)
**适合**：追求数学严格性，研究者路线  
**模块**：泛化理论 → VC维 → PAC学习 → 信息几何 → 最优传输  
**目标**：建立扎实的理论基础，理解泛化界和样本复杂度

### 6. 🤖 强化学习 (3-4周)
**适合**：对决策和控制感兴趣  
**模块**：马尔可夫决策过程 → 价值迭代 → 策略梯度 → 博弈论  
**目标**：掌握强化学习基础和多智能体系统

### 7. 🌐 图与因果 (3-4周)
**适合**：关注关系数据和因果推断  
**模块**：图神经网络 → 消息传递 → 因果推断 → Do-Calculus  
**目标**：理解图结构数据处理和因果关系识别

### 8. 🔮 前沿应用 (4-5周)
**适合**：跟进最新研究方向  
**模块**：多模态几何 → CLIP → 扩散模型 → 缩放定律 → 神经进化  
**目标**：了解当前研究热点和工业应用

## 📈 项目特点

### 技术深度
- **数学严谨性**：公式推导完整，不停留在概念解释
- **算法细节**：包含复杂度分析、收敛性证明
- **前沿覆盖**：Transformer、CLIP、扩散模型、缩放定律等最新技术
- **跨学科整合**：融合数学、物理、信息论等多学科视角

### 教学设计
- **渐进式学习**：从基础到高级，20层依赖结构
- **可视化驱动**：Manim动画帮助建立几何直观
- **概念关联**：133个概念形成完整知识网络
- **多种路径**：8条学习路径适应不同背景和目标

### 工程质量
- **模块化架构**：32个独立模块，易于维护和扩展
- **智能缓存**：优化加载速度，流畅体验
- **错误处理**：完善的边界检查和异常捕获
- **配置驱动**：统一的配置管理，便于定制

### 动画系统
- **专业级动画**：基于Manim的高质量数学动画
- **场景管理**：模块化的场景定义和参数配置
- **批量生成**：支持一键生成所有模块视频
- **质量可调**：从预览到4K的多种质量选项

## 🆕 最近更新

### v2.0 (2025-12)
- ✅ **Manim动画系统**：集成Manim引擎，支持150+个高质量动画场景
- ✅ **视频生成工具**：新增`run_manim.py`，支持批量生成教学视频
- ✅ **场景模块化**：重构scenes目录，按功能组织动画场景
- ✅ **视频后处理**：新增`post_process_video.py`，优化视频质量
- ✅ **资源管理**：完善assets目录结构，支持模块化资源管理
- ✅ **配置优化**：更新`manim.cfg`，提供专业级渲染配置
- ✅ **文档完善**：详细的场景开发指南和API文档

### v1.5 (2025-10)
- 添加缩放定律、多模态几何等前沿模块
- 优化批量渲染性能
- 实现智能缓存系统
- 完善错误处理机制

### v1.0 (2025-09)
- 32个交互模块上线
- 8条学习路径完成
- 基础可视化系统实现

## 🎨 开发与扩展

### 创建新的交互模块

1. 在 `interactive/` 目录下创建新文件
2. 继承基类并实现 `render()` 方法：

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from common.quiz_system import QuizSystem, QuizTemplates

class InteractiveMyModule:
    @staticmethod
    def render():
        st.subheader("🎯 我的模块")
        st.markdown("模块描述和数学公式")
        
        # 参数控制
        param = st.slider("参数", 0.0, 1.0, 0.5)
        
        # 可视化
        fig = go.Figure()
        # ... 绘图代码
        st.plotly_chart(fig)
        
        # 测验（可选）
        quiz_system = QuizSystem("my_module")
        quizzes = QuizTemplates.get_my_module_quizzes()
        quiz_system.render_quiz(quizzes)
```

3. 在 `config.py` 中添加模块配置

### 创建Manim动画场景

1. 在 `scenes/` 目录下创建模块文件夹
2. 继承 `BaseScene` 类：

```python
from manim import *
from common.base_scene import BaseScene

class MyScene(BaseScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "my_module"
    
    def construct(self):
        title = self.add_title("场景标题", "副标题")
        
        # 动画内容
        formula = MathTex(r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}")
        self.play(Write(formula))
        
        # 更多动画...
        self.wait()
```

3. 在 `run_manim.py` 中注册场景映射

4. 使用命令生成视频：
```bash
python run_manim.py --module my_module --scene my_scene --quality high
```

## 🛠️ 技术栈

### 核心技术
- **Streamlit** - 交互式Web应用框架
- **Manim** - 数学动画引擎（3Blue1Brown同款）
- **Plotly** - 高质量交互式图表
- **NumPy** - 数值计算基础
- **Matplotlib** - 静态图表绘制
- **SciPy** - 科学计算
- **Pandas** - 数据处理
- **Scikit-learn** - 机器学习算法
- **Pillow** - 图像处理

### 开发工具

```bash
# 启动开发模式（自动重载）
streamlit run app.py --server.runOnSave true

# 指定端口
streamlit run app.py --server.port 8501

# 清除缓存
streamlit cache clear

# 生成Manim动画
python run_manim.py --all --quality high

# 开发单个场景
manim scenes/my_scene.py MyScene -pqh
```

### 视频质量选项
- `-ql` - 低质量（快速预览）
- `-qm` - 中等质量（默认）
- `-qh` - 高质量（1080p）
- `-qp` - 生产质量（4K）

## 💡 使用场景

### 📚 自学深度学习
- 按照学习路径系统学习
- 通过交互式可视化理解抽象概念
- 实时调整参数观察效果

### 👨‍🏫 教学辅助
- 课堂演示数学原理
- 让学生动手调整参数
- 直观展示算法效果

### 🔬 算法研究
- 快速原型验证
- 参数影响分析
- 算法对比实验

### 🛠️ 工程实践
- 模型参数估算
- 显存需求计算
- 架构选择参考

## 🎯 项目亮点

### 1. 系统化的知识体系
- 32个模块覆盖深度学习全栈知识
- 从基础数学到前沿研究的完整路径
- 8条学习路径适合不同背景的学习者

### 2. 深度的交互体验
- 150+个交互场景，不是被动观看而是主动探索
- 实时参数调整，立即看到效果
- 多维度可视化（2D/3D/动画/热图）

### 3. 直观的数学理解
- 将抽象公式转化为几何图形
- 通过动态演示理解算法本质
- 对比实验展示不同方法的差异

### 4. 实用的工程工具
- Transformer参数量计算器
- 显存占用估算器
- 架构效率对比工具
- 缩放定律预测器

### 5. 优秀的代码架构
- 模块化设计，易于扩展
- 统一的基类和接口
- 配置驱动的场景管理
- 完善的类型注解

## 📊 数据统计

- 📦 **32个交互模块** - 100%实现率
- 🎬 **150+个动画场景** - 基于Manim的专业数学动画
- 🧠 **133个核心概念** - 20层知识依赖结构
- 🛤️ **8条学习路径** - 覆盖1-8周的学习计划
- 💻 **8000+行代码** - 精心打磨的实现
- 🎨 **5级难度分级** - 从入门到精通
- 📹 **4K视频支持** - 专业级教学视频生成
- 🎯 **智能推荐** - 基于学习进度的个性化推荐


## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **报告Bug** - 提交Issue描述问题
2. **建议功能** - 提出新的模块或场景想法
3. **改进文档** - 完善README或代码注释
4. **添加模块** - 实现新的交互模块
5. **优化代码** - 提升性能或可读性

### 贡献流程

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add: 新增XX功能'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 开发规范

- 遵循现有的代码风格
- 添加必要的注释和文档
- 确保交互场景运行正常
- 更新相关配置文件

## ❓ 常见问题

### Q: 需要什么基础？
A: 建议有Python基础和基本的数学知识。完全新手可以从"快速入门"路径开始。

### Q: 可以离线使用吗？
A: 可以。所有计算都在本地进行，不需要网络连接。

### Q: 支持哪些浏览器？
A: 推荐使用Chrome、Firefox或Edge的最新版本。

### Q: 如何保存自己的学习进度？
A: 目前支持浏览器的本地缓存，未来会添加账号系统。

### Q: 可以用于商业用途吗？
A: 本项目采用MIT许可证，可以自由使用。

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目和工具：

- [Streamlit](https://streamlit.io/) - 优秀的Web应用框架
- [Manim](https://docs.manim.community/) - 3Blue1Brown的数学动画引擎
- [Plotly](https://plotly.com/) - 强大的交互式图表库
- [NumPy](https://numpy.org/) - Python科学计算基础
- [Matplotlib](https://matplotlib.org/) - 经典的可视化库
- [SciPy](https://scipy.org/) - 科学计算工具集
- [Scikit-learn](https://scikit-learn.org/) - 机器学习算法库

特别感谢：
- **Grant Sanderson (3Blue1Brown)** - 创造Manim，让数学变得直观
- **深度学习研究社区** - 提供丰富的理论基础和算法实现
- **开源贡献者们** - 让高质量的教育资源变得普惠

---

<div align="center">

### 🌟 如果这个项目对你有帮助，请给个 Star！

**让数学变得有趣，让学习变得高效！**

[开始使用](#-快速开始) | [查看文档](#-完整模块列表32个) | [学习路径](#-学习路径8条) | [参与贡献](#-贡献指南)

</div>