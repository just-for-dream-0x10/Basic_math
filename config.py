import os
from pathlib import Path

# 项目配置
PROJECT_ROOT = Path(__file__).parent
VISION_DIR = PROJECT_ROOT
OUTPUT_DIR = VISION_DIR / "output"
MEDIA_DIR = VISION_DIR / "media"
ASSETS_DIR = VISION_DIR / "assets"

# Manim配置
MANIM_CONFIG = {
    "quality": "high_quality",
    "pixel_height": 1080,
    "pixel_width": 1920,
    "frame_rate": 30,
}

# Streamlit配置
STREAMLIT_CONFIG = {
    "title": "数学笔记可视化",
    "layout": "wide",
    "page_icon": "🧮",
}

# 颜色主题
COLORS = {
    "primary": "#3B82F6",
    "secondary": "#8B5CF6", 
    "accent": "#10B981",
    "background": "#1E293B",
    "text": "#F1F5F9",
    "grid": "#475569",
    "highlight": "#F59E0B",
    "error": "#EF4444",
    "success": "#22C55E",
}

# 数学符号映射
MATH_SYMBOLS = {
    "alpha": "α",
    "beta": "β", 
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "sigma": "σ",
    "phi": "φ",
    "omega": "ω",
    "sum": "∑",
    "integral": "∫",
    "partial": "∂",
    "infinity": "∞",
    "sqrt": "√",
    "approx": "≈",
    "neq": "≠",
    "leq": "≤",
    "geq": "≥",
    "pm": "±",
    "times": "×",
    "div": "÷",
}

# 模块配置
MODULES = {
    "matrix": {
        "name": "矩阵论",
        "file": "0.2.Matrix_Foundations.md",
        "description": "矩阵的几何与变换",
        "color": COLORS["primary"],
        "scenes": ["matrix_transform", "svd_decomposition", "eigenvalues"]
    },
    "convolution": {
        "name": "卷积",
        "file": "1.convolution.md", 
        "description": "卷积核与特征提取",
        "color": COLORS["secondary"],
        "scenes": ["convolution_operation", "kernel_types", "feature_extraction"]
    },
    "loss": {
        "name": "损失函数",
        "file": "2.lossfunction.md",
        "description": "最小二乘与交叉熵",
        "color": COLORS["accent"],
        "scenes": ["least_squares", "cross_entropy", "gradient_descent"]
    },
    "optimizer": {
        "name": "优化器",
        "file": "3.grand_optimizer.md",
        "description": "梯度下降与自适应优化",
        "color": COLORS["highlight"],
        "scenes": ["sgd", "momentum", "adam"]
    },
    "svm": {
        "name": "SVM",
        "file": "6.SVM.md",
        "description": "支持向量机与核方法",
        "color": COLORS["error"],
        "scenes": ["margin", "kernel_trick", "dual_problem"]
    },
    "regularization": {
        "name": "L1 & L2 正则化",
        "file": "5.L1&L2.md",
        "description": "正则化与模型复杂度控制",
        "color": "#EC4899",  # Pink色
        "scenes": ["l1_regularization", "l2_regularization", "elastic_net"]
    },
    "lagrange": {
        "name": "拉格朗日乘子法",
        "file": "4.Lagrange_Multiplier.md",
        "description": "约束优化与对偶问题",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["circle_linear", "ellipse_quadratic", "svm_dual", "kkt_conditions"]
    },
    "vcdim": {
        "name": "VC维理论",
        "file": "7.VCdime.md",
        "description": "模型复杂度与泛化能力",
        "color": "#F59E0B",  # Amber色
        "scenes": ["shattering", "vc_calculation", "vc_bound", "model_comparison", "sample_complexity"]
    },
    "calculus": {
        "name": "微积分基础",
        "file": "0.1.Calculus_in_Deep_Learning.md",
        "description": "导数、梯度、链式法则与自动微分",
        "color": "#10B981",  # Green色
        "scenes": ["derivative", "taylor", "chain_rule", "gradient_problems", "autograd"]
    },
    "probability": {
        "name": "概率与信息论",
        "file": "0.3.Probability_Information.md",
        "description": "熵、KL散度、互信息与贝叶斯推断",
        "color": "#3B82F6",  # Blue色
        "scenes": ["distributions", "entropy", "kl_divergence", "cross_entropy", "mutual_info", "bayes"]
    },
    "ml_curves": {
        "name": "机器学习曲线",
        "file": "10.Important_Curves.md",
        "description": "ROC、PR、学习曲线与模型评估",
        "color": "#F59E0B",  # Amber色
        "scenes": ["roc", "pr", "learning_curve", "validation_curve", "confusion_matrix", "calibration"]
    },
    "noise": {
        "name": "噪声理论",
        "file": "9.noise.md",
        "description": "噪声、过拟合与泛化能力",
        "color": "#EF4444",  # Red色
        "scenes": ["noise_nature", "overfitting", "train_test_error", "learning_curves", "triangle_balance", "robustness"]
    },
    "training_dynamics": {
        "name": "训练动力学",
        "file": "18.Training_Dynamics.md",
        "description": "从炼丹到化学：初始化、归一化与超参数理论",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["initialization", "normalization", "noise_temperature", "linear_scaling", "ntk", "diagnosis"]
    },
    "multimodal_geometry": {
        "name": "多模态几何",
        "file": "24.MultimodalGeometry.md",
        "description": "CLIP、InfoNCE与跨模态对齐",
        "color": "#EC4899",  # Pink色
        "scenes": ["hypersphere_alignment", "info_nce", "temperature", "contrastive_dynamics", "grassmannian", "tensor_fusion", "cross_attention"]
    },
    "vcdim_derivation": {
        "name": "VC维详细推导",
        "file": "7.VCdimeDerivationProcess.md",
        "description": "从Hoeffding到泛化界的完整数学推导",
        "color": "#F59E0B",  # Amber色
        "scenes": ["hoeffding", "growth_function", "vc_bound", "radon", "effective_vcdim", "limitations", "derivation_flow"]
    },
    "signal_processing": {
        "name": "信号处理",
        "file": "25.Singal_processing.md",
        "description": "傅里叶、STFT、小波变换与SSM/Mamba",
        "color": "#10B981",  # Emerald色
        "scenes": ["fourier_limits", "stft", "heisenberg", "wavelet", "ssm", "mfcc", "comparison"]
    },
    "cnn_math_foundations": {
        "name": "CNN数学基础",
        "file": "11.CNN_Mathematical_Foundations.md.md",
        "description": "从希尔伯特空间到群论的深层理解",
        "color": "#3B82F6",  # Blue色
        "scenes": ["convolution_theorem", "pooling", "relu_frequency", "group_theory", "architecture_comparison", "complete_framework"]
    },
    "neural_geometry": {
        "name": "神经几何维度",
        "file": "0.4.Neural_Geometry_Dimensions.md",
        "description": "神经网络的几何构造与参数流",
        "color": "#06B6D4",  # Cyan色
        "scenes": ["scaling_laws", "architecture_comparison", "geometry_flow", "lora_decomposition"]
    },
    "hilbert_space": {
        "name": "希尔伯特空间",
        "file": "12.Hilbert_space.md",
        "description": "傅里叶变换与神经网络的数学基础",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["fourier_basics", "convolution_theorem", "cnn_frequency", "graph_fourier"]
    },
    "diffusion_model": {
        "name": "扩散模型",
        "file": "15.DiffusionModel.md",
        "description": "随机微分方程与生成式AI的物理基础",
        "color": "#F97316",  # Orange色
        "scenes": ["diffusion_process", "score_function", "sde_solvers", "langevin_dynamics"]
    },
    "kernel_regression": {
        "name": "核回归与注意力",
        "file": "13.KernelRegression.md",
        "description": "注意力机制的数学本质与核方法理论",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["kernel_basics", "attention_mechanism", "multi_head_attention", "linear_attention"]
    },
    "neuroevolution": {
        "name": "神经进化",
        "file": "14.Neuroevolution.md",
        "description": "进化策略与零阶优化算法",
        "color": "#F97316",  # Orange色
        "scenes": ["es_vs_gradient", "openai_es", "pbt_training", "cma_es"]
    },
    "probabilistic_programming": {
        "name": "概率编程",
        "file": "17.ProbabilisticProgramming.md",
        "description": "贝叶斯深度学习与不确定性量化",
        "color": "#EC4899",  # Pink色
        "scenes": ["bayesian_basics", "vi_vs_mcmc", "reparameterization", "uncertainty_analysis"]
    },
    "mdp": {
        "name": "马尔可夫决策过程",
        "file": "16.MDP.md",
        "description": "强化学习的数学基础与贝尔曼方程",
        "color": "#F59E0B",  # Amber色
        "scenes": ["mdp_basics", "value_iteration", "q_learning", "policy_gradient"]
    },
    "information_geometry": {
        "name": "信息几何",
        "file": "19.Information_Geometry.md",
        "description": "黎曼流形上的优化与自然梯度",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["parameter_vs_probability", "fisher_information", "natural_gradient", "adam_geometry"]
    },
    "gcn": {
        "name": "图神经网络",
        "file": "20.GCN.md",
        "description": "图神经网络与谱图理论：非欧几里得空间的谐波分析",
        "color": "#10B981",  # Green色
        "scenes": ["graph_basics", "laplacian_matrix", "spectral_theory", "gcn_propagation"]
    },
    "causation": {
        "name": "因果推断",
        "file": "21.Causation.md",
        "description": "因果推断：结构方程与Do-Calculus",
        "color": "#F59E0B",  # Amber色
        "scenes": ["dag_basics", "simpson_paradox", "do_calculus", "counterfactual"]
    },
    "optimal_transport": {
        "name": "最优传输理论",
        "file": "22.OptimalTransport.md",
        "description": "最优传输：从搬土问题到生成模型",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["wasserstein_distance", "transport_problem", "sinkhorn_algorithm", "generative_models"]
    },
    "game_theory": {
        "name": "博弈论",
        "file": "23.GameTheory.md",
        "description": "博弈论：从静态优化到动态均衡",
        "color": "#EF4444",  # Red色
        "scenes": ["nash_equilibrium", "minmax_dynamics", "jacobian_analysis", "stackelberg", "lola"]
    },
    "scaling_laws": {
        "name": "缩放定律",
        "file": "AppxB_ScalingLaws.md",
        "description": "Scaling Laws：预知未来的数学与Chinchilla最优前沿",
        "color": "#06B6D4",  # Cyan色
        "scenes": ["power_law", "chinchilla_optimal", "compute_budget", "train_vs_inference", "llama3_strategy"]
    },
    "classification_optimization": {
        "name": "分类模型优化逻辑",
        "file": "8.TheEssentialOptimizationLogicOfClassificationModels.md",
        "description": "从三个视角理解分类：最小二乘、最大似然、SVM",
        "color": "#F59E0B",  # Amber色
        "scenes": ["unified_comparison", "least_squares", "mle", "svm", "loss_comparison", "boundary_evolution", "practical"]
    },
    "dimensions_parameters": {
        "name": "维度与参数估算",
        "file": "AppxD_Dimensions_Parameters.md",
        "description": "工程速查：计算参数量、显存占用、优化策略",
        "color": "#8B5CF6",  # Purple色
        "scenes": ["transformer_calc", "cnn_calc", "memory_calc", "architecture_comparison", "memory_anatomy", "precision_quant"]
    }
}

# 创建必要的目录
for dir_path in [OUTPUT_DIR, MEDIA_DIR]:
    dir_path.mkdir(exist_ok=True)

# 创建assets子目录
for module in MODULES.keys():
    (ASSETS_DIR / module).mkdir(exist_ok=True)