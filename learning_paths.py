"""
学习路径配置
定义不同背景用户的推荐学习路径
"""

# 学习路径定义
LEARNING_PATHS = {
    "beginner": {
        "name": "🌱 初学者路径",
        "description": "适合机器学习新手，从数学基础开始",
        "difficulty": "入门",
        "duration": "2-3周",
        "modules": [
            {
                "module": "matrix",
                "title": "矩阵论",
                "why": "线性代数是机器学习的语言",
                "prerequisites": [],
                "key_concepts": ["线性变换", "特征向量", "矩阵乘法"],
                "time": "3天",
                "scenes": ["matrix_transform"]
            },
            {
                "module": "calculus",
                "title": "微积分基础",
                "why": "理解梯度下降的数学原理",
                "prerequisites": ["matrix"],
                "key_concepts": ["导数", "梯度", "链式法则"],
                "time": "4天",
                "scenes": ["derivative", "chain_rule"]
            },
            {
                "module": "loss",
                "title": "损失函数",
                "why": "理解模型如何学习",
                "prerequisites": ["calculus"],
                "key_concepts": ["梯度下降", "收敛", "局部最小值"],
                "time": "3天",
                "scenes": ["gradient_descent"]
            },
            {
                "module": "optimizer",
                "title": "优化器",
                "why": "掌握不同的训练方法",
                "prerequisites": ["loss"],
                "key_concepts": ["SGD", "Momentum", "Adam"],
                "time": "3天",
                "scenes": ["optimizer_comparison"]
            },
            {
                "module": "ml_curves",
                "title": "机器学习曲线",
                "why": "学会评估模型性能",
                "prerequisites": ["optimizer"],
                "key_concepts": ["ROC", "混淆矩阵", "学习曲线"],
                "time": "3天",
                "scenes": ["roc", "confusion_matrix", "learning_curve"]
            }
        ]
    },
    
    "deep_learning": {
        "name": "🧠 深度学习路径",
        "description": "专注于神经网络和深度学习",
        "difficulty": "进阶",
        "duration": "4-5周",
        "modules": [
            {
                "module": "calculus",
                "title": "微积分：反向传播",
                "why": "理解神经网络训练的核心",
                "prerequisites": [],
                "key_concepts": ["链式法则", "自动微分", "梯度消失"],
                "time": "3天",
                "scenes": ["chain_rule", "gradient_problems", "autograd"]
            },
            {
                "module": "loss",
                "title": "损失函数与优化",
                "why": "掌握训练技巧",
                "prerequisites": ["calculus"],
                "key_concepts": ["交叉熵", "梯度下降", "损失地形"],
                "time": "3天",
                "scenes": ["gradient_descent"]
            },
            {
                "module": "optimizer",
                "title": "优化器详解",
                "why": "选择合适的优化算法",
                "prerequisites": ["loss"],
                "key_concepts": ["Adam", "学习率", "动量"],
                "time": "2天",
                "scenes": ["optimizer_comparison"]
            },
            {
                "module": "regularization",
                "title": "正则化技术",
                "why": "防止过拟合",
                "prerequisites": ["optimizer"],
                "key_concepts": ["L1/L2", "权重衰减", "Dropout"],
                "time": "3天",
                "scenes": ["l1_l2_comparison"]
            },
            {
                "module": "training_dynamics",
                "title": "训练动力学",
                "why": "理解训练的物理本质",
                "prerequisites": ["regularization"],
                "key_concepts": ["初始化", "归一化", "噪声温度"],
                "time": "4天",
                "scenes": ["initialization", "normalization"]
            },
            {
                "module": "convolution",
                "title": "卷积神经网络",
                "why": "理解CNN的数学原理",
                "prerequisites": ["training_dynamics"],
                "key_concepts": ["卷积操作", "特征提取", "权重共享"],
                "time": "3天",
                "scenes": ["convolution_demo"]
            },
            {
                "module": "cnn_math_foundations",
                "title": "CNN数学基础",
                "why": "深入理解CNN的理论",
                "prerequisites": ["convolution"],
                "key_concepts": ["群论", "频域分析", "平移不变性"],
                "time": "4天",
                "scenes": ["convolution_theorem", "group_theory"]
            },
            {
                "module": "kernel_regression",
                "title": "注意力机制",
                "why": "理解Transformer的核心",
                "prerequisites": ["cnn_math_foundations"],
                "key_concepts": ["核回归", "注意力", "Query-Key-Value"],
                "time": "4天",
                "scenes": ["attention_mechanism"]
            },
            {
                "module": "neural_geometry",
                "title": "神经网络几何",
                "why": "理解网络架构设计",
                "prerequisites": ["kernel_regression"],
                "key_concepts": ["维度缩放", "LoRA", "参数流"],
                "time": "3天",
                "scenes": ["dimension_analysis"]
            },
            {
                "module": "scaling_laws",
                "title": "缩放定律",
                "why": "预测模型性能",
                "prerequisites": ["neural_geometry"],
                "key_concepts": ["幂律", "计算最优", "Chinchilla"],
                "time": "3天",
                "scenes": ["power_law", "chinchilla_optimal"]
            },
            {
                "module": "ml_curves",
                "title": "模型评估",
                "why": "诊断和优化模型",
                "prerequisites": ["scaling_laws"],
                "key_concepts": ["学习曲线", "验证曲线", "过拟合诊断"],
                "time": "2天",
                "scenes": ["learning_curve", "validation_curve"]
            }
        ]
    },
    
    "theory": {
        "name": "📚 理论深度路径",
        "description": "适合研究者，深入数学理论",
        "difficulty": "高级",
        "duration": "6-8周",
        "modules": [
            {
                "module": "matrix",
                "title": "线性代数理论",
                "why": "建立坚实的数学基础",
                "prerequisites": [],
                "key_concepts": ["特征值", "SVD", "投影"],
                "time": "4天",
                "scenes": ["matrix_transform"]
            },
            {
                "module": "probability",
                "title": "概率与信息论",
                "why": "理解学习的理论基础",
                "prerequisites": ["matrix"],
                "key_concepts": ["熵", "KL散度", "互信息"],
                "time": "5天",
                "scenes": ["entropy", "kl_divergence", "mutual_info"]
            },
            {
                "module": "hilbert_space",
                "title": "希尔伯特空间",
                "why": "理解函数空间与内积",
                "prerequisites": ["matrix"],
                "key_concepts": ["内积", "正交性", "傅里叶变换"],
                "time": "5天",
                "scenes": ["fourier_basics", "convolution_theorem"]
            },
            {
                "module": "lagrange",
                "title": "拉格朗日乘子法",
                "why": "掌握约束优化理论",
                "prerequisites": ["matrix"],
                "key_concepts": ["对偶问题", "KKT条件", "凸优化"],
                "time": "4天",
                "scenes": ["constraint_optimization"]
            },
            {
                "module": "information_geometry",
                "title": "信息几何",
                "why": "理解参数空间的黎曼结构",
                "prerequisites": ["probability"],
                "key_concepts": ["费雪信息", "自然梯度", "KL球"],
                "time": "5天",
                "scenes": ["natural_optimization"]
            },
            {
                "module": "vcdim",
                "title": "VC维理论",
                "why": "理解泛化能力的本质",
                "prerequisites": ["probability"],
                "key_concepts": ["VC维", "PAC学习", "泛化界"],
                "time": "4天",
                "scenes": ["vc_theory"]
            },
            {
                "module": "vcdim_derivation",
                "title": "VC维完整推导",
                "why": "深入理解泛化理论",
                "prerequisites": ["vcdim"],
                "key_concepts": ["Hoeffding", "增长函数", "Radon定理"],
                "time": "5天",
                "scenes": ["hoeffding", "growth_function"]
            },
            {
                "module": "svm",
                "title": "支持向量机",
                "why": "理论与实践的完美结合",
                "prerequisites": ["lagrange", "vcdim"],
                "key_concepts": ["最大间隔", "核方法", "对偶问题"],
                "time": "4天",
                "scenes": ["svm_classifier"]
            },
            {
                "module": "regularization",
                "title": "正则化理论",
                "why": "理解正则化的数学本质",
                "prerequisites": ["lagrange"],
                "key_concepts": ["约束优化", "稀疏性", "贝叶斯视角"],
                "time": "3天",
                "scenes": ["l1_l2_comparison"]
            },
            {
                "module": "optimal_transport",
                "title": "最优传输理论",
                "why": "理解分布间的几何距离",
                "prerequisites": ["probability", "lagrange"],
                "key_concepts": ["Wasserstein距离", "对偶问题", "Sinkhorn"],
                "time": "5天",
                "scenes": ["transport_theory"]
            },
            {
                "module": "causation",
                "title": "因果推断",
                "why": "超越相关性看到因果",
                "prerequisites": ["probability"],
                "key_concepts": ["Do-Calculus", "DAG", "反事实"],
                "time": "5天",
                "scenes": ["causal_inference"]
            },
            {
                "module": "game_theory",
                "title": "博弈论",
                "why": "理解多智能体优化",
                "prerequisites": ["lagrange"],
                "key_concepts": ["纳什均衡", "雅可比", "Stackelberg"],
                "time": "5天",
                "scenes": ["strategic_reasoning"]
            }
        ]
    },
    
    "practitioner": {
        "name": "⚙️ 工程实践路径",
        "description": "适合工程师，快速掌握实用技能",
        "difficulty": "实战",
        "duration": "1-2周",
        "modules": [
            {
                "module": "loss",
                "title": "损失函数选择",
                "why": "选择正确的优化目标",
                "prerequisites": [],
                "key_concepts": ["交叉熵", "MSE", "损失函数对比"],
                "time": "2天",
                "scenes": ["gradient_descent"]
            },
            {
                "module": "optimizer",
                "title": "优化器调参",
                "why": "加速模型训练",
                "prerequisites": ["loss"],
                "key_concepts": ["学习率", "Adam", "学习率衰减"],
                "time": "2天",
                "scenes": ["optimizer_comparison"]
            },
            {
                "module": "regularization",
                "title": "防止过拟合",
                "why": "提升泛化能力",
                "prerequisites": ["optimizer"],
                "key_concepts": ["L1/L2", "Dropout", "Early Stopping"],
                "time": "2天",
                "scenes": ["l1_l2_comparison"]
            },
            {
                "module": "ml_curves",
                "title": "模型诊断",
                "why": "识别和解决问题",
                "prerequisites": ["regularization"],
                "key_concepts": ["学习曲线", "混淆矩阵", "ROC/PR"],
                "time": "3天",
                "scenes": ["learning_curve", "confusion_matrix", "roc"]
            },
            {
                "module": "svm",
                "title": "SVM调参",
                "why": "传统ML的强大工具",
                "prerequisites": ["ml_curves"],
                "key_concepts": ["C参数", "核函数", "支持向量"],
                "time": "2天",
                "scenes": ["svm_classifier"]
            }
        ]
    },
    
    "custom": {
        "name": "🎯 自定义路径",
        "description": "根据你的需求自由探索",
        "difficulty": "自定义",
        "duration": "灵活",
        "modules": []  # 用户自己选择
    }
}


# 概念依赖关系图 - 完整版
# 概念依赖关系图 - 按层次结构组织
CONCEPT_DEPENDENCIES = {
    # === 第一层：数学基础（无前置依赖）===
    "矩阵": [],
    "向量": [],
    "导数": [],
    "概率": [],
    "范数": [],
    "优化": [],
    "正交性": [],
    "内积": [],
    "几何": [],
    "图论": [],
    "博弈论": [],
    "微积分": [],
    
    # === 第二层：基础数学工具 ===
    "线性变换": ["矩阵", "向量"],
    "特征值": ["矩阵"],
    "特征向量": ["矩阵", "特征值"],
    "SVD": ["矩阵", "正交性"],
    "偏导数": ["导数"],
    "梯度": ["偏导数", "向量"],
    "链式法则": ["导数"],
    "熵": ["概率"],
    "条件概率": ["概率"],
    "正交基": ["向量", "内积"],
    "泛化误差": ["概率"],
    "样本复杂度": ["概率"],
    "梯度流": ["梯度"],
    "约束优化": ["优化"],
    "贝尔曼方程": [],
    "最优传输": [],
    "条件独立": ["概率"],
    "DAG": [],
    "互信息": ["熵"],
    "潜在结果": [],
    "干预": [],
    
    # === 第三层：机器学习基础 ===
    "梯度下降": ["梯度"],
    "反向传播": ["链式法则", "梯度"],
    "最小二乘": ["范数", "优化"],
    "交叉熵": ["熵", "概率"],
    "KL散度": ["熵", "概率"],
    "最大似然": ["概率", "优化"],
    "泛化": ["泛化误差"],
    "泛化界": ["泛化误差"],
    "正则化": ["优化"],
    
    # === 第四层：优化算法 ===
    "动量": ["梯度下降"],
    "Adam": ["梯度下降", "动量"],
    "学习率调度": ["梯度下降"],
    "批归一化": ["梯度下降"],
    
    # === 第五层：正则化与泛化 ===
    "L1正则化": ["范数", "优化"],
    "L2正则化": ["范数", "优化"],
    "权重衰减": ["L2正则化"],
    "Dropout": ["正则化"],
    
    # === 第六层：高级优化理论 ===
    "拉格朗日乘子": ["约束优化"],
    "KKT条件": ["拉格朗日乘子"],
    "对偶问题": ["拉格朗日乘子"],
    "凸优化": ["优化"],
    
    # === 第七层：核方法与SVM ===
    "内积空间": ["向量"],
    "希尔伯特空间": ["内积空间"],
    "核函数": ["内积空间"],
    "核技巧": ["核函数", "希尔伯特空间"],
    "SVM": ["拉格朗日乘子", "对偶问题", "核技巧"],
    
    # === 第八层：深度学习架构 ===
    "卷积": ["线性变换"],
    "池化": ["卷积"],
    "感受野": ["卷积"],
    "残差连接": ["梯度流"],
    "Softmax": ["概率"],
    "注意力机制": ["Softmax", "内积"],
    "Transformer": ["注意力机制", "残差连接"],
    
    # === 第九层：频域与信号处理 ===
    "傅里叶变换": ["正交基"],
    "卷积定理": ["傅里叶变换"],
    "STFT": ["傅里叶变换"],
    "小波变换": ["傅里叶变换"],
    
    # === 第十层：概率图模型 ===
    "采样": ["概率"],
    "贝叶斯网络": ["条件概率"],
    "变分推断": ["KL散度", "优化"],
    "MCMC": ["概率", "采样"],
    "ELBO": ["变分推断", "KL散度"],
    
    # === 第十一层：生成模型 ===
    "对抗训练": ["优化"],
    "得分匹配": ["梯度"],
    "SDE": [],
    "变量变换": ["微积分"],
    "雅可比": ["矩阵", "导数"],
    "VAE": ["变分推断", "ELBO"],
    "GAN": ["对抗训练", "纳什均衡"],
    "扩散模型": ["得分匹配", "SDE"],
    "归一化流": ["变量变换", "雅可比"],
    
    # === 第十二层：强化学习 ===
    "动态规划": ["优化"],
    "时序差分": [],
    "MDP": ["贝尔曼方程"],
    "价值迭代": ["MDP", "动态规划"],
    "策略梯度": ["MDP", "梯度下降"],
    "Q学习": ["MDP", "时序差分"],
    
    # === 第十三层：图神经网络 ===
    "聚合函数": [],
    "图拉普拉斯": ["矩阵", "图论"],
    "谱图卷积": ["图拉普拉斯", "傅里叶变换"],
    "消息传递": ["图论", "聚合函数"],
    "图注意力": ["注意力机制", "图论"],
    
    # === 第十四层：信息几何 ===
    "黎曼几何": ["几何"],
    "费雪信息": ["概率", "梯度"],
    "自然梯度": ["费雪信息", "黎曼几何"],
    "KL球": ["KL散度", "几何"],
    
    # === 第十五层：泛化理论 ===
    "PAC学习": ["泛化误差"],
    "VC维": ["PAC学习", "样本复杂度"],
    "Rademacher复杂度": ["泛化界"],
    
    # === 第十六层：因果推断 ===
    "因果图": ["DAG", "条件独立"],
    "Do算子": ["因果图", "干预"],
    "反事实": ["因果图", "潜在结果"],
    
    # === 第十七层：最优传输 ===
    "熵正则化": ["熵"],
    "Wasserstein距离": ["最优传输"],
    "Kantorovich对偶": ["Wasserstein距离"],
    "Sinkhorn": ["Wasserstein距离", "熵正则化"],
    
    # === 第十八层：博弈论 ===
    "双层优化": ["优化"],
    "纳什均衡": ["博弈论"],
    "Stackelberg": ["纳什均衡", "双层优化"],
    "演化稳定": ["纳什均衡"],
    
    # === 第十九层：多模态学习 ===
    "NCE": ["概率"],
    "流形学习": ["几何"],
    "对比学习": ["互信息", "NCE"],
    "模态对齐": ["流形学习"],
    "CLIP": ["对比学习", "Transformer"],
    
    # === 第二十层：训练动力学 ===
    "无限宽度极限": [],
    "过参数化": [],
    "神经可塑性": [],
    "NTK": ["核技巧", "无限宽度极限"],
    "双下降": ["过参数化", "泛化"],
    "临界学习期": ["神经可塑性"],
    
    # === 实用层：工程工具 ===
    "幂律": [],
    "经验拟合": [],
    "矩阵维度": ["矩阵"],
    "计算复杂度": [],
    "批大小": [],
    "缩放定律": ["幂律", "经验拟合"],
    "参数计算": ["矩阵维度"],
    "FLOPs估算": ["计算复杂度"],
    "显存估算": ["参数计算", "批大小"],
}


# 推荐阅读顺序（全局）- 按难易程度从低到高排序
RECOMMENDED_ORDER = [
    # === 第一阶段：数学基础 (难度 1-2) ===
    "matrix",           # 1. 线性代数基础
    "calculus",         # 2. 微积分基础
    "probability",      # 3. 概率与信息论
    
    # === 第二阶段：机器学习入门 (难度 2-3) ===
    "loss",             # 4. 损失函数
    "optimizer",        # 5. 优化器
    "ml_curves",        # 6. 机器学习曲线
    "noise",            # 7. 噪声理论
    
    # === 第三阶段：模型与正则化 (难度 3) ===
    "regularization",   # 8. L1/L2正则化
    "convolution",      # 9. 卷积神经网络
    "classification_optimization",  # 10. 分类模型优化逻辑
    
    # === 第四阶段：高级理论 (难度 3-4) ===
    "lagrange",         # 11. 拉格朗日乘子法
    "neural_geometry",  # 12. 神经几何维度
    "training_dynamics",# 13. 训练动力学
    "cnn_math_foundations",  # 14. CNN数学基础
    "hilbert_space",    # 15. 希尔伯特空间
    
    # === 第五阶段：深度学习进阶 (难度 4) ===
    "svm",              # 16. 支持向量机
    "kernel_regression",# 17. 核回归与注意力
    "diffusion_model",  # 18. 扩散模型
    "information_geometry",  # 19. 信息几何
    "signal_processing",# 20. 信号处理
    
    # === 第六阶段：专业领域 (难度 4-5) ===
    "vcdim",            # 21. VC维理论
    "vcdim_derivation", # 22. VC维详细推导
    "neuroevolution",   # 23. 神经进化
    "probabilistic_programming",  # 24. 概率编程
    "mdp",              # 25. 马尔可夫决策过程
    
    # === 第七阶段：前沿研究 (难度 5) ===
    "gcn",              # 26. 图神经网络
    "causation",        # 27. 因果推断
    "optimal_transport",# 28. 最优传输理论
    "game_theory",      # 29. 博弈论
    "multimodal_geometry",  # 30. 多模态几何
    
    # === 第八阶段：实用附录 (难度 2-3) ===
    "scaling_laws",     # 31. 缩放定律
    "dimensions_parameters",  # 32. 维度与参数估算
]


def get_path_by_background(background):
    """根据用户背景返回推荐路径"""
    if background == "新手":
        return LEARNING_PATHS["beginner"]
    elif background == "深度学习":
        return LEARNING_PATHS["deep_learning"]
    elif background == "理论研究":
        return LEARNING_PATHS["theory"]
    elif background == "工程实践":
        return LEARNING_PATHS["practitioner"]
    else:
        return LEARNING_PATHS["custom"]


def get_next_module(current_module, path_name="beginner"):
    """获取当前模块的下一个推荐模块"""
    path = LEARNING_PATHS.get(path_name, LEARNING_PATHS["beginner"])
    modules = path["modules"]
    
    for i, module in enumerate(modules):
        if module["module"] == current_module and i < len(modules) - 1:
            return modules[i + 1]
    
    return None


def get_prerequisites(module_key):
    """获取某个模块的先修要求"""
    for path in LEARNING_PATHS.values():
        for module in path.get("modules", []):
            if module["module"] == module_key:
                return module.get("prerequisites", [])
    return []


def estimate_completion_time(modules_list):
    """估算完成一组模块需要的时间"""
    total_days = 0
    for path in LEARNING_PATHS.values():
        for module in path.get("modules", []):
            if module["module"] in modules_list:
                time_str = module.get("time", "0天")
                days = int(time_str.replace("天", ""))
                total_days += days
    
    return f"{total_days}天" if total_days > 0 else "未知"


def get_difficulty_score(module_key):
    """获取模块难度分数 (1-5)"""
    difficulty_map = {
        # 第一阶段：数学基础
        "matrix": 2,
        "calculus": 2,
        "probability": 3,
        
        # 第二阶段：机器学习入门
        "loss": 2,
        "optimizer": 2,
        "ml_curves": 2,
        "noise": 2,
        
        # 第三阶段：模型与正则化
        "regularization": 3,
        "convolution": 3,
        "classification_optimization": 3,
        
        # 第四阶段：高级理论
        "lagrange": 4,
        "neural_geometry": 3,
        "training_dynamics": 4,
        "cnn_math_foundations": 4,
        "hilbert_space": 4,
        
        # 第五阶段：深度学习进阶
        "svm": 4,
        "kernel_regression": 4,
        "diffusion_model": 4,
        "information_geometry": 4,
        "signal_processing": 4,
        
        # 第六阶段：专业领域
        "vcdim": 5,
        "vcdim_derivation": 5,
        "neuroevolution": 4,
        "probabilistic_programming": 4,
        "mdp": 4,
        
        # 第七阶段：前沿研究
        "gcn": 5,
        "causation": 5,
        "optimal_transport": 5,
        "game_theory": 5,
        "multimodal_geometry": 5,
        
        # 第八阶段：实用附录
        "scaling_laws": 3,
        "dimensions_parameters": 2,
    }
    return difficulty_map.get(module_key, 3)


def get_module_connections(module_key):
    """获取与该模块相关联的其他模块（学习路径推荐）"""
    connections = {
        # 第一阶段：数学基础
        "matrix": ["calculus", "lagrange", "neural_geometry"],
        "calculus": ["loss", "optimizer", "training_dynamics"],
        "probability": ["vcdim", "information_geometry", "probabilistic_programming"],
        
        # 第二阶段：机器学习入门
        "loss": ["optimizer", "classification_optimization"],
        "optimizer": ["regularization", "training_dynamics"],
        "ml_curves": ["noise", "vcdim"],
        "noise": ["regularization", "vcdim"],
        
        # 第三阶段：模型与正则化
        "regularization": ["svm", "vcdim", "training_dynamics"],
        "convolution": ["cnn_math_foundations", "hilbert_space"],
        "classification_optimization": ["svm", "lagrange"],
        
        # 第四阶段：高级理论
        "lagrange": ["svm", "optimal_transport"],
        "neural_geometry": ["scaling_laws", "dimensions_parameters"],
        "training_dynamics": ["neuroevolution", "scaling_laws"],
        "cnn_math_foundations": ["hilbert_space", "signal_processing"],
        "hilbert_space": ["kernel_regression", "gcn"],
        
        # 第五阶段：深度学习进阶
        "svm": ["vcdim", "kernel_regression"],
        "kernel_regression": ["diffusion_model", "gcn"],
        "diffusion_model": ["optimal_transport", "probabilistic_programming"],
        "information_geometry": ["optimal_transport", "game_theory"],
        "signal_processing": ["gcn", "multimodal_geometry"],
        
        # 第六阶段：专业领域
        "vcdim": ["vcdim_derivation"],
        "vcdim_derivation": [],
        "neuroevolution": ["game_theory"],
        "probabilistic_programming": ["causation"],
        "mdp": ["neuroevolution", "game_theory"],
        
        # 第七阶段：前沿研究
        "gcn": ["causation", "multimodal_geometry"],
        "causation": [],
        "optimal_transport": ["game_theory"],
        "game_theory": [],
        "multimodal_geometry": [],
        
        # 第八阶段：实用附录
        "scaling_laws": ["dimensions_parameters"],
        "dimensions_parameters": [],
    }
    return connections.get(module_key, [])


def recommend_next_modules(completed_modules, current_path=None):
    """智能推荐下一步应该学习的模块"""
    from config import MODULES
    
    recommendations = []
    
    # 如果有当前路径，优先推荐路径中的下一个
    if current_path and current_path in LEARNING_PATHS:
        path = LEARNING_PATHS[current_path]
        for module in path.get("modules", []):
            module_key = module["module"]
            if module_key not in completed_modules:
                # 检查先修条件是否满足
                prereqs = module.get("prerequisites", [])
                if all(p in completed_modules for p in prereqs):
                    recommendations.append({
                        "module": module_key,
                        "title": MODULES[module_key]["name"],
                        "reason": f"路径推荐：{path['name']}的下一步",
                        "priority": 10,
                        "difficulty": get_difficulty_score(module_key),
                        "time": module.get("time", "未知")
                    })
                    break
    
    # 基于已完成模块推荐相关模块
    for completed in completed_modules:
        connected = get_module_connections(completed)
        for conn in connected:
            if conn not in completed_modules:
                # 检查先修条件
                prereqs = get_prerequisites(conn)
                if all(p in completed_modules for p in prereqs):
                    recommendations.append({
                        "module": conn,
                        "title": MODULES[conn]["name"],
                        "reason": f"因为你已学习了 {MODULES[completed]['name']}",
                        "priority": 5,
                        "difficulty": get_difficulty_score(conn),
                        "time": "3天"
                    })
    
    # 推荐基础模块（如果还没学）
    basic_modules = ["matrix", "calculus", "loss"]
    for basic in basic_modules:
        if basic not in completed_modules:
            recommendations.append({
                "module": basic,
                "title": MODULES[basic]["name"],
                "reason": "基础模块推荐",
                "priority": 3,
                "difficulty": get_difficulty_score(basic),
                "time": "3天"
            })
    
    # 去重并排序
    seen = set()
    unique_recs = []
    for rec in recommendations:
        if rec["module"] not in seen:
            seen.add(rec["module"])
            unique_recs.append(rec)
    
    unique_recs.sort(key=lambda x: (-x["priority"], x["difficulty"]))
    
    return unique_recs[:5]  # 返回前5个推荐


def get_learning_stats(completed_modules):
    """获取学习统计信息"""
    from config import MODULES
    
    total_modules = len(MODULES)
    completed_count = len(completed_modules)
    completion_rate = (completed_count / total_modules * 100) if total_modules > 0 else 0
    
    # 计算已学习的概念数量
    concepts_learned = set()
    for module_key in completed_modules:
        for path in LEARNING_PATHS.values():
            for module in path.get("modules", []):
                if module["module"] == module_key:
                    concepts_learned.update(module.get("key_concepts", []))
    
    # 计算平均难度
    difficulties = [get_difficulty_score(m) for m in completed_modules]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
    
    # 估算总学习时间
    total_time = estimate_completion_time(list(completed_modules))
    
    return {
        "total_modules": total_modules,
        "completed_count": completed_count,
        "completion_rate": completion_rate,
        "concepts_count": len(concepts_learned),
        "avg_difficulty": avg_difficulty,
        "total_time": total_time,
        "concepts": list(concepts_learned)
    }


# 学习成就系统
ACHIEVEMENTS = {
    "first_steps": {
        "name": "🌱 初出茅庐",
        "description": "完成第一个模块",
        "condition": lambda stats: stats["completed_count"] >= 1
    },
    "fundamentals": {
        "name": "📚 基础扎实",
        "description": "完成矩阵论、微积分和概率论",
        "condition": lambda stats: all(m in stats.get("completed_set", set()) 
                                      for m in ["matrix", "calculus", "probability"])
    },
    "optimizer_master": {
        "name": "⚡ 优化大师",
        "description": "完成损失函数、优化器和正则化",
        "condition": lambda stats: all(m in stats.get("completed_set", set()) 
                                      for m in ["loss", "optimizer", "regularization"])
    },
    "theorist": {
        "name": "🎓 理论家",
        "description": "完成VC维、拉格朗日和SVM",
        "condition": lambda stats: all(m in stats.get("completed_set", set()) 
                                      for m in ["vcdim", "lagrange", "svm"])
    },
    "halfway": {
        "name": "🎯 半程英雄",
        "description": "完成50%的模块",
        "condition": lambda stats: stats["completion_rate"] >= 50
    },
    "completionist": {
        "name": "🏆 完美主义者",
        "description": "完成所有模块",
        "condition": lambda stats: stats["completion_rate"] >= 100
    },
    "concept_collector": {
        "name": "💡 概念收集家",
        "description": "学习了超过30个核心概念",
        "condition": lambda stats: stats["concepts_count"] >= 30
    },
    "deep_diver": {
        "name": "🔬 深度探索者",
        "description": "平均学习难度达到4.0",
        "condition": lambda stats: stats["avg_difficulty"] >= 4.0
    }
}


def check_achievements(completed_modules):
    """检查已解锁的成就"""
    stats = get_learning_stats(completed_modules)
    stats["completed_set"] = completed_modules
    
    unlocked = []
    for key, achievement in ACHIEVEMENTS.items():
        if achievement["condition"](stats):
            unlocked.append({
                "key": key,
                "name": achievement["name"],
                "description": achievement["description"]
            })
    
    return unlocked
