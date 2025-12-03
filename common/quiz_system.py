"""
交互式测验系统
为每个模块提供知识测验功能，增强学习效果
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import random


class Quiz:
    """单个测验题目"""
    
    def __init__(self, 
                 question: str,
                 options: List[str],
                 correct_answer: int,
                 explanation: str = "",
                 hint: str = ""):
        """
        初始化测验题目
        
        Args:
            question: 问题文本
            options: 选项列表
            correct_answer: 正确答案索引 (0-based)
            explanation: 答案解释
            hint: 提示信息
        """
        self.question = question
        self.options = options
        self.correct_answer = correct_answer
        self.explanation = explanation
        self.hint = hint


class QuizSystem:
    """测验系统管理器"""
    
    def __init__(self, module_name: str):
        """
        初始化测验系统
        
        Args:
            module_name: 模块名称，用于session_state的key
        """
        self.module_name = module_name
        self.quiz_key = f"quiz_{module_name}"
        self.score_key = f"quiz_score_{module_name}"
        self.completed_key = f"quiz_completed_{module_name}"
        
        # 初始化session state
        if self.score_key not in st.session_state:
            st.session_state[self.score_key] = 0
        if self.completed_key not in st.session_state:
            st.session_state[self.completed_key] = set()
    
    def render_quiz(self, quizzes: List[Quiz], shuffle: bool = False):
        """
        渲染测验界面
        
        Args:
            quizzes: 测验题目列表
            shuffle: 是否随机打乱题目顺序
        """
        if not quizzes:
            return
        
        st.markdown("---")
        st.markdown("### 🎯 知识检验")
        st.markdown("通过以下测验巩固你的理解：")
        
        # 是否打乱题目
        quiz_list = quizzes.copy()
        if shuffle:
            random.shuffle(quiz_list)
        
        # 显示当前得分
        total_questions = len(quiz_list)
        current_score = len(st.session_state[self.completed_key])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("已完成", f"{current_score}/{total_questions}")
        with col2:
            percentage = (current_score / total_questions * 100) if total_questions > 0 else 0
            st.metric("完成率", f"{percentage:.0f}%")
        with col3:
            if current_score == total_questions:
                st.success("🎉 全部完成！")
        
        st.markdown("")
        
        # 渲染每个题目
        for idx, quiz in enumerate(quiz_list):
            self._render_single_quiz(quiz, idx, total_questions)
        
        # 重置按钮
        if st.button("🔄 重置测验", key=f"reset_{self.module_name}"):
            st.session_state[self.completed_key] = set()
            st.session_state[self.score_key] = 0
            st.rerun()
    
    def _render_single_quiz(self, quiz: Quiz, idx: int, total: int):
        """渲染单个测验题目"""
        quiz_id = f"{self.quiz_key}_{idx}"
        is_completed = quiz_id in st.session_state[self.completed_key]
        
        # 题目容器
        with st.container():
            # 题目标题
            status_icon = "✅" if is_completed else "❓"
            st.markdown(f"#### {status_icon} 题目 {idx + 1}/{total}")
            st.markdown(quiz.question)
            
            # 显示提示按钮
            if quiz.hint and not is_completed:
                with st.expander("💡 需要提示？"):
                    st.info(quiz.hint)
            
            # 选项单选
            answer_key = f"{quiz_id}_answer"
            if answer_key not in st.session_state:
                st.session_state[answer_key] = None
            
            selected = st.radio(
                "请选择答案：",
                options=range(len(quiz.options)),
                format_func=lambda x: quiz.options[x],
                key=f"{quiz_id}_radio",
                disabled=is_completed
            )
            
            # 提交按钮
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_btn = st.button(
                    "提交答案" if not is_completed else "已完成",
                    key=f"{quiz_id}_submit",
                    disabled=is_completed,
                    type="primary" if not is_completed else "secondary"
                )
            
            # 检查答案
            if submit_btn and not is_completed:
                if selected == quiz.correct_answer:
                    st.success("✅ 回答正确！")
                    st.session_state[self.completed_key].add(quiz_id)
                    st.session_state[self.score_key] += 1
                    if quiz.explanation:
                        with st.expander("📖 查看详细解释"):
                            st.markdown(quiz.explanation)
                    st.rerun()
                else:
                    st.error("❌ 回答错误，请再试一次！")
                    correct_option = quiz.options[quiz.correct_answer]
                    st.info(f"💡 提示：正确答案是「{correct_option}」")
                    if quiz.explanation:
                        with st.expander("📖 查看详细解释"):
                            st.markdown(quiz.explanation)
            
            # 如果已完成，显示正确答案
            if is_completed:
                st.success(f"✅ 正确答案：{quiz.options[quiz.correct_answer]}")
                if quiz.explanation:
                    with st.expander("📖 查看解释"):
                        st.markdown(quiz.explanation)
            
            st.markdown("---")
    
    def create_quick_quiz(self, 
                         question: str,
                         options: List[str],
                         correct_idx: int,
                         explanation: str = "") -> Quiz:
        """快速创建一个测验题目"""
        return Quiz(question, options, correct_idx, explanation)


# 预定义一些常用的测验题目模板
class QuizTemplates:
    """测验题目模板库"""
    
    @staticmethod
    def get_loss_function_quizzes() -> List[Quiz]:
        """损失函数相关测验"""
        return [
            Quiz(
                question="MSE的梯度∂L/∂ŷ=-(y-ŷ)，MAE的梯度是sign(y-ŷ)。这导致什么本质区别？",
                options=[
                    "MSE梯度随误差线性变化，MAE恒定为±1",
                    "MSE更快",
                    "MAE更准确",
                    "没有区别"
                ],
                correct_answer=0,
                explanation="MSE梯度与误差成正比，大误差有大梯度（快速修正但易受异常值影响）。MAE梯度恒为±1，对所有误差一视同仁（鲁棒但收敛慢）。",
                hint="梯度大小"
            ),
            Quiz(
                question="Huber损失L_δ在|e|=δ处不可导。实际训练时为什么没问题？",
                options=[
                    "因为永远不会精确等于δ",
                    "用次梯度(subgradient)，在δ处梯度可取[-δ,δ]任意值",
                    "会导致训练失败",
                    "需要特殊处理"
                ],
                correct_answer=1,
                explanation="凸函数在不可导点可用次梯度。Huber在δ处左导数-δ，右导数δ，SGD可用其间任意值。实践中几乎不影响。",
                hint="凸优化理论"
            ),
            Quiz(
                question="为什么分类用Cross-Entropy而非MSE？考虑y∈{0,1}，ŷ=σ(z)的情况。",
                options=[
                    "MSE的梯度∂L/∂z=σ'(z)(ŷ-y)在ŷ接近0/1时消失",
                    "Cross-Entropy更快",
                    "MSE不能用于分类",
                    "Cross-Entropy更简单"
                ],
                correct_answer=0,
                explanation="MSE梯度含σ'(z)项，当预测极端错误（ŷ≈0但y=1）时σ'≈0导致学习停滞。Cross-Entropy梯度∝(ŷ-y)无此问题。",
                hint="饱和问题"
            ),
            Quiz(
                question="Focal Loss: FL(p)=-(1-p)^γ·log(p)相比CE增加了(1-p)^γ项。这解决什么？",
                options=[
                    "加速训练",
                    "降低易分类样本权重，关注hard examples（类别不平衡）",
                    "增加模型容量",
                    "防止过拟合"
                ],
                correct_answer=1,
                explanation="当p(正确类)→1时，(1-p)^γ→0，大幅降低易分类样本的loss。模型聚焦于难分类样本，解决类别不平衡问题。γ=2常用。",
                hint="RetinaNet"
            )
        ]
    
    @staticmethod
    def get_calculus_quizzes() -> List[Quiz]:
        """微积分相关测验"""
        return [
            Quiz(
                question="泰勒展开f(x)=Σf^(n)(a)(x-a)^n/n!为什么能用多项式逼近？本质是什么？",
                options=[
                    "多项式简单",
                    "用各阶导数匹配目标函数在a点的局部性质（0阶值、1阶斜率、2阶曲率...）",
                    "计算快",
                    "只是近似"
                ],
                correct_answer=1,
                explanation="n阶泰勒多项式在a点匹配f的0到n阶导数。余项O((x-a)^(n+1))量化误差。解析函数可无穷展开。",
                hint="导数是局部信息"
            ),
            Quiz(
                question="反向传播中∂L/∂w=∂L/∂z·∂z/∂w。为何从后往前算比从前往后快？",
                options=[
                    "没区别",
                    "后向传播复用∂L/∂z，避免重复计算（动态规划）",
                    "前向更快",
                    "随机的"
                ],
                correct_answer=1,
                explanation="后向：先算∂L/∂z₃，复用于所有w。前向：每个w都要算一遍完整链。计算图中后向=O(边)，前向=O(边×参数)。",
                hint="DAG上的动态规划"
            ),
            Quiz(
                question="梯度消失：sigmoid的导数σ'(x)=σ(1-σ)∈[0,0.25]。L层网络会怎样？",
                options=[
                    "没影响",
                    "梯度∝(0.25)^L指数衰减，深层网络无法训练",
                    "梯度爆炸",
                    "收敛更快"
                ],
                correct_answer=1,
                explanation="链式法则：∂L/∂w₁=∂L/∂z_L·∏σ'(z_i)·∂z₁/∂w₁。每层乘≤0.25，L层后→0。解决：ReLU(导数0或1)、残差连接、BN。",
                hint="连乘效应"
            ),
            Quiz(
                question="二阶导数（Hessian）H_{ij}=∂²f/∂w_i∂w_j有什么用？为何实践中少用？",
                options=[
                    "没用",
                    "提供曲率信息，牛顿法更快，但O(n²)空间和O(n³)计算",
                    "总是更好",
                    "只能用一阶"
                ],
                correct_answer=1,
                explanation="牛顿法：w-=H⁻¹∇f，考虑曲率，收敛更快。但n=10⁶参数时H需1TB内存。实践用近似：L-BFGS、Adam（对角近似）。",
                hint="空间复杂度"
            )
        ]
    
    @staticmethod
    def get_optimizer_quizzes() -> List[Quiz]:
        """优化器相关测验"""
        return [
            Quiz(
                question="动量更新v_t=βv_{t-1}+∇f，为什么能加速收敛？从特征值角度看。",
                options=[
                    "增大学习率",
                    "在一致方向累积速度，抑制震荡方向（相当于低通滤波）",
                    "减少计算量",
                    "避免过拟合"
                ],
                correct_answer=1,
                explanation="一致梯度方向累积增强，震荡方向相互抵消。类似IIR滤波器，β=0.9时相当于对最近10步梯度加权平均。",
                hint="信号处理视角"
            ),
            Quiz(
                question="Adam的m_t(一阶矩)和v_t(二阶矩)分别近似什么？为什么需要偏差修正？",
                options=[
                    "均值和方差，因为初始化为0会偏向0",
                    "梯度和Hessian",
                    "动量和学习率",
                    "不需要修正"
                ],
                correct_answer=0,
                explanation="m_t≈E[g]，v_t≈E[g²]。初始m_0=v_0=0导致初期估计偏小。修正：m̂=m_t/(1-β₁ᵗ)，v̂=v_t/(1-β₂ᵗ)。",
                hint="指数移动平均"
            ),
            Quiz(
                question="为什么Adam的更新η·m̂/(√v̂+ε)相当于自适应学习率？",
                options=[
                    "每个参数的有效学习率≈η/√E[g²]，梯度大的参数学习率小",
                    "所有参数用相同学习率",
                    "随机调整学习率",
                    "学习率固定"
                ],
                correct_answer=0,
                explanation="√v̂≈RMS(g)是梯度的均方根。相当于每个参数按其历史梯度幅度归一化，实现参数级自适应学习率。",
                hint="RMS归一化"
            ),
            Quiz(
                question="学习率衰减schedule：为什么常用余弦退火而非简单线性衰减？",
                options=[
                    "计算更快",
                    "余弦提供平滑的衰减+周期性重启，帮助跳出局部最优",
                    "余弦更简单",
                    "没有区别"
                ],
                correct_answer=1,
                explanation="余弦退火：lr=lr_min+(lr_max-lr_min)·(1+cos(πt/T))/2。平滑衰减避免突变。Warm restarts能跳出sharp minima。",
                hint="SGDR"
            )
        ]
    
    @staticmethod
    def get_matrix_quizzes() -> List[Quiz]:
        """矩阵变换相关测验"""
        return [
            Quiz(
                question="为什么det(AB)=det(A)·det(B)？从体积角度理解。",
                options=[
                    "定义如此",
                    "行列式度量体积，连续变换的体积变化率相乘",
                    "矩阵乘法规则",
                    "只是巧合"
                ],
                correct_answer=1,
                explanation="det(A)是A变换后的体积缩放因子。先B后A的总缩放=det(B)·det(A)=det(AB)。det=0意味着降维（体积→0）。",
                hint="几何解释"
            ),
            Quiz(
                question="SVD分解A=UΣV^T中，U、Σ、V^T分别代表什么变换？",
                options=[
                    "三个旋转",
                    "旋转V^T、缩放Σ、旋转U（任意矩阵=旋转+缩放+旋转）",
                    "三个缩放",
                    "无几何意义"
                ],
                correct_answer=1,
                explanation="V^T：输入空间旋转到主轴；Σ：沿主轴缩放（奇异值）；U：输出空间旋转。将复杂变换分解为简单操作。",
                hint="旋转-缩放-旋转"
            ),
            Quiz(
                question="协方差矩阵C的特征向量是PCA的主成分。为什么？",
                options=[
                    "定义如此",
                    "特征向量是方差最大的方向（Cv=λv，λ是该方向方差）",
                    "计算简单",
                    "随机选择"
                ],
                correct_answer=1,
                explanation="max v^TCv s.t. ||v||=1的解是最大特征值对应的特征向量。λ=v^TCv是该方向的方差。依次找正交方向，得主成分。",
                hint="拉格朗日乘数法"
            ),
            Quiz(
                question="矩阵的秩rank(A)和核空间Null(A)有什么关系？为什么？",
                options=[
                    "无关系",
                    "rank(A) + dim(Null(A)) = n（秩-零化度定理）",
                    "rank(A) = dim(Null(A))",
                    "rank(A) · dim(Null(A)) = n"
                ],
                correct_answer=1,
                explanation="n维输入分解为两部分：rank(A)维被映射到非零（列空间），dim(Null(A))维被映射到零。总维度守恒：n=r+n-r。",
                hint="维数公式"
            )
        ]
    
    @staticmethod
    def get_probability_quizzes() -> List[Quiz]:
        """概率论相关测验"""
        return [
            Quiz(
                question="贝叶斯公式P(H|E)=P(E|H)P(H)/P(E)。为什么P(E|H)≠P(H|E)？举例说明。",
                options=[
                    "相等",
                    "因果不对称：P(症状|疾病)≠P(疾病|症状)，需考虑先验P(H)",
                    "计算错误",
                    "无意义"
                ],
                correct_answer=1,
                explanation="例：P(阳性|癌症)=0.9高，但P(癌症|阳性)可能很低，因为P(癌症)很小。贝叶斯融合了似然、先验和证据。",
                hint="罕见病检测"
            ),
            Quiz(
                question="中心极限定理：n个i.i.d.随机变量之和趋向正态分布。为什么这如此重要？",
                options=[
                    "证明正态分布存在",
                    "解释为何自然界大量现象呈正态分布（多因素叠加）",
                    "只是理论结果",
                    "没有应用"
                ],
                correct_answer=1,
                explanation="大量独立随机因素的总效应→正态分布（无论单个分布）。例：测量误差、人的身高、考试成绩。统计推断的基础。",
                hint="多因素叠加"
            ),
            Quiz(
                question="协方差Cov(X,Y)=E[(X-μ_X)(Y-μ_Y)]为什么能度量相关性？什么时候=0？",
                options=[
                    "随机定义",
                    "X↑时Y倾向↑则Cov>0；独立则=0（但=0不一定独立）",
                    "总是正",
                    "无意义"
                ],
                correct_answer=1,
                explanation="Cov衡量线性相关。独立→Cov=0，但Cov=0不保证独立（如Y=X²，E[XY]=0但相关）。相关系数ρ=Cov/σ_Xσ_Y∈[-1,1]归一化。",
                hint="线性相关"
            ),
            Quiz(
                question="最大似然估计(MLE)为何等价于最小化KL散度？",
                options=[
                    "不等价",
                    "argmax Π p(x_i|θ) 等价于 argmin KL(p_data||p_θ)",
                    "MLE更好",
                    "KL更好"
                ],
                correct_answer=1,
                explanation="MLE最大化log似然=最小化-log p_θ(x)=最小化H(p_data,p_θ)=最小化H(p_data)+KL(p_data||p_θ)。交叉熵训练=MLE。",
                hint="信息论视角"
            )
        ]
    
    @staticmethod
    def get_gradient_descent_quizzes() -> List[Quiz]:
        """梯度下降相关测验"""
        return [
            Quiz(
                question="从泰勒展开f(θ+Δθ)≈f(θ)+∇f·Δθ看，为什么负梯度是最速下降方向？",
                options=[
                    "使∇f·Δθ最大化",
                    "使∇f·Δθ最小化（Cauchy-Schwarz不等式）",
                    "使Δθ的模最小",
                    "使Hessian矩阵对角化"
                ],
                correct_answer=1,
                explanation="固定|Δθ|时，内积∇f·Δθ在Δθ=-η∇f时最小（负号使内积为负且最大）。这是Cauchy-Schwarz不等式的结论。",
                hint="内积何时最小"
            ),
            Quiz(
                question="二次函数f(x,y)=100x²+y²，梯度下降收敛轨迹是什么？为什么？",
                options=[
                    "直线快速收敛",
                    "之字形震荡，因为病态条件数κ=100",
                    "螺旋收敛",
                    "发散"
                ],
                correct_answer=1,
                explanation="x方向陡峭（100x²），y方向平缓（y²），形成狭长峡谷。固定学习率导致x方向震荡、y方向缓慢前进。条件数κ=100。",
                hint="峡谷地形"
            ),
            Quiz(
                question="训练中遇到平台期（梯度≈0但调大学习率会震荡），最可能是什么？",
                options=[
                    "全局最小值",
                    "鞍点（Hessian有正负特征值）",
                    "过拟合",
                    "学习率太小"
                ],
                correct_answer=1,
                explanation="鞍点：梯度为0但Hessian有负特征值（某方向是最大值）。高维空间中鞍点远多于局部最小值。动量能帮助逃离。",
                hint="梯度为0≠最小值"
            ),
            Quiz(
                question="为什么大batch size泛化性能往往更差？",
                options=[
                    "计算更慢",
                    "收敛到sharp minima，小batch的噪声帮助找flat minima",
                    "需要更多内存",
                    "梯度估计不准"
                ],
                correct_answer=1,
                explanation="大batch稳定收敛但易陷入尖锐最小值（对参数扰动敏感）。小batch噪声提供隐式正则化，倾向找平坦最小值（泛化更好）。",
                hint="sharp vs flat"
            )
        ]
    
    @staticmethod
    def get_regularization_quizzes() -> List[Quiz]:
        """正则化相关测验"""
        return [
            Quiz(
                question="为什么L1正则化产生稀疏解？从几何和次梯度角度解释。",
                options=[
                    "L1约束是菱形，更可能在轴上与等高线相交；次梯度在0处为[-1,1]",
                    "L1计算更快",
                    "L1随机选择特征",
                    "L1没有稀疏性"
                ],
                correct_answer=0,
                explanation="几何：L1球是菱形，顶点在轴上，易与等高线在轴交点相切。代数：|w|在w=0处次梯度∈[-1,1]，若真实梯度<1可保持w=0。",
                hint="约束优化的KKT条件"
            ),
            Quiz(
                question="Dropout为何等价于模型集成？训练时实际在训练多少个子网络？",
                options=[
                    "只训练1个网络",
                    "训练2^n个网络（n为神经元数），每次随机采样一个",
                    "训练n个网络",
                    "训练无穷多个"
                ],
                correct_answer=1,
                explanation="每个神经元丢弃/保留2种状态，n个神经元有2^n种组合。每次训练随机采样一个子网络。测试时相当于几何平均这2^n个模型。",
                hint="指数级子网络"
            ),
            Quiz(
                question="Early Stopping为什么是一种正则化？从bias-variance角度分析。",
                options=[
                    "减少训练时间",
                    "限制优化步数等价于限制模型复杂度，降低variance",
                    "增加bias",
                    "不是正则化"
                ],
                correct_answer=1,
                explanation="训练步数↑→模型复杂度↑→variance↑。Early stopping在validation误差最小时停止，避免过度优化训练集。等价于L2正则（见论文）。",
                hint="优化≈正则化"
            ),
            Quiz(
                question="Weight Decay和L2正则在标准SGD下等价，但在Adam下不等价。为什么？",
                options=[
                    "完全等价",
                    "Adam的自适应学习率破坏了等价性，需用AdamW（解耦weight decay）",
                    "Adam不能用正则化",
                    "Weight Decay已过时"
                ],
                correct_answer=1,
                explanation="SGD：w-=η(g+λw)等价于w*=(1-ηλ)w-ηg。Adam：自适应lr破坏关系。AdamW直接执行w*=(1-λ)w-η·g，效果更好。",
                hint="AdamW vs Adam"
            )
        ]
    
    @staticmethod
    def get_noise_quizzes() -> List[Quiz]:
        """噪声与鲁棒性相关测验"""
        return [
            Quiz(
                question="为什么添加高斯噪声等价于L2正则化？从贝叶斯角度分析。",
                options=[
                    "不等价",
                    "噪声→后验=似然·先验，高斯噪声似然+高斯先验→MAP=L2正则MLE",
                    "只是巧合",
                    "无关"
                ],
                correct_answer=1,
                explanation="添加噪声y=f(x)+ε，ε~N(0,σ²)相当于似然p(y|w)~N(f(x),σ²)。高斯先验p(w)~N(0,λ⁻¹)下MAP=argmax p(w|D)=MLE+λ||w||²。",
                hint="贝叶斯=MLE+正则"
            ),
            Quiz(
                question="对抗样本x'=x+εsign(∇_x L)为何有效？为什么||ε||∞很小却能欺骗模型？",
                options=[
                    "模型太弱",
                    "高维空间中微小扰动累积，线性模型：Δf=w^T·ε=Σw_i·ε_i，维度n越大影响越大",
                    "随机的",
                    "对抗样本无效"
                ],
                correct_answer=1,
                explanation="虽然每个ε_i很小，但n维累积可以很大。例如n=1000，ε=0.001，若|w_i|≈1，则|Δf|≈1。深度网络的线性性导致脆弱。",
                hint="高维累积效应"
            ),
            Quiz(
                question="对抗训练min_w E_x[max_||δ||≤ε L(x+δ,y;w)]为何能提高鲁棒性？",
                options=[
                    "只是正则化",
                    "在最坏情况（对抗扰动）下优化，学习局部平滑的决策边界",
                    "计算更慢",
                    "无效"
                ],
                correct_answer=1,
                explanation="训练时加入最坏情况样本，相当于min-max game。决策边界在对抗方向上变平滑。但计算昂贵（每步需内循环求max）。",
                hint="鲁棒优化"
            ),
            Quiz(
                question="为何批归一化(BN)能降低对输入噪声的敏感性？",
                options=[
                    "BN不影响噪声",
                    "归一化使输出对输入的幅度变化不变（∂BN(x)/∂x∝1/σ，自适应）",
                    "BN增加噪声",
                    "只是加速训练"
                ],
                correct_answer=1,
                explanation="BN：y=(x-μ)/σ·γ+β。输入尺度变化被σ吸收，输出依赖标准化后的值。Jacobian∂y/∂x∝γ/σ，自动调节对输入扰动的敏感度。",
                hint="归一化的鲁棒性"
            )
        ]
    
    @staticmethod
    def get_ml_curves_quizzes() -> List[Quiz]:
        """机器学习曲线相关测验"""
        return [
            Quiz(
                question="学习曲线：训练误差单调上升，验证误差下降后趋平。说明什么？",
                options=[
                    "过拟合",
                    "欠拟合（high bias），增加数据无效，需增加模型容量",
                    "完美",
                    "数据问题"
                ],
                correct_answer=1,
                explanation="训练误差高且上升→模型容量不足，连训练集都拟合不好。验证误差趋平→增加数据无法改善。解决：更复杂模型、更多特征。",
                hint="bias-variance诊断"
            ),
            Quiz(
                question="ROC曲线下AUC的概率解释是什么？与排序有何关系？",
                options=[
                    "准确率",
                    "P(score(正例)>score(负例))，度量排序质量",
                    "召回率",
                    "F1"
                ],
                correct_answer=1,
                explanation="AUC=随机抽取正负样本对，模型给正例打分更高的概率。等价于Wilcoxon统计量。AUC=1意味着完美排序。",
                hint="成对比较"
            ),
            Quiz(
                question="Precision-Recall曲线下AUPRC在类别极度不平衡（1:1000）时，基线是多少？",
                options=[
                    "0.5",
                    "正类比例（如0.001），随机预测的期望Precision",
                    "1.0",
                    "0"
                ],
                correct_answer=1,
                explanation="随机猜测：Precision=正类比例。例如1:1000时，随机猜测AUPRC≈0.001，远小于ROC-AUC基线0.5。凸显了PR曲线对不平衡的敏感性。",
                hint="不平衡下的基线"
            ),
            Quiz(
                question="混淆矩阵的F_β分数：F_β=(1+β²)·P·R/(β²P+R)。β>1时偏向什么？",
                options=[
                    "Precision",
                    "Recall（β越大越重视Recall，如β=2时Recall权重是P的4倍）",
                    "准确率",
                    "无影响"
                ],
                correct_answer=1,
                explanation="β控制Precision和Recall的权衡。β=1是F1（均衡），β=2是F2（重Recall，如医疗诊断），β=0.5重Precision（如推荐系统）。",
                hint="不同场景的权衡"
            )
        ]
    
    @staticmethod
    def get_svm_quizzes() -> List[Quiz]:
        """支持向量机相关测验"""
        return [
            Quiz(
                question="SVM最大化间隔2/||w||等价于min ||w||²。为何最大间隔能提高泛化？",
                options=[
                    "计算简单",
                    "大间隔→决策边界对扰动鲁棒，VC维分析：d_VC∝R²||w||²（R是数据半径）",
                    "随机的",
                    "只是定义"
                ],
                correct_answer=1,
                explanation="大间隔意味着决策边界远离数据，对噪声和扰动更鲁棒。统计学习理论：间隔越大，VC维越小，泛化误差界越紧。",
                hint="VC维理论"
            ),
            Quiz(
                question="核技巧K(x,x')=⟨φ(x),φ(x')⟩。为什么RBF核K(x,x')=exp(-γ||x-x'||²)对应无限维？",
                options=[
                    "不是无限维",
                    "泰勒展开exp(x)=Σx^n/n!，对应所有阶多项式特征（无穷维）",
                    "只是近似",
                    "有限维"
                ],
                correct_answer=1,
                explanation="RBF核的泰勒展开包含所有阶次，对应无限维特征空间。但通过核技巧，无需显式计算，只需O(d)时间计算核值。",
                hint="泰勒展开"
            ),
            Quiz(
                question="对偶问题max_α Σα_i - ½ΣΣα_iα_jy_iy_jK(x_i,x_j)。为何只依赖内积？",
                options=[
                    "巧合",
                    "拉格朗日对偶+KKT条件导出，原问题的w*=Σα_iy_ix_i用支持向量表示",
                    "定义如此",
                    "无关"
                ],
                correct_answer=1,
                explanation="求解L对w的极值：∂L/∂w=0→w=Σα_iy_ix_i。代入得对偶形式，只依赖内积⟨x_i,x_j⟩→可替换为核K(x_i,x_j)。",
                hint="Representer定理"
            ),
            Quiz(
                question="为何只有支持向量的α_i>0？从KKT互补松弛性解释。",
                options=[
                    "随机的",
                    "KKT：α_i[y_i(w·x_i+b)-1+ξ_i]=0，非支持向量满足约束>0故α_i=0",
                    "所有α都>0",
                    "无意义"
                ],
                correct_answer=1,
                explanation="非支持向量在间隔内或边界外，松弛条件不紧（>0），互补松弛性→α_i=0。只有边界上的点α_i>0，参与决策。稀疏性的来源。",
                hint="稀疏性"
            )
        ]
    
    @staticmethod
    def get_convolution_quizzes() -> List[Quiz]:
        """卷积相关测验"""
        return [
            Quiz(
                question="离散卷积(f*g)[n]=Σf[k]g[n-k]与矩阵乘法有什么关系？",
                options=[
                    "无关",
                    "卷积=特殊的矩阵乘法（Toeplitz矩阵），但参数共享",
                    "完全相同",
                    "卷积更复杂"
                ],
                correct_answer=1,
                explanation="将卷积写成矩阵形式：y=Wx，W是Toeplitz矩阵（每行是核的平移）。全连接无约束，卷积强制参数共享→大幅减少参数。",
                hint="im2col技巧"
            ),
            Quiz(
                question="为什么卷积具有平移不变性？从傅里叶变换角度理解。",
                options=[
                    "定义如此",
                    "傅里叶变换下卷积→乘法，平移→相位，卷积后平移不变",
                    "只是近似",
                    "不具有平移不变性"
                ],
                correct_answer=1,
                explanation="F{f*g}=F{f}·F{g}。平移定理：F{f(x-a)}=e^(-iωa)F{f}。卷积中相位项抵消，输出随输入平移而平移。注意：池化破坏严格平移不变性。",
                hint="频域分析"
            ),
            Quiz(
                question="感受野(Receptive Field)如何随层数增长？k×k核，L层后感受野多大？",
                options=[
                    "k×k",
                    "1+L(k-1)（线性增长），深层能看到更大范围",
                    "k^L",
                    "不变"
                ],
                correct_answer=1,
                explanation="每层增加k-1：RF_L=1+(k-1)L。例如3×3核，10层后RF=21×21。空洞卷积(dilated)可指数增长：RF=1+(k-1)Σ2^i。",
                hint="递推关系"
            ),
            Quiz(
                question="深度可分离卷积将标准卷积分解为逐通道(depthwise)+逐点(pointwise)。参数减少多少？",
                options=[
                    "不变",
                    "从C_in·C_out·k²降到C_in·k²+C_in·C_out，约1/C_out+1/k²",
                    "减少一半",
                    "增加"
                ],
                correct_answer=1,
                explanation="标准：C_in·C_out·k²。分离：(C_in·k²)+(C_in·C_out·1²)。若C_out=256, k=3，参数减少约9倍。MobileNet的核心。",
                hint="MobileNet"
            )
        ]
    
    @staticmethod
    def get_vcdim_quizzes() -> List[Quiz]:
        """VC维相关测验"""
        return [
            Quiz(
                question="VC维定义：能打散的最大点集。为何d维线性分类器VC维=d+1而非d？",
                options=[
                    "错误，应该是d",
                    "参数有d+1个（w∈R^d, b∈R），对应d+1个自由度",
                    "随机的",
                    "无穷大"
                ],
                correct_answer=1,
                explanation="线性分类器w·x+b=0有d+1个参数。可打散d+1个一般位置的点（如单纯形顶点），但无法打散所有d+2点配置。VC维=参数数。",
                hint="参数数量"
            ),
            Quiz(
                question="PAC学习：以概率1-δ，泛化误差≤训练误差+O(√(d·log(n/d)/n))。这说明什么？",
                options=[
                    "VC维无关",
                    "需要n≫d才能保证泛化，VC维d是样本复杂度的关键",
                    "n越小越好",
                    "δ无关紧要"
                ],
                correct_answer=1,
                explanation="误差界∝√(d/n)。要使泛化误差小，需n≫d（通常n>10d）。VC维d控制了学习该假设类所需的样本数。",
                hint="样本复杂度"
            ),
            Quiz(
                question="Shattering：对4个点的XOR配置{(+,+,−,−)}，为何2D线性分类器无法打散？",
                options=[
                    "可以打散",
                    "XOR不线性可分，任何直线无法分开对角的+和−",
                    "需要更多点",
                    "随机的"
                ],
                correct_answer=1,
                explanation="XOR：(0,0)→+, (0,1)→−, (1,0)→−, (1,1)→+。对角点同类但被另外两点分隔，线性不可分。证明VC维<4，实际=3。",
                hint="对角线问题"
            ),
            Quiz(
                question="神经网络VC维：W个权重，d_VC=O(W·log W)。为何不是O(W)？",
                options=[
                    "应该是O(W)",
                    "网络的非线性组合能力超过参数数，但受限于拓扑（对数因子来自深度）",
                    "无穷大",
                    "随机"
                ],
                correct_answer=1,
                explanation="参数数W给出下界。但网络结构限制：不是所有W维函数可表示。深度L层：d_VC≈O(WL)或O(W log W)，取决于激活函数。",
                hint="结构约束"
            )
        ]
    
    @staticmethod
    def get_lagrange_quizzes() -> List[Quiz]:
        """拉格朗日乘数法相关测验"""
        return [
            Quiz(
                question="为什么最优点满足∇f=λ∇g？从梯度与等高线关系理解。",
                options=[
                    "定义如此",
                    "最优点处f和g的梯度必须平行（否则沿约束面还能改进）",
                    "计算简单",
                    "随机的"
                ],
                correct_answer=1,
                explanation="若∇f与∇g不平行，则∇f在约束面g=0的切向有分量，沿此方向可改进f。最优点∇f⊥切面，即∇f∥∇g，故∇f=λ∇g。",
                hint="梯度垂直于等高线"
            ),
            Quiz(
                question="KKT条件中的互补松弛性λ_i·g_i(x)=0是什么意思？",
                options=[
                    "λ和g都为0",
                    "不等式约束要么不活跃(g_i<0,λ_i=0)，要么紧(g_i=0,λ_i>0)",
                    "λ=g",
                    "无意义"
                ],
                correct_answer=1,
                explanation="不活跃约束(g_i<0)不影响最优解，λ_i=0。活跃约束(g_i=0)影响解，λ_i>0。这是非线性规划的核心性质。",
                hint="SVM中支持向量"
            ),
            Quiz(
                question="对偶问题max_λ min_x L(x,λ)与原问题min_x max_λ L(x,λ)何时等价？",
                options=[
                    "总是等价",
                    "强对偶成立时等价（凸优化+Slater条件）",
                    "永不等价",
                    "随机"
                ],
                correct_answer=1,
                explanation="一般有弱对偶：对偶最优≤原问题最优。强对偶需条件：原问题凸+存在严格可行点(Slater)。SVM满足此条件。",
                hint="对偶间隙"
            ),
            Quiz(
                question="为什么L1正则化可用拉格朗日对偶理解？||w||₁≤t对应什么？",
                options=[
                    "无关",
                    "约束优化min L(w) s.t. ||w||₁≤t与惩罚min L(w)+λ||w||₁等价",
                    "L1不可微",
                    "只能用L2"
                ],
                correct_answer=1,
                explanation="约束形式和惩罚形式是同一问题的原问题和对偶问题。不同的λ对应不同的t。Lasso=有约束的最小二乘。",
                hint="正则化的对偶视角"
            )
        ]
    
    @staticmethod
    def get_classification_optimization_quizzes() -> List[Quiz]:
        """分类优化相关测验"""
        return [
            Quiz(
                question="0-1损失L_{0-1}=I(ŷ≠y)不可微。常用替代：Hinge、Logistic、Exp。它们是什么关系？",
                options=[
                    "无关",
                    "都是0-1损失的凸上界：Hinge紧（SVM），Logistic光滑（LR），Exp松（AdaBoost）",
                    "完全相同",
                    "随机的"
                ],
                correct_answer=1,
                explanation="凸替代确保可优化。Hinge: max(0,1-yf)最紧但不可微。Logistic: log(1+exp(-yf))光滑。Exp: exp(-yf)松但boost理论用。",
                hint="凸松弛"
            ),
            Quiz(
                question="Softmax的梯度∂L/∂z_k=p_k-I(k=y)。为何正类梯度∝-(1-p)，负类∝p？",
                options=[
                    "定义如此",
                    "正类：错得越离谱(p小)梯度越大推向正确；负类：置信度高(p大)惩罚重",
                    "随机的",
                    "无意义"
                ],
                correct_answer=1,
                explanation="正类k=y：梯度=p-1，当p→0时梯度→-1（大推力）。负类k≠y：梯度=p，当p→1时梯度→1（强抑制）。自适应调节。",
                hint="自适应学习"
            ),
            Quiz(
                question="Label Smoothing将硬标签y变为(1-ε)y+ε/K。为何能防止过拟合和过置信？",
                options=[
                    "随机噪声",
                    "阻止模型追求极端概率（logit→∞），相当于正则化+校准",
                    "数据增强",
                    "无效"
                ],
                correct_answer=1,
                explanation="硬标签驱动logit→∞（过置信）。平滑后目标概率<1，模型输出有界，防止过拟合。等价于增加熵正则H(p)。改善校准。",
                hint="正则化视角"
            ),
            Quiz(
                question="类别不平衡（1:100）时，为何简单上采样/下采样不如调整损失权重或Focal Loss？",
                options=[
                    "采样总是更好",
                    "采样改变数据分布可能过拟合；调整损失保持分布同时重新加权梯度",
                    "无区别",
                    "采样更快"
                ],
                correct_answer=1,
                explanation="上采样少数类→重复样本过拟合。下采样多数类→丢失信息。加权损失或Focal Loss保留全部数据，仅调节梯度贡献，更鲁棒。",
                hint="加权 vs 采样"
            )
        ]
    
    @staticmethod
    def get_kernel_regression_quizzes() -> List[Quiz]:
        """核回归相关测验"""
        return [
            Quiz(
                question="Nadaraya-Watson估计：ŷ(x)=Σw_i(x)y_i，w_i∝K((x-x_i)/h)。为何是加权平均？",
                options=[
                    "定义如此",
                    "权重归一化Σw_i=1，近点权重大，实现局部平滑估计",
                    "线性回归",
                    "随机的"
                ],
                correct_answer=1,
                explanation="w_i(x)=K((x-x_i)/h)/Σ_j K((x-x_j)/h)归一化为概率。预测是邻域y值的加权平均，权重随距离衰减。",
                hint="非参数平滑"
            ),
            Quiz(
                question="带宽h→0和h→∞时，核回归退化为什么？",
                options=[
                    "不变",
                    "h→0：最近邻插值（过拟合）；h→∞：全局均值（欠拟合）",
                    "都是线性",
                    "随机"
                ],
                correct_answer=1,
                explanation="h→0：只有最近点权重非0→阶跃插值。h→∞：所有点权重相等→ŷ=ȳ全局均值。中间h平衡bias-variance。",
                hint="两个极端"
            ),
            Quiz(
                question="核回归的bias-variance权衡：h小variance高，h大bias高。最优h∝n^(-1/5)为何？",
                options=[
                    "经验值",
                    "MSE=bias²+variance，最小化得h_opt∝n^(-1/(4+d))，d=1时指数1/5",
                    "随机的",
                    "无理论"
                ],
                correct_answer=1,
                explanation="Bias∝h²（泰勒展开）, Variance∝1/(nh^d)。MSE最小化：∂(h⁴+1/(nh^d))/∂h=0→h∝n^(-1/(4+d))。维度诅咒：d大h需更大。",
                hint="维度诅咒"
            ),
            Quiz(
                question="局部多项式回归vs核回归：在边界x=0或1处，NW估计有偏。为何局部线性无偏？",
                options=[
                    "无区别",
                    "NW用常数拟合，边界处不对称→偏；局部线性用线性拟合，自动校正边界效应",
                    "NW总是更好",
                    "随机的"
                ],
                correct_answer=1,
                explanation="边界处邻域不对称，NW的加权平均向内偏。局部线性min Σw_i(y_i-a-b(x_i-x))²拟合斜率b，消除一阶偏差。",
                hint="边界效应"
            )
        ]
    
    @staticmethod
    def get_cnn_math_foundations_quizzes() -> List[Quiz]:
        """CNN数学基础相关测验"""
        return [
            Quiz(
                question="卷积y=x*w参数量O(k²·C_in·C_out)，全连接O(H²W²·C_in·C_out)。减少多少倍？",
                options=[
                    "相同",
                    "约(HW/k)²倍，如224×224图像3×3核减少约5000倍",
                    "只减少2倍",
                    "增加参数"
                ],
                correct_answer=1,
                explanation="卷积：k²CC'参数在HW位置共享。全连接：每个输出位置独立→(HW)²CC'。比值≈(HW/k)²。224图像3核：(224/3)²≈5500倍。",
                hint="参数共享的威力"
            ),
            Quiz(
                question="1×1卷积的作用：既不改变空间尺寸，为何广泛使用（如Inception、ResNet）？",
                options=[
                    "没用",
                    "跨通道信息融合+降维/升维，相当于每个位置的全连接",
                    "只是占位",
                    "加速计算"
                ],
                correct_answer=1,
                explanation="1×1卷积：对每个空间位置的C个通道做线性组合→C'通道。实现通道间交互、bottleneck降维（减少计算）、增加非线性。",
                hint="Network-in-Network"
            ),
            Quiz(
                question="为何池化破坏平移等变性？Max pooling后平移1像素，输出变化多大？",
                options=[
                    "不变",
                    "可能完全不同（取决于池化窗口内最大值位置），破坏了精确等变性",
                    "线性变化",
                    "随机"
                ],
                correct_answer=1,
                explanation="平移等变：f(T(x))=T(f(x))。池化下采样后，小平移可能不改变max位置（不变）或跳到新窗口（突变）。空间信息损失。",
                hint="下采样的代价"
            ),
            Quiz(
                question="为何ResNet能训练1000层？残差连接y=F(x)+x如何解决梯度消失？",
                options=[
                    "参数更多",
                    "梯度直通：∂L/∂x=∂L/∂y·(1+∂F/∂x)，至少有1的恒等项绕过F",
                    "学习率更大",
                    "随机的"
                ],
                correct_answer=1,
                explanation="无残差：梯度连乘→指数衰减。残差：求导链∂y/∂x=1+∂F/∂x，即使∂F/∂x→0，梯度仍能通过恒等映射流动。高速公路机制。",
                hint="梯度高速公路"
            )
        ]
    
    @staticmethod
    def get_gcn_quizzes() -> List[Quiz]:
        """图卷积网络相关测验"""
        return [
            Quiz(
                question="GCN更新h_i'=σ(Σ_{j∈N(i)} W·h_j/√(d_i·d_j))。为何要归一化√(d_i·d_j)？",
                options=[
                    "随机的",
                    "度数归一化防止高度节点主导，保持特征尺度稳定（对称归一化）",
                    "加速计算",
                    "无作用"
                ],
                correct_answer=1,
                explanation="无归一化：高度节点求和项多→特征爆炸。度归一化：除以√(d_i·d_j)使期望保持，类似BatchNorm。等价于D^(-1/2)AD^(-1/2)。",
                hint="图上的归一化"
            ),
            Quiz(
                question="谱GCN：g_θ*x=U·g_θ(Λ)·U^T·x，U是拉普拉斯L的特征向量。为何O(n³)不可行？",
                options=[
                    "可行",
                    "特征分解O(n³)，大图(n>10⁶)计算和存储都不可行，需多项式近似",
                    "很快",
                    "无关"
                ],
                correct_answer=1,
                explanation="L=UΛU^T分解O(n³)时间，O(n²)空间存U。ChebNet用Chebyshev多项式近似→K阶邻居O(K|E|)，GCN取K=1。",
                hint="多项式近似"
            ),
            Quiz(
                question="过平滑(Over-smoothing)：多层GCN后所有节点特征趋同。为什么？",
                options=[
                    "不会发生",
                    "反复邻居平均→特征收敛到全局均值，失去节点区分度（图的低通滤波）",
                    "参数问题",
                    "随机"
                ],
                correct_answer=1,
                explanation="GCN本质是拉普拉斯平滑：h^(l+1)=D^(-1/2)AD^(-1/2)h^(l)。多次迭代→h收敛到最小特征值对应的平稳分布（常向量）。",
                hint="低通滤波效应"
            ),
            Quiz(
                question="图采样：GraphSAGE采样固定K个邻居而非全部。优势和代价是什么？",
                options=[
                    "完全等价",
                    "优势：O(K)复杂度可扩展到大图；代价：引入方差，需更多训练步",
                    "总是更差",
                    "只是加速"
                ],
                correct_answer=1,
                explanation="全邻居聚合：度数不均导致复杂度O(max_degree)不可控。采样：固定K→mini-batch可行，但采样方差需用更多epoch补偿。",
                hint="可扩展性权衡"
            )
        ]
    
    @staticmethod
    def get_mdp_quizzes() -> List[Quiz]:
        """马尔可夫决策过程相关测验"""
        return [
            Quiz(
                question="贝尔曼最优方程V*(s)=max_a[R(s,a)+γΣP(s'|s,a)V*(s')]。为何是不动点？",
                options=[
                    "随机定义",
                    "最优价值是自洽的：最优策略下当前价值=即时奖励+未来最优价值",
                    "近似解",
                    "无意义"
                ],
                correct_answer=1,
                explanation="V*满足自身定义的方程→不动点。价值迭代：V_{k+1}=T(V_k)反复应用Bellman算子T，γ<1保证压缩映射→收敛到唯一不动点V*。",
                hint="压缩映射定理"
            ),
            Quiz(
                question="为何折扣因子γ<1？当γ→1时，MDP有什么问题？",
                options=[
                    "γ=1总是更好",
                    "γ<1保证无限horizon奖励有界Σγ^t·r<∞；γ→1需特殊处理（平均奖励）",
                    "随机选择",
                    "无影响"
                ],
                correct_answer=1,
                explanation="γ<1：几何级数收敛，V有界。γ=1：无限horizon的Σr可能发散，需用平均奖励lim(1/T)Σr或引入终止状态。γ还表示对未来的重视程度。",
                hint="无限级数收敛"
            ),
            Quiz(
                question="策略梯度∇_θJ(θ)=E[∇log π_θ(a|s)·Q(s,a)]。为何梯度∝log概率？",
                options=[
                    "定义如此",
                    "似然比技巧：∇E_π[f]=E_π[f·∇log π]，将期望内外的梯度交换",
                    "随机的",
                    "无关"
                ],
                correct_answer=1,
                explanation="J=E_π[R]，直接求梯度需对分布求导。技巧：∇_θE_π[R]=E_π[R·∇log π]。∇log π是score函数，加权好动作的log概率。REINFORCE算法基础。",
                hint="似然比技巧"
            ),
            Quiz(
                question="Actor-Critic：Actor学π_θ，Critic学V_w。为何比纯策略梯度方差小？",
                options=[
                    "无区别",
                    "Critic提供baseline减方差：∇J∝(Q-V)·∇log π，优势函数A=Q-V中心化",
                    "更慢",
                    "随机"
                ],
                correct_answer=1,
                explanation="纯策略梯度用蒙特卡洛估计Q→高方差。Critic学习V(s)作baseline，优势A(s,a)=Q-V衡量动作相对好坏，减小方差同时保持无偏。",
                hint="优势函数"
            )
        ]
    
    @staticmethod
    def get_causation_quizzes() -> List[Quiz]:
        """因果推断相关测验"""
        return [
            Quiz(
                question="Pearl因果层次：关联→干预→反事实。为何观察数据P(Y|X)不能推断干预P(Y|do(X))？",
                options=[
                    "可以推断",
                    "观察有选择偏差(X由Z决定)，干预切断X的因：P(Y|do(X))≠P(Y|X)",
                    "完全相同",
                    "随机"
                ],
                correct_answer=1,
                explanation="观察：X=f(Z)→Y，Z混杂。干预do(X)：强制设X，切断X←Z边。例：观察到吸烟与肺癌相关，但基因Z→吸烟和肺癌。do(戒烟)≠观察戒烟者。",
                hint="do算子"
            ),
            Quiz(
                question="后门准则：给定Z，若Z阻断X和Y的所有后门路径且不是X后代，则可识别因果效应。为什么？",
                options=[
                    "随机的",
                    "条件化Z消除混杂，保留因果路径X→Y：P(Y|do(X))=ΣP(Y|X,Z)P(Z)",
                    "无关",
                    "近似"
                ],
                correct_answer=1,
                explanation="后门路径：X←Z→Y（混杂）。条件化Z：P(Y|X,Z)中Z固定，阻断混杂路径，只留因果路径。调整公式：加权平均不同Z层的条件概率。",
                hint="调整公式"
            ),
            Quiz(
                question="工具变量(IV)：Z影响X但不直接影响Y（只通过X）。为何能识别因果？",
                options=[
                    "不能",
                    "Z→X的变异是外生的（无混杂），利用这部分变异估计X→Y：β=Cov(Y,Z)/Cov(X,Z)",
                    "随机选择",
                    "无意义"
                ],
                correct_answer=1,
                explanation="存在未观测混杂U→X,Y时，OLS有偏。IV Z：Z→X有关，Z⊥U独立。Z引起的X变异是干净的，用Cov(Y,Z)/Cov(X,Z)估计因果效应（2SLS）。",
                hint="两阶段最小二乘"
            ),
            Quiz(
                question="反事实推断：ATE vs ATT。为何个体因果效应Y_i(1)-Y_i(0)不可观测？",
                options=[
                    "可观测",
                    "根本因果问题：同一个体不能同时处理和不处理，只能观察一个潜在结果",
                    "测量误差",
                    "随机"
                ],
                correct_answer=1,
                explanation="Y_i(1)和Y_i(0)是潜在结果，但只能观察其一。因果推断估计群体效应ATE=E[Y(1)-Y(0)]或ATT=E[Y(1)-Y(0)|T=1]，需假设（如无混杂）。",
                hint="潜在结果框架"
            )
        ]
    
    @staticmethod
    def get_signal_processing_quizzes() -> List[Quiz]:
        """信号处理相关测验"""
        return [
            Quiz(
                question="傅里叶变换F(ω)=∫f(t)e^(-iωt)dt。为何e^(iωt)=cos(ωt)+i·sin(ωt)是频率基？",
                options=[
                    "定义如此",
                    "e^(iωt)是频率ω的复指数，实部cos和虚部sin正交，张成频域空间",
                    "随机选择",
                    "无意义"
                ],
                correct_answer=1,
                explanation="欧拉公式：e^(iωt)组合cos和sin。不同ω的e^(iωt)正交：∫e^(iω₁t)·e^(-iω₂t)dt∝δ(ω₁-ω₂)。F(ω)是f在基e^(iωt)上的投影（内积）。",
                hint="正交基展开"
            ),
            Quiz(
                question="奈奎斯特频率f_N=f_s/2。为何采样频率f_s<2f_max会混叠？从频谱周期性理解。",
                options=[
                    "不会混叠",
                    "采样使频谱以f_s周期复制，f_s<2f_max导致相邻副本重叠（混叠）",
                    "随机的",
                    "无关"
                ],
                correct_answer=1,
                explanation="采样：时域×脉冲串→频域卷积→频谱每f_s重复。若f_max>f_s/2，副本重叠，高频混叠到低频，信息损失。解决：抗混叠滤波器预滤波。",
                hint="频域卷积"
            ),
            Quiz(
                question="卷积定理：时域卷积↔频域乘法，(f*g)(t)↔F(ω)·G(ω)。为何滤波用FFT快？",
                options=[
                    "无区别",
                    "时域卷积O(n²)，FFT变换O(nlogn)后频域相乘O(n)，总O(nlogn)<<n²",
                    "FFT更慢",
                    "随机"
                ],
                correct_answer=1,
                explanation="直接卷积：每个输出点需n次乘加→O(n²)。FFT方法：FFT(f)和FFT(g)各O(nlogn)，乘法O(n)，IFFT O(nlogn)→总O(nlogn)。",
                hint="快速卷积"
            ),
            Quiz(
                question="小波变换vs傅里叶变换：小波提供时频局部化。为何傅里叶不能？",
                options=[
                    "可以",
                    "傅里叶基e^(iωt)无限延伸，只有频率分辨率；小波ψ(t)局部化，有时频分辨率",
                    "小波更差",
                    "相同"
                ],
                correct_answer=1,
                explanation="不确定性原理：Δt·Δω≥1/2。傅里叶：Δt=∞（全局），Δω=0（精确频率）。小波：母小波ψ局部化，缩放+平移→多尺度时频分析。",
                hint="时频不确定性"
            )
        ]
    
    @staticmethod
    def get_diffusion_model_quizzes() -> List[Quiz]:
        """扩散模型相关测验"""
        return [
            Quiz(
                question="前向扩散q(x_t|x_{t-1})=N(√(1-β_t)x_{t-1}, β_t I)。为何T步后x_T≈N(0,I)？",
                options=[
                    "不会",
                    "递推：x_t=√ᾱ_t·x_0+√(1-ᾱ_t)·ε，ᾱ_T→0时x_T≈N(0,I)（噪声主导）",
                    "随机的",
                    "需要无穷步"
                ],
                correct_answer=1,
                explanation="定义ᾱ_t=∏(1-β_s)。递推得x_t=√ᾱ_t·x_0+√(1-ᾱ_t)·ε（重参数化）。设计β_t使ᾱ_T≈0→x_T纯噪声。",
                hint="递推公式"
            ),
            Quiz(
                question="反向过程p_θ(x_{t-1}|x_t)建模为高斯。训练目标是什么？预测噪声还是x_0？",
                options=[
                    "预测x_0",
                    "预测噪声ε：min E||ε-ε_θ(x_t,t)||²，等价于去噪score matching",
                    "预测均值",
                    "随机"
                ],
                correct_answer=1,
                explanation="原始DDPM预测ε。x_t=√ᾱ_t·x_0+√(1-ᾱ_t)·ε，预测ε等价于预测x_0。训练：最小化噪声预测误差，等价于score matching∇log p。",
                hint="去噪目标"
            ),
            Quiz(
                question="DDPM采样需T步（如T=1000），DDIM如何加速到10-50步？",
                options=[
                    "不能加速",
                    "非马尔可夫过程，直接从x_t跳到x_{t-k}，保持边缘分布q(x_t|x_0)不变",
                    "降低质量",
                    "随机"
                ],
                correct_answer=1,
                explanation="DDPM是马尔可夫链，必须逐步。DDIM构造非马尔可夫过程：x_t→x_{t-k}直接跳跃，保持q(x_t|x_0)不变。采样变确定性，可插值。",
                hint="确定性采样"
            ),
            Quiz(
                question="为何扩散模型训练稳定？相比GAN的min_G max_D目标。",
                options=[
                    "GAN更好",
                    "扩散是单一最小化目标（似然），无min-max对抗；GAN是鞍点问题，难平衡G-D",
                    "相同",
                    "随机"
                ],
                correct_answer=1,
                explanation="GAN：min-max鞍点，G和D互相对抗，易mode collapse、训练不稳定。扩散：最大化ELBO的单一最小化，无对抗，稳定但慢（需多步采样）。",
                hint="优化目标"
            )
        ]
    
    @staticmethod
    def get_neural_geometry_quizzes() -> List[Quiz]:
        """神经几何相关测验"""
        return [
            Quiz(
                question="流形假设：数据x∈R^D分布在d维流形M上(d≪D)。神经网络如何利用？",
                options=[
                    "不利用",
                    "学习映射f:M→R^k展开流形，保持流形结构（如局部距离）到表示空间",
                    "随机映射",
                    "无关"
                ],
                correct_answer=1,
                explanation="高维数据实际自由度低（如人脸图像由姿态、光照等少数因子决定）。网络学习非线性降维f，保持语义邻近性。自编码器重构，监督学习分类。",
                hint="非线性降维"
            ),
            Quiz(
                question="Loss landscape的局部几何：Hessian H的特征值谱。Sharp vs Flat minima有何区别？",
                options=[
                    "无区别",
                    "Sharp：特征值大（曲率高）易过拟合；Flat：特征值小（平坦盆地）泛化好",
                    "Sharp更好",
                    "随机"
                ],
                correct_answer=1,
                explanation="Sharp：参数小扰动→loss大变化→对初始化/数据敏感。Flat：loss平坦盆地→扰动鲁棒→泛化好。大batch倾向Sharp，小batch噪声找Flat。",
                hint="泛化差距"
            ),
            Quiz(
                question="神经切线核(NTK)：无穷宽网络的极限行为。训练时参数几乎不动？",
                options=[
                    "完全不动",
                    "无穷宽时参数微动（lazy regime），网络行为近似线性（NTK固定）",
                    "正常移动",
                    "随机"
                ],
                correct_answer=1,
                explanation="宽度→∞：初始化权重O(1/√width)，训练后变化O(1/width)→相对不变。网络线性化在初始，预测f(x;θ)≈f(x;θ_0)+∇_θf·Δθ=线性模型。",
                hint="懒惰训练"
            ),
            Quiz(
                question="模式连通性：两个独立训练的网络θ_A和θ_B，线性插值θ_t=(1-t)θ_A+tθ_B。Loss曲线？",
                options=[
                    "单调",
                    "通常有barrier（中间loss高），但存在低loss的弯曲路径连接两者",
                    "线性",
                    "不连通"
                ],
                correct_answer=1,
                explanation="线性路径常有高loss barrier（不同初始化学到不同特征排列）。但loss landscape中存在低维弯曲路径连接。模式连通性理论研究此现象。",
                hint="多个最小值间的路径"
            )
        ]
    
    @staticmethod
    def get_information_geometry_quizzes() -> List[Quiz]:
        """信息几何相关测验"""
        return [
            Quiz(
                question="KL散度KL(p||q)=E_p[log(p/q)]为何不对称？什么时候等于0？",
                options=[
                    "对称",
                    "KL(p||q)≠KL(q||p)，非距离；当且仅当p=q a.e.时KL=0（吉布斯不等式）",
                    "总是0",
                    "随机"
                ],
                correct_answer=1,
                explanation="KL不对称：前向KL(p||q)惩罚q在p非零处为0（mode-seeking）；反向KL(q||p)惩罚q在p为0处非0（mean-seeking）。VI用反向KL。",
                hint="前向vs反向"
            ),
            Quiz(
                question="Fisher信息矩阵I_{ij}=E[∂log p/∂θ_i·∂log p/∂θ_j]。为何等于Hessian的期望？",
                options=[
                    "不等",
                    "E[∂²log p/∂θ_i∂θ_j]=-I_{ij}（score函数性质：E[∂log p]=0）",
                    "随机",
                    "定义"
                ],
                correct_answer=1,
                explanation="∂log p的期望=0（归一化条件）。对θ求导：E[∂²log p]=0。展开：∂²log p=-∂log p·∂log p+...→I=-E[Hessian(log p)]。",
                hint="score函数性质"
            ),
            Quiz(
                question="自然梯度θ←θ-ηI⁻¹∇f相比普通梯度有何优势？为什么？",
                options=[
                    "无区别",
                    "I⁻¹预条件，在参数空间的黎曼度量下最速下降（参数重整化不变）",
                    "更慢",
                    "随机"
                ],
                correct_answer=1,
                explanation="普通梯度：欧氏空间最速下降，参数重整化改变方向。自然梯度：Fisher度量下测地线，保持分布空间的方向。对病态问题（如NN）收敛更快。",
                hint="度量不变性"
            ),
            Quiz(
                question="KL散度与似然关系：min_θ KL(p_data||p_θ)等价于什么？",
                options=[
                    "最小化MSE",
                    "最大化似然MLE（因KL=H(p_data,p_θ)-H(p_data)，后者常数）",
                    "无关",
                    "随机"
                ],
                correct_answer=1,
                explanation="KL(p_data||p_θ)=E_data[log p_data/p_θ]=-E[log p_θ]+常数=负对数似然+常数。最小化KL=最大化似然。交叉熵训练的理论基础。",
                hint="MLE的信息论解释"
            )
        ]
    
    @staticmethod
    def get_scaling_laws_quizzes() -> List[Quiz]:
        """缩放定律相关测验"""
        return [
            Quiz(
                question="OpenAI缩放定律：L(N)=L_∞+(N_c/N)^α。为何是幂律而非指数？",
                options=[
                    "随机",
                    "幂律：log L线性依赖log N，长尾衰减平滑；指数衰减太快，不符合实证",
                    "指数更好",
                    "无意义"
                ],
                correct_answer=1,
                explanation="实证：log-log图线性→幂律L∝N^(-α)（α≈0.076）。幂律无特征尺度（scale-free），长尾平滑。指数L∝e^(-βN)衰减太快，不匹配数据。",
                hint="log-log线性"
            ),
            Quiz(
                question="Chinchilla最优：计算C固定，模型N和数据D如何分配？N∝D还是N∝D²？",
                options=[
                    "N=D",
                    "N∝D（约1:20 tokens/parameter），而非N固定增加D（GPT-3做法）",
                    "N∝D²",
                    "随机"
                ],
                correct_answer=1,
                explanation="L(N,D)≈A/N^α+B/D^β。固定C=6ND，最优：N∝D（α≈β时）。Chinchilla：70B参数用1.4T tokens，优于Gopher 280B+300B tokens。",
                hint="同步增长"
            ),
            Quiz(
                question="为何小模型在某些任务上超过大模型？迁移缩放定律如何解释？",
                options=[
                    "不可能",
                    "预训练Loss低≠下游任务好，迁移gap∝预训练数据与任务的分布距离",
                    "随机",
                    "小模型总更好"
                ],
                correct_answer=1,
                explanation="大模型在通用预训练数据上Loss低，但特定任务上可能过参数化或分布不匹配。小模型在目标分布上直接训练可能更优。需考虑迁移效率。",
                hint="分布匹配"
            ),
            Quiz(
                question="涌现能力(Emergence)：某能力在规模超过阈值后突然出现。为何不是平滑的？",
                options=[
                    "平滑",
                    "评估指标不连续（如准确率0/1），底层能力可能平滑增长但跨过阈值后体现",
                    "随机",
                    "不存在"
                ],
                correct_answer=1,
                explanation="如few-shot学习：能力连续增长，但任务成功需最低能力阈值→二值化。底层Loss平滑下降，但任务指标突变。度量选择影响涌现的观察。",
                hint="阈值效应"
            )
        ]
    
    @staticmethod
    def get_game_theory_quizzes() -> List[Quiz]:
        """博弈论相关测验"""
        return [
            Quiz(
                question="纳什均衡(s*,t*)满足：u_i(s*,t*)≥u_i(s_i,t*)对所有s_i。为何纯策略NE可能不存在？",
                options=[
                    "总是存在",
                    "如石头剪刀布，循环优势无纯策略NE；混合策略NE总存在（Nash定理）",
                    "随机的",
                    "无意义"
                ],
                correct_answer=1,
                explanation="纯策略NE：每人最优响应纯策略。石头剪刀布：任何纯策略组合都有人想改。混合策略NE：随机化使对手无利可图（均匀混合）。",
                hint="混合策略扩展"
            ),
            Quiz(
                question="囚徒困境：背叛是严格优势策略，但(背叛,背叛)帕累托劣于(合作,合作)。为何？",
                options=[
                    "不存在",
                    "个体最优≠集体最优：背叛dominant，但双背叛收益<双合作（社会困境）",
                    "相同",
                    "随机"
                ],
                correct_answer=1,
                explanation="收益矩阵：(合作,合作)→(3,3)，(背叛,背叛)→(1,1)。对每人背叛更优（dominant），但NE(1,1)帕累托劣于(3,3)。需重复博弈或惩罚机制。",
                hint="社会困境"
            ),
            Quiz(
                question="重复囚徒困境中，Tit-for-Tat策略为何有效？Folk定理说明什么？",
                options=[
                    "无效",
                    "TFT互惠：合作→合作，背叛→惩罚；Folk定理：折扣δ够大，任何可行收益均可维持",
                    "随机",
                    "只适用一次"
                ],
                correct_answer=1,
                explanation="TFT：首轮合作，之后模仿对手上轮。触发惩罚机制维持合作。Folk定理：无限重复+δ够大→多个NE，包括合作均衡（威胁惩罚背叛）。",
                hint="声誉机制"
            ),
            Quiz(
                question="零和博弈：u_1+u_2=0。minimax定理保证什么？与NE关系？",
                options=[
                    "无关",
                    "混合策略下max_x min_y u=min_y max_x u（鞍点），鞍点策略即NE",
                    "不存在",
                    "随机"
                ],
                correct_answer=1,
                explanation="零和：一方所得=另一方所失。minimax定理（von Neumann）：混合策略下最优防御=最优攻击。鞍点(x*,y*)是NE：双方无改进余地。",
                hint="对抗均衡"
            )
        ]
    
    @staticmethod
    def get_neuroevolution_quizzes() -> List[Quiz]:
        """神经进化相关测验"""
        return [
            Quiz(
                question="神经进化用进化算法（EA）优化神经网络。为何适合强化学习（RL）？",
                options=[
                    "更快",
                    "无需梯度，适合非可微奖励、稀疏奖励、长horizon（EA基于最终fitness）",
                    "需要更少数据",
                    "总找全局最优"
                ],
                correct_answer=1,
                explanation="RL中奖励常非可微（如游戏胜负）、稀疏（只在终局）。EA：评估整个episode的fitness，无需反向传播。但样本效率低（需多个rollout）。",
                hint="黑盒优化"
            ),
            Quiz(
                question="NEAT的创新式复杂化(complexification)：从简单拓扑开始进化。为何优于固定结构？",
                options=[
                    "无区别",
                    "避免过早复杂化和维度灾难，逐步增加连接/节点，自动搜索结构空间",
                    "总是更慢",
                    "随机"
                ],
                correct_answer=1,
                explanation="固定结构：人工设计或大规模搜索。NEAT：最小拓扑开始，变异增加连接/节点，选择压力自动平衡复杂度。历史标记(innovation number)对齐基因。",
                hint="增量进化"
            ),
            Quiz(
                question="ES(进化策略)优化θ：θ←θ+α·Σf_i·ε_i，ε_i~N(0,I)是扰动。为何梯度自由？",
                options=[
                    "需要梯度",
                    "有限差分近似梯度：∇E[f(θ+σε)]≈(1/σ)E[f(θ+σε)·ε]，黑盒评估f",
                    "随机优化",
                    "无理论"
                ],
                correct_answer=1,
                explanation="ES：θ加噪声ε，评估fitness f(θ+σε)，用f加权ε方向更新。数学上近似策略梯度（似然比）。可并行评估，但样本效率低。",
                hint="有限差分梯度"
            ),
            Quiz(
                question="为何大规模并行使ES在RL中复兴（如OpenAI ES）？相比A3C等。",
                options=[
                    "ES更准确",
                    "ES可扩展性好：千核并行评估fitness，通信量小；A3C需梯度同步",
                    "ES样本效率高",
                    "随机"
                ],
                correct_answer=1,
                explanation="A3C：异步梯度更新，需频繁通信参数/梯度。ES：worker独立评估(θ+ε_i)返回标量f_i，master聚合→通信O(1)。墙钟时间快但样本效率低。",
                hint="通信效率"
            )
        ]
    
    @staticmethod
    def get_probabilistic_programming_quizzes() -> List[Quiz]:
        """概率编程相关测验"""
        return [
            Quiz(
                question="贝叶斯推断p(θ|D)=p(D|θ)p(θ)/p(D)中，p(D)=∫p(D|θ)p(θ)dθ为何难算？",
                options=[
                    "简单",
                    "高维积分，θ空间大多数区域p(D|θ)≈0（典型集问题），需MCMC或VI",
                    "可解析",
                    "不需要"
                ],
                correct_answer=1,
                explanation="p(D)归一化常数需遍历参数空间积分。高维时绝大部分θ对D贡献可忽略，但难以找到高密度区（典型集）。MCMC采样或VI近似。",
                hint="归一化常数"
            ),
            Quiz(
                question="Metropolis-Hastings：提议θ'~q(θ'|θ)，以α=min(1,p(θ')/p(θ)·q(θ|θ')/q(θ'|θ))接受。为何平稳分布=p(θ)？",
                options=[
                    "不是",
                    "细致平衡：p(θ)q(θ'|θ)α(θ→θ')=p(θ')q(θ|θ')α(θ'→θ)，保证平稳",
                    "随机的",
                    "需要证明"
                ],
                correct_answer=1,
                explanation="细致平衡(detailed balance)：每对状态间的转移流相等。MH的接受率α恰好满足此条件，保证马尔可夫链收敛到目标分布p(θ)。",
                hint="平稳条件"
            ),
            Quiz(
                question="变分推断(VI)：用简单分布q(θ)近似后验p(θ|D)。优化目标ELBO是什么？",
                options=[
                    "最大似然",
                    "min KL(q||p) = max ELBO = E_q[log p(D,θ)] + H(q)（证据下界）",
                    "最小误差",
                    "随机"
                ],
                correct_answer=1,
                explanation="log p(D)=ELBO+KL(q||p)，ELBO≤log p(D)。最大化ELBO↔最小化KL。ELBO=似然期望+熵，可优化（无需p(D)）。",
                hint="KL散度"
            ),
            Quiz(
                question="为何VI比MCMC快但可能有偏？渐近性质对比。",
                options=[
                    "VI总是更好",
                    "VI：优化→快，但q族限制可能无法精确逼近p；MCMC：渐近无偏但需burn-in",
                    "MCMC总是更好",
                    "相同"
                ],
                correct_answer=1,
                explanation="VI：q族（如均值场）可能表达能力不足，欠拟合后验。但优化快，确定性。MCMC：理论上渐近精确，但需足够样本，计算慢。实践：VI探索，MCMC精调。",
                hint="近似质量vs速度"
            )
        ]
    
    @staticmethod
    def get_dimensions_parameters_quizzes() -> List[Quiz]:
        """维度与参数相关测验"""
        return [
            Quiz(
                question="全连接层参数量W·d_in·d_out。Transformer中d_model=512，FFN扩展4倍，两层参数量是多少？",
                options=[
                    "512²",
                    "2·512·(4·512)=2M（上升512→2048→512，参数主要在这里）",
                    "4·512²",
                    "512·2048"
                ],
                correct_answer=1,
                explanation="FFN：512→2048（512·2048参数），2048→512（2048·512参数）。总计≈2M参数。大部分Transformer参数在FFN而非注意力。",
                hint="两次线性变换"
            ),
            Quiz(
                question="CNN参数量k²·C_in·C_out，但FLOPs=k²·C_in·C_out·H·W。为何FLOPs高得多？",
                options=[
                    "计算错误",
                    "参数共享：k²个参数在H·W个位置重复使用，FLOPs=参数量·空间维度",
                    "FLOPs更少",
                    "无关系"
                ],
                correct_answer=1,
                explanation="每个输出位置(H·W个)都要做k²·C_in·C_out次乘加。参数共享使得参数量小但计算量大。224²图像：FLOPs=参数×50k。",
                hint="参数共享的代价"
            ),
            Quiz(
                question="嵌入层vocab_size·d_model参数量。GPT-2：vocab=50k, d=768，嵌入层占比多大？",
                options=[
                    "很小",
                    "约38M参数，占总117M的~32%（词表越大占比越高）",
                    "占比<5%",
                    "占比>80%"
                ],
                correct_answer=1,
                explanation="嵌入：50k·768=38M。GPT-2-small总参数117M，嵌入占32%。大模型（GPT-3）嵌入占比降低因为层数深。词表大小是关键瓶颈。",
                hint="词表规模影响"
            ),
            Quiz(
                question="为何大模型参数/FLOPs/内存增长速度不同？训练GPT需要多少内存相对于参数量？",
                options=[
                    "内存=参数量",
                    "内存≈4×参数（模型2份+梯度+优化器状态），混合精度可减半",
                    "内存≈参数量/2",
                    "内存=FLOPs"
                ],
                correct_answer=1,
                explanation="内存：参数(fp32)、梯度、Adam状态(m,v)→4x参数。混合精度(fp16训练)→2x。激活值与batch·seq_len成正比，可用梯度检查点优化。",
                hint="优化器状态"
            )
        ]
    
    @staticmethod
    def get_hilbert_space_quizzes() -> List[Quiz]:
        """希尔伯特空间相关测验"""
        return [
            Quiz(
                question="希尔伯特空间是完备的内积空间。完备性在机器学习中为何重要？",
                options=[
                    "没有用处",
                    "保证柯西序列收敛，优化算法的极限点存在于空间内（闭性）",
                    "只是定义",
                    "加速计算"
                ],
                correct_answer=1,
                explanation="完备性：空间中柯西序列的极限仍在空间内。梯度下降产生参数序列{w_t}，完备性保证收敛点存在。非完备空间可能收敛到边界外。",
                hint="柯西序列"
            ),
            Quiz(
                question="再生核希尔伯特空间(RKHS)：内积⟨f,K(x,·)⟩=f(x)。这个性质有什么用？",
                options=[
                    "无实际用处",
                    "评估f(x)只需内积，核技巧无需显式φ(x)（representer定理基础）",
                    "只是理论",
                    "加速训练"
                ],
                correct_answer=1,
                explanation="再生性：点评估=内积。使得无限维RKHS中的函数可通过有限样本表示：f(x)=Σα_iK(x_i,x)。SVM、核岭回归的理论基础。",
                hint="representer定理"
            ),
            Quiz(
                question="L²空间：||f||²=∫|f(x)|²dx<∞。神经网络输出f_θ何时不在L²？为什么关心？",
                options=[
                    "总在L²",
                    "无界激活（如ReLU）无L2正则时可能||f||→∞，泛化界需||f||有界",
                    "从不在L²",
                    "无关紧要"
                ],
                correct_answer=1,
                explanation="L²空间=平方可积函数。泛化误差界常假设||f||<∞（Rademacher复杂度）。ReLU网络无约束时可能爆炸。批归一化/权重衰减隐式限制范数。",
                hint="泛化理论"
            ),
            Quiz(
                question="Fourier基{e^(inx)}是L²[-π,π]的正交基。为何深度学习少用Fourier，多用神经网络？",
                options=[
                    "Fourier总是更好",
                    "Fourier基全局，不适合局部模式；神经网络(如ReLU)提供自适应局部基",
                    "计算太慢",
                    "无区别"
                ],
                correct_answer=1,
                explanation="Fourier：全局正弦波，表示局部特征需大量高频分量。神经网络：自适应学习局部特征检测器，稀疏表示。小波/Gabor介于两者，CNN学到类似特征。",
                hint="局部vs全局"
            )
        ]
    
    @staticmethod
    def get_multimodal_geometry_quizzes() -> List[Quiz]:
        """多模态几何相关测验"""
        return [
            Quiz(
                question="CLIP对比学习：max⟨f_I(img),f_T(text)⟩。为何余弦相似度比欧氏距离更好？",
                options=[
                    "欧氏距离更好",
                    "余弦度量方向不变于尺度，高维嵌入方向比长度更重要（归一化后内积）",
                    "计算更快",
                    "没有区别"
                ],
                correct_answer=1,
                explanation="高维空间中，归一化后向量聚集在超球面。余弦=归一化内积度量角度，不受范数影响。欧氏距离||x-y||²=2-2cos(θ)在单位球面也退化为余弦。",
                hint="超球面几何"
            ),
            Quiz(
                question="多模态对齐：图像→R^d_I，文本→R^d_T，如何映射到共同空间？线性投影够吗？",
                options=[
                    "不可能对齐",
                    "非线性映射学习流形对齐（如CNN+Transformer），线性仅在低维/简单情况",
                    "线性总是够",
                    "随机映射"
                ],
                correct_answer=1,
                explanation="图像流形与文本流形结构不同。线性映射：保持全局拓扑，可能无法对齐局部流形。深度网络：非线性扭曲空间，学习复杂对应关系（如Procrustes分析）。",
                hint="流形对齐"
            ),
            Quiz(
                question="对比损失：L=-log exp(s⁺)/Σexp(s_i)。batch_size=N时，负样本数是多少？为何大batch关键？",
                options=[
                    "N个负样本",
                    "N-1个负样本，大batch提供更多难负样本，改善嵌入空间结构",
                    "1个负样本",
                    "batch size无关"
                ],
                correct_answer=1,
                explanation="InfoNCE：1个正样本vs N-1个负样本。batch越大，负样本越多，梯度估计越准。CLIP用32k batch。负样本充当锚点，推开不相关样本。",
                hint="负采样规模"
            ),
            Quiz(
                question="模态间隙(modality gap)：图像嵌入和文本嵌入形成分离的锥形区域。如何度量？",
                options=[
                    "没有间隙",
                    "计算模态内vs模态间平均距离，温度系数τ调节间隙大小",
                    "无法度量",
                    "间隙总是0"
                ],
                correct_answer=1,
                explanation="模态内距离<模态间距离→形成锥形。温度τ：小τ→sharp分布，大间隙；大τ→平滑分布，小间隙。度量：avg_dist(I,I') vs avg_dist(I,T)。",
                hint="锥形结构"
            )
        ]
    
    @staticmethod
    def get_optimal_transport_quizzes() -> List[Quiz]:
        """最优传输相关测验"""
        return [
            Quiz(
                question="Wasserstein距离W(P,Q)=inf_γ E_{(x,y)~γ}[c(x,y)]。为何比KL散度更好用于GAN？",
                options=[
                    "KL更好",
                    "Wasserstein考虑几何距离，不重叠分布仍有有意义梯度；KL要求支撑集重叠",
                    "计算更快",
                    "没有区别"
                ],
                correct_answer=1,
                explanation="KL(P||Q)：若P和Q支撑不重叠→KL=∞，梯度消失。Wasserstein：度量移动概率质量的最小代价，即使不重叠也连续可导。WGAN利用此性质。",
                hint="支撑集不重叠"
            ),
            Quiz(
                question="Kantorovich对偶：W(P,Q)=sup_{||f||_L≤1} E_P[f]-E_Q[f]。为何转为对偶形式？",
                options=[
                    "无优势",
                    "原问题求传输矩阵O(n²)变量，对偶求1-Lipschitz函数f_θ（神经网络参数化）",
                    "对偶更慢",
                    "等价且同样难"
                ],
                correct_answer=1,
                explanation="原问题：离散化后n²个γ_{ij}。对偶：求Lipschitz函数f（用神经网络+梯度惩罚实现）。WGAN-GP：max E_x[f(x)]-E_z[f(G(z))] s.t. ||∇f||≈1。",
                hint="WGAN"
            ),
            Quiz(
                question="Sinkhorn算法：W_ε(P,Q)=min_γ ⟨γ,C⟩-εH(γ)。为何加熵正则项ε？",
                options=[
                    "无作用",
                    "熵正则使问题严格凸，可用快速迭代算法O(n²logn)而非线性规划O(n³)",
                    "只是近似",
                    "降低准确度"
                ],
                correct_answer=1,
                explanation="无正则：线性规划O(n³)。熵正则：解变为γ=diag(u)·exp(-C/ε)·diag(v)，Sinkhorn迭代交替归一化u,v→O(n²)每步。ε→0退化到精确OT。",
                hint="熵正则化"
            ),
            Quiz(
                question="Gromov-Wasserstein距离：比较不同度量空间(X,d_X)和(Y,d_Y)上的分布。与标准OT区别？",
                options=[
                    "完全相同",
                    "GW比较内部几何结构d_X和d_Y，无需点对应；标准OT需要共同空间",
                    "GW更简单",
                    "GW无法计算"
                ],
                correct_answer=1,
                explanation="标准OT：X和Y在同一空间，比较P(x)和Q(y)。GW：比较(X,d_X)和(Y,d_Y)的形状，匹配d_X(x,x')≈d_Y(y,y')。用于图匹配、无对应的分布比较。",
                hint="几何匹配"
            )
        ]
    
    @staticmethod
    def get_training_dynamics_quizzes() -> List[Quiz]:
        """训练动力学相关测验"""
        return [
            Quiz(
                question="神经正切核(NTK)：在宽度n→∞时，网络训练等价于线性模型。为何实际深度网络非线性？",
                options=[
                    "NTK错误",
                    "有限宽度+特征学习使NTK动态变化，逃离lazy regime进入feature learning",
                    "总是线性",
                    "无关"
                ],
                correct_answer=1,
                explanation="无限宽：参数微动，NTK冻结→核回归（lazy训练）。有限宽：特征表示显著变化，NTK演化→非线性特征学习。深度学习的力量在feature learning。",
                hint="lazy vs rich"
            ),
            Quiz(
                question="Loss landscape：为何过参数化网络（m≫n）几乎无bad局部最小值？",
                options=[
                    "有很多局部最小值",
                    "高维下：鞍点指数多于局部最小值；过参数化使所有局部最小值近似全局最优",
                    "随机的",
                    "无局部最小值"
                ],
                correct_answer=1,
                explanation="维度诅咒的好处：d维中局部最小需d个正特征值（概率2^(-d)）。鞍点更常见。过参数化：多条路径到最优，loss landscape更平坦连通。",
                hint="高维几何"
            ),
            Quiz(
                question="双下降曲线：测试误差随模型复杂度先降后升再降。第二次下降为何发生？",
                options=[
                    "不会发生",
                    "插值阈值后，过参数化选择最小范数解（隐式正则化），泛化改善",
                    "过拟合",
                    "数据问题"
                ],
                correct_answer=1,
                explanation="经典：bias-variance权衡→U型。现代：过参数化m>n时，插值解非唯一，SGD隐式选择最小范数→第二次下降。关键：插值+隐式正则。",
                hint="隐式正则化"
            ),
            Quiz(
                question="临界学习期(critical period)：训练初期的噪声影响终生。为什么早期最敏感？",
                options=[
                    "早期不重要",
                    "早期确定特征层次结构，后期难以改变（表示固化）；类似大脑发育关键期",
                    "任何时期相同",
                    "随机现象"
                ],
                correct_answer=1,
                explanation="早期训练：快速形成粗粒度特征检测器，决定后续学习路径。噪声/数据质量影响特征方向。后期：微调已有表示，难以重构底层特征。迁移学习利用此性质。",
                hint="特征层次"
            )
        ]
    
    @staticmethod
    def get_vcdim_derivation_quizzes() -> List[Quiz]:
        """VC维推导相关测验"""
        return [
            Quiz(
                question="Sauer引理：d个VC维的假设类H，在n点上最多打散S(n,d)=Σ_{i=0}^d C(n,i)种。当n>d时？",
                options=[
                    "2^n种",
                    "S(n,d)≤(en/d)^d，多项式级而非指数级（组合学约束）",
                    "n^d种",
                    "无限"
                ],
                correct_answer=1,
                explanation="虽然总共2^n种标记，VC维d限制表达能力。Sauer引理：最多(en/d)^d≪2^n。这是PAC可学习的关键：可行标记数多项式→样本复杂度多项式。",
                hint="组合约束"
            ),
            Quiz(
                question="PAC学习：以1-δ概率，R(h)≤R̂(h)+ε。需要多少样本n？",
                options=[
                    "O(n)",
                    "O((d/ε²)·log(1/δ))，VC维d和精度ε²反比，置信度log(1/δ)线性",
                    "O(d)",
                    "无穷"
                ],
                correct_answer=1,
                explanation="Hoeffding+Union Bound+Sauer：n≥(8d/ε²)ln(4/δ)保证一致收敛。样本复杂度与d线性、ε²反比。PAC=Probably Approximately Correct。",
                hint="样本复杂度"
            ),
            Quiz(
                question="结构风险最小化(SRM)：min{R̂(h)+Ω(h)}，Ω与VC维相关。为何比ERM好？",
                options=[
                    "无区别",
                    "ERM只最小化训练误差易过拟合；SRM平衡拟合+复杂度（Ω∝√(d/n)）",
                    "ERM更好",
                    "随机"
                ],
                correct_answer=1,
                explanation="ERM：min R̂(h)→选最复杂h（VC维大）→过拟合。SRM：加入复杂度惩罚Ω(h)∝√(d_h/n)→自动平衡。等价于正则化。",
                hint="奥卡姆剃刀"
            ),
            Quiz(
                question="No Free Lunch定理：所有算法平均性能相同。为何实践中有优劣？",
                options=[
                    "定理错误",
                    "NFL对所有可能问题平均；实际问题有结构（如平滑性），算法利用先验假设",
                    "随机",
                    "无意义"
                ],
                correct_answer=1,
                explanation="NFL：对所有2^{2^n}个目标函数平均，任意算法等价随机搜索。但实际问题非均匀：有平滑、稀疏等结构。算法编码归纳偏置（inductive bias）利用结构。",
                hint="归纳偏置"
            )
        ]
