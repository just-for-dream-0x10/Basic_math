"""
学习路径UI组件
提供可视化的学习路径展示和进度跟踪
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from learning_paths import LEARNING_PATHS, RECOMMENDED_ORDER, get_next_module


def render_learning_paths():
    """渲染学习路径选择页面"""
    from learning_paths import get_learning_stats, check_achievements
    
    st.title("🎓 学习路径指南")
    
    st.markdown("""
    欢迎来到机器学习数学之旅！根据你的背景和目标，我们为你准备了不同的学习路径。
    
    选择一个路径，我们将引导你循序渐进地掌握机器学习的数学基础。
    """)
    
    # 初始化session state
    if 'completed_modules' not in st.session_state:
        st.session_state['completed_modules'] = set()
    
    # 如果有学习进度，显示整体统计
    if st.session_state['completed_modules']:
        stats = get_learning_stats(st.session_state['completed_modules'])
        achievements = check_achievements(st.session_state['completed_modules'])
        
        st.markdown("## 📊 你的学习概览")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("完成模块", f"{stats['completed_count']}/{stats['total_modules']}")
        with col2:
            st.metric("完成率", f"{stats['completion_rate']:.1f}%")
        with col3:
            st.metric("掌握概念", f"{stats['concepts_count']} 个")
        with col4:
            st.metric("解锁成就", f"{len(achievements)} 个")
        
        # 显示学习进度可视化
        if stats['completion_rate'] > 0:
            render_learning_progress_chart(st.session_state['completed_modules'])
        
        # 成就展示
        if achievements:
            with st.expander("🏆 查看已解锁成就", expanded=False):
                cols = st.columns(min(3, len(achievements)))
                for i, ach in enumerate(achievements):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='padding: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); 
                                   border-radius: 10px; text-align: center; margin: 0.5rem 0;'>
                            <h3 style='margin: 0; font-size: 2rem;'>{ach['name']}</h3>
                            <p style='margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;'>{ach['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # 路径选择
    st.markdown("## 📚 选择你的学习路径")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        path_options = {
            "🌱 初学者路径": "beginner",
            "🧠 深度学习路径": "deep_learning",
            "📚 理论深度路径": "theory",
            "⚙️ 工程实践路径": "practitioner",
            "🎯 自定义路径": "custom"
        }
        
        selected_path_name = st.radio(
            "选择路径",
            list(path_options.keys()),
            help="根据你的背景和学习目标选择"
        )
        
        selected_path = path_options[selected_path_name]
    
    with col2:
        path_info = LEARNING_PATHS[selected_path]
        
        st.markdown(f"### {path_info['name']}")
        st.markdown(f"**描述**: {path_info['description']}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("难度", path_info['difficulty'])
        with col_b:
            st.metric("预计时长", path_info['duration'])
    
    st.markdown("---")
    
    # 显示路径详情
    if selected_path != "custom":
        render_path_details(selected_path)
    else:
        render_custom_path()


def render_path_details(path_key):
    """渲染路径详细信息"""
    from learning_paths import get_difficulty_score
    from config import MODULES
    
    path = LEARNING_PATHS[path_key]
    modules = path["modules"]
    
    # 初始化session state
    if 'completed_modules' not in st.session_state:
        st.session_state['completed_modules'] = set()
    
    # 路径统计
    completed_in_path = sum(1 for m in modules if m['module'] in st.session_state['completed_modules'])
    path_progress = (completed_in_path / len(modules) * 100) if modules else 0
    
    st.markdown("## 🗺️ 学习地图")
    
    # 显示路径进度条
    st.progress(path_progress / 100, text=f"路径完成度: {completed_in_path}/{len(modules)} ({path_progress:.1f}%)")
    
    # 创建流程图
    fig = create_pathway_flowchart(modules, st.session_state['completed_modules'])
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加"开始这个路径"按钮
    if st.session_state.get('current_path') != path_key:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 开始这个学习路径", key=f"start_path_{path_key}", use_container_width=True):
                st.session_state['current_path'] = path_key
                st.success(f"✅ 已选择：{path['name']}")
                st.rerun()
    
    st.markdown("---")
    
    # 详细模块列表
    st.markdown("## 📖 模块详情")
    
    for i, module in enumerate(modules, 1):
        module_key = module['module']
        is_completed = module_key in st.session_state['completed_modules']
        can_start = all(p in st.session_state['completed_modules'] for p in module.get('prerequisites', []))
        
        # 状态标识
        if is_completed:
            status_icon = "✅"
            status_color = "#22c55e"
        elif can_start:
            status_icon = "🔓"
            status_color = "#3b82f6"
        else:
            status_icon = "🔒"
            status_color = "#94a3b8"
        
        with st.expander(f"{status_icon} 第 {i} 步：{module['title']} ({module['time']})", expanded=(not is_completed and can_start)):
            col_info, col_action = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"**为什么学这个？** {module['why']}")
                
                # 先修要求
                if module['prerequisites']:
                    prereq_status = []
                    for prereq in module['prerequisites']:
                        prereq_info = next((m for m in modules if m['module'] == prereq), None)
                        if prereq_info:
                            prereq_name = prereq_info['title']
                            is_prereq_done = prereq in st.session_state['completed_modules']
                            prereq_status.append(f"{'✅' if is_prereq_done else '❌'} {prereq_name}")
                    st.markdown(f"**先修要求**: {', '.join(prereq_status)}")
                else:
                    st.markdown("**先修要求**: ✅ 无（可以直接学习）")
                
                # 核心概念标签
                st.markdown("**核心概念**:")
                concepts_html = " ".join([f"<span style='background: {status_color}20; color: {status_color}; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.85rem; margin: 0.2rem;'>{c}</span>" for c in module['key_concepts']])
                st.markdown(concepts_html, unsafe_allow_html=True)
                
                st.markdown(f"**推荐场景**: {', '.join(module['scenes'])}")
                
                # 难度显示
                difficulty = get_difficulty_score(module_key)
                st.markdown(f"**难度等级**: {'⭐' * difficulty}")
            
            with col_action:
                if is_completed:
                    st.success("已完成")
                    if st.button("🔄 复习", key=f"review_{module_key}_{i}"):
                        st.session_state['selected_module'] = module_key
                        st.rerun()
                elif can_start:
                    if st.button("🎯 开始学习", key=f"start_{module_key}_{i}", use_container_width=True):
                        st.session_state['selected_module'] = module_key
                        st.session_state['current_path'] = path_key
                        st.rerun()
                else:
                    st.warning("🔒 需要完成先修模块")
    
    st.markdown("---")
    
    # 学习建议
    col_tips, col_quiz = st.columns([1, 1])
    
    with col_tips:
        st.markdown("## 💡 学习建议")
        
        tips = [
            "📝 **做笔记**: 记录关键公式和理解",
            "🔄 **多次练习**: 调整参数，观察变化",
            "🤔 **思考为什么**: 不要只看现象，要理解原理",
            "🔗 **建立联系**: 思考不同概念之间的关系",
            "💻 **动手实践**: 在实际项目中应用所学知识"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")
    
    with col_quiz:
        st.markdown("## 🎯 快速评估")
        
        if completed_in_path > 0:
            st.markdown("测试你对已学内容的掌握程度：")
            
            if st.button("📝 开始测验", key=f"quiz_{path_key}"):
                render_quick_quiz(path_key, modules)
        else:
            st.info("完成一些模块后，这里会提供快速测验来检验你的理解。")


def render_custom_path():
    """渲染自定义路径"""
    st.markdown("## 🎯 创建你的自定义学习路径")
    
    st.markdown("""
    选择你感兴趣的模块，我们会根据依赖关系为你排序。
    """)
    
    from config import MODULES
    
    # 模块选择
    selected_modules = st.multiselect(
        "选择你想学习的模块",
        list(MODULES.keys()),
        format_func=lambda x: f"{MODULES[x]['name']} - {MODULES[x]['description']}"
    )
    
    if selected_modules:
        # 显示推荐顺序
        st.markdown("### 📊 推荐学习顺序")
        
        ordered_modules = []
        for module in RECOMMENDED_ORDER:
            if module in selected_modules:
                ordered_modules.append(module)
        
        # 添加不在推荐列表中的模块
        for module in selected_modules:
            if module not in ordered_modules:
                ordered_modules.append(module)
        
        for i, module_key in enumerate(ordered_modules, 1):
            module_info = MODULES[module_key]
            st.markdown(f"{i}. **{module_info['name']}** - {module_info['description']}")
        
        from learning_paths import estimate_completion_time
        total_time = estimate_completion_time(ordered_modules)
        st.info(f"预计完成时间: {total_time}")


def create_pathway_flowchart(modules, completed_modules=None):
    """创建学习路径流程图"""
    if completed_modules is None:
        completed_modules = set()
    
    # 使用Plotly创建流程图
    fig = go.Figure()
    
    n = len(modules)
    
    # 计算位置（垂直布局）
    for i, module in enumerate(modules):
        y_pos = n - i - 1
        module_key = module['module']
        is_completed = module_key in completed_modules
        
        # 根据完成状态选择颜色
        if is_completed:
            node_color = '#22c55e'
            line_color = '#16a34a'
            icon = '✓'
        else:
            # 检查是否可以开始（先修条件满足）
            can_start = all(p in completed_modules for p in module.get('prerequisites', []))
            if can_start:
                node_color = '#3b82f6'
                line_color = '#2563eb'
                icon = '○'
            else:
                node_color = '#94a3b8'
                line_color = '#64748b'
                icon = '○'
        
        # 添加节点
        fig.add_trace(go.Scatter(
            x=[0],
            y=[y_pos],
            mode='markers+text',
            marker=dict(size=50, color=node_color, line=dict(color=line_color, width=2)),
            text=f"{icon} {i+1}. {module['title']}",
            textposition='middle right',
            textfont=dict(size=11, color='white'),
            hovertext=f"<b>{module['title']}</b><br>时长: {module['time']}<br>概念: {', '.join(module['key_concepts'][:3])}<br>状态: {'✅ 已完成' if is_completed else ('🔓 可学习' if can_start else '🔒 需要先修')}",
            hoverinfo='text',
            showlegend=False
        ))
        
        # 添加连接线
        if i < n - 1:
            next_module = modules[i + 1]
            next_completed = next_module['module'] in completed_modules
            
            # 如果当前和下一个都完成了，用绿色连接线
            if is_completed and next_completed:
                line_color_conn = '#22c55e'
            elif is_completed:
                line_color_conn = '#3b82f6'
            else:
                line_color_conn = '#94a3b8'
            
            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[y_pos, y_pos - 1],
                mode='lines',
                line=dict(color=line_color_conn, width=2),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # 添加箭头
            fig.add_annotation(
                x=0,
                y=y_pos - 0.5,
                ax=0,
                ay=y_pos - 0.4,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=line_color_conn
            )
    
    fig.update_layout(
        title="学习路径流程图",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 4]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=max(400, n * 80),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='closest',
        plot_bgcolor='rgba(15, 20, 25, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def show_learning_progress(current_module_key=None):
    """显示学习进度和下一步推荐"""
    from learning_paths import recommend_next_modules, get_learning_stats, check_achievements
    from config import MODULES
    
    # 初始化session state
    if 'completed_modules' not in st.session_state:
        st.session_state['completed_modules'] = set()
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None
    
    # 获取学习统计
    stats = get_learning_stats(st.session_state['completed_modules'])
    
    # 如果有当前路径，显示进度
    if st.session_state.get('current_path'):
        path = LEARNING_PATHS.get(st.session_state['current_path'], LEARNING_PATHS['beginner'])
        path_modules = path.get('modules', [])
        total = len(path_modules)
        completed = len(st.session_state['completed_modules'])
        
        progress = completed / total if total > 0 else 0
        
        st.sidebar.progress(progress, text=f"进度: {completed}/{total}")
        
        # 显示当前路径名称
        st.sidebar.markdown(f"**当前路径**: {path['name']}")
        
        # 显示统计信息（折叠）
        with st.sidebar.expander("📊 学习统计", expanded=False):
            st.metric("完成率", f"{stats['completion_rate']:.1f}%")
            st.metric("已学概念", f"{stats['concepts_count']} 个")
            st.metric("累计学时", stats['total_time'])
            
            # 成就展示
            achievements = check_achievements(st.session_state['completed_modules'])
            if achievements:
                st.markdown("**🏆 已解锁成就:**")
                for ach in achievements:
                    st.markdown(f"- {ach['name']}")
        
        # 当前模块
        if current_module_key:
            if current_module_key in MODULES:
                current_name = MODULES[current_module_key]['name']
                
                # 检查是否在路径中
                current_in_path = any(m['module'] == current_module_key for m in path_modules)
                
                if current_in_path:
                    # 找到当前模块在路径中的位置
                    current_idx = None
                    for idx, m in enumerate(path_modules):
                        if m['module'] == current_module_key:
                            current_idx = idx
                            break
                    
                    if current_idx is not None:
                        st.sidebar.info(f"📍 当前: 第{current_idx+1}步 - {current_name}")
                        
                        # 标记完成按钮
                        if current_module_key not in st.session_state['completed_modules']:
                            if st.sidebar.button("✅ 标记为已完成", key="complete_module"):
                                st.session_state['completed_modules'].add(current_module_key)
                                
                                # 检查是否解锁新成就
                                new_achievements = check_achievements(st.session_state['completed_modules'])
                                if len(new_achievements) > len(achievements):
                                    st.sidebar.balloons()
                                
                                st.rerun()
                        else:
                            st.sidebar.success(f"✅ 已完成")
                        
                        # 显示下一步
                        if current_idx < len(path_modules) - 1:
                            next_module = path_modules[current_idx + 1]
                            st.sidebar.markdown(f"**下一步**: {next_module['title']}")
                            
                            if st.sidebar.button(f"➡️ 开始：{next_module['title']}", key="next_module"):
                                st.session_state['selected_module'] = next_module['module']
                                st.rerun()
                        else:
                            st.sidebar.success("🎉 恭喜完成当前路径的所有模块！")
                            
                            if st.sidebar.button("🔄 重新选择路径"):
                                st.session_state['current_path'] = None
                                st.rerun()
    else:
        # 没有选择路径，显示智能推荐
        st.sidebar.info("💡 点击上方'🎓 学习路径'选择一个学习路径")
        
        # 如果已经学了一些模块，提供智能推荐
        if st.session_state['completed_modules']:
            recommendations = recommend_next_modules(
                st.session_state['completed_modules'], 
                st.session_state.get('current_path')
            )
            
            if recommendations:
                with st.sidebar.expander("🎯 智能推荐", expanded=True):
                    st.markdown("**基于你的学习历史推荐:**")
                    for i, rec in enumerate(recommendations[:3], 1):
                        st.markdown(f"""
                        **{i}. {rec['title']}**  
                        {rec['reason']}  
                        难度: {'⭐' * rec['difficulty']} | 时长: {rec['time']}
                        """)
                        if st.button(f"开始学习", key=f"rec_{rec['module']}"):
                            st.session_state['selected_module'] = rec['module']
                            st.rerun()


def render_quick_quiz(path_key, modules):
    """渲染快速测验"""
    st.markdown("---")
    st.markdown("## 📝 快速知识测验")
    
    # 初始化测验状态
    if 'quiz_started' not in st.session_state:
        st.session_state['quiz_started'] = False
    if 'quiz_answers' not in st.session_state:
        st.session_state['quiz_answers'] = {}
    
    # 测验题库（基于不同模块）
    quiz_questions = {
        "matrix": [
            {
                "question": "矩阵乘法的几何意义是什么？",
                "options": ["向量旋转", "线性变换", "数值相乘", "矩阵相加"],
                "correct": 1,
                "explanation": "矩阵乘法表示线性变换，可以实现旋转、缩放、投影等几何操作。"
            },
            {
                "question": "特征向量在矩阵变换下的特点是？",
                "options": ["方向改变", "长度不变", "方向不变", "消失"],
                "correct": 2,
                "explanation": "特征向量在矩阵变换下保持方向不变，只是长度被缩放（缩放系数就是特征值）。"
            }
        ],
        "calculus": [
            {
                "question": "梯度的方向指向什么？",
                "options": ["函数下降最快的方向", "函数上升最快的方向", "函数不变的方向", "随机方向"],
                "correct": 1,
                "explanation": "梯度指向函数增长最快的方向，所以梯度下降要沿着负梯度方向。"
            },
            {
                "question": "链式法则在神经网络中的作用是？",
                "options": ["前向传播", "反向传播", "权重初始化", "激活函数"],
                "correct": 1,
                "explanation": "链式法则是反向传播算法的数学基础，用于计算复合函数的梯度。"
            }
        ],
        "loss": [
            {
                "question": "交叉熵损失函数主要用于什么任务？",
                "options": ["回归", "分类", "聚类", "降维"],
                "correct": 1,
                "explanation": "交叉熵是分类任务的标准损失函数，衡量预测概率分布与真实分布的差异。"
            },
            {
                "question": "学习率过大会导致什么问题？",
                "options": ["收敛太慢", "无法收敛/震荡", "过拟合", "欠拟合"],
                "correct": 1,
                "explanation": "学习率过大会导致参数更新步长过大，可能跳过最优点或在最优点附近震荡。"
            }
        ],
        "optimizer": [
            {
                "question": "Adam优化器的主要优势是什么？",
                "options": ["速度快", "自适应学习率", "内存少", "不需要调参"],
                "correct": 1,
                "explanation": "Adam结合了动量和自适应学习率，为每个参数维护独立的学习率。"
            },
            {
                "question": "动量(Momentum)解决了什么问题？",
                "options": ["收敛速度慢", "参数震荡", "过拟合", "梯度消失"],
                "correct": 1,
                "explanation": "动量通过累积历史梯度信息，减少参数更新的震荡，加速收敛。"
            }
        ],
        "regularization": [
            {
                "question": "L1正则化会产生什么效果？",
                "options": ["参数平滑", "稀疏解", "快速收敛", "防止震荡"],
                "correct": 1,
                "explanation": "L1正则化倾向于产生稀疏解，将一些权重压缩到0，实现特征选择。"
            },
            {
                "question": "正则化的主要目的是什么？",
                "options": ["加速训练", "防止过拟合", "提高精度", "减少计算量"],
                "correct": 1,
                "explanation": "正则化通过约束模型复杂度来防止过拟合，提高模型的泛化能力。"
            }
        ]
    }
    
    # 获取已完成模块的问题
    completed_modules = st.session_state.get('completed_modules', set())
    available_questions = []
    for module in modules:
        if module['module'] in completed_modules and module['module'] in quiz_questions:
            available_questions.extend(quiz_questions[module['module']])
    
    if not available_questions:
        st.info("还没有可用的测验题目。完成更多模块后再来测试吧！")
        return
    
    # 显示测验
    st.markdown(f"📚 基于你已完成的 **{len(completed_modules)}** 个模块生成了 **{len(available_questions)}** 道题目")
    
    if not st.session_state['quiz_started']:
        if st.button("🚀 开始测验", key="start_quiz_btn"):
            st.session_state['quiz_started'] = True
            st.session_state['quiz_answers'] = {}
            st.rerun()
        return
    
    # 显示题目
    st.markdown("---")
    for i, q in enumerate(available_questions):
        st.markdown(f"### 问题 {i+1}: {q['question']}")
        
        answer = st.radio(
            "选择你的答案：",
            options=q['options'],
            key=f"quiz_q{i}",
            index=st.session_state['quiz_answers'].get(i, None)
        )
        
        # 保存答案
        st.session_state['quiz_answers'][i] = q['options'].index(answer)
        
        st.markdown("---")
    
    # 提交按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📊 查看结果", key="submit_quiz", use_container_width=True):
            # 计算得分
            correct_count = 0
            for i, q in enumerate(available_questions):
                if st.session_state['quiz_answers'].get(i) == q['correct']:
                    correct_count += 1
            
            score = (correct_count / len(available_questions)) * 100
            
            # 显示结果
            st.markdown("---")
            st.markdown("## 🎯 测验结果")
            
            # 得分展示
            if score >= 80:
                st.success(f"🌟 优秀！你的得分：{score:.0f}%")
                st.balloons()
            elif score >= 60:
                st.info(f"👍 不错！你的得分：{score:.0f}%")
            else:
                st.warning(f"💪 继续加油！你的得分：{score:.0f}%")
            
            # 详细解析
            st.markdown("### 📖 答案解析")
            for i, q in enumerate(available_questions):
                user_answer = st.session_state['quiz_answers'].get(i)
                is_correct = user_answer == q['correct']
                
                if is_correct:
                    st.markdown(f"""
                    <div style='padding: 1rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid #22c55e; border-radius: 5px; margin: 0.5rem 0;'>
                        <strong>✅ 问题 {i+1}: {q['question']}</strong><br>
                        <span style='color: #22c55e;'>你的答案正确！</span><br>
                        <small>{q['explanation']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='padding: 1rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; border-radius: 5px; margin: 0.5rem 0;'>
                        <strong>❌ 问题 {i+1}: {q['question']}</strong><br>
                        <span style='color: #ef4444;'>你的答案：{q['options'][user_answer] if user_answer is not None else '未作答'}</span><br>
                        <span style='color: #22c55e;'>正确答案：{q['options'][q['correct']]}</span><br>
                        <small>{q['explanation']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 重新测验按钮
            if st.button("🔄 重新测验", key="restart_quiz"):
                st.session_state['quiz_started'] = False
                st.session_state['quiz_answers'] = {}
                st.rerun()


def render_learning_progress_chart(completed_modules):
    """渲染学习进度图表"""
    from learning_paths import get_difficulty_score, RECOMMENDED_ORDER
    from config import MODULES
    
    st.markdown("### 📈 学习进度可视化")
    
    # 创建雷达图显示不同领域的掌握程度
    categories = {
        "基础数学": ["matrix", "calculus", "probability"],
        "优化理论": ["loss", "optimizer", "lagrange"],
        "正则化": ["regularization", "vcdim"],
        "模型应用": ["svm", "convolution"],
        "模型评估": ["ml_curves"]
    }
    
    category_scores = {}
    for cat_name, modules in categories.items():
        completed_in_cat = sum(1 for m in modules if m in completed_modules)
        total_in_cat = len(modules)
        category_scores[cat_name] = (completed_in_cat / total_in_cat * 100) if total_in_cat > 0 else 0
    
    # 创建雷达图
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(category_scores.values()),
        theta=list(category_scores.keys()),
        fill='toself',
        name='学习进度',
        line=dict(color='rgb(59, 130, 246)', width=2),
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="各领域掌握程度",
        height=400
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 显示时间线
        st.markdown("**📅 学习时间线**")
        
        ordered_completed = []
        for module in RECOMMENDED_ORDER:
            if module in completed_modules:
                ordered_completed.append(module)
        
        if ordered_completed:
            for i, module_key in enumerate(ordered_completed, 1):
                module_info = MODULES[module_key]
                difficulty = get_difficulty_score(module_key)
                st.markdown(f"""
                <div style='padding: 0.5rem; margin: 0.3rem 0; background: rgba(34, 197, 94, 0.1); 
                           border-left: 3px solid #22c55e; border-radius: 5px;'>
                    <strong>{i}. {module_info['name']}</strong><br>
                    <small>难度: {'⭐' * difficulty}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("开始学习后，这里会显示你的学习轨迹")


def show_concept_map():
    """显示层次化的概念关系图"""
    st.markdown("## 🕸️ 知识图谱")
    
    st.markdown("""
    机器学习数学知识按层次组织，从基础数学到前沿研究共20个层次。箭头表示依赖关系：学习后续概念需要先掌握前置知识。
    """)
    
    from learning_paths import CONCEPT_DEPENDENCIES
    
    # 选择可视化模式
    view_mode = st.radio(
        "选择视图模式",
        ["层次视图", "关系网络视图", "概念列表"],
        horizontal=True,
        help="层次视图按学习顺序组织，网络视图展示依赖关系，列表视图方便查找"
    )
    
    if view_mode == "层次视图":
        render_hierarchical_view()
    elif view_mode == "关系网络视图":
        render_network_view()
    else:
        render_list_view()


def render_hierarchical_view():
    """渲染层次化视图"""
    from learning_paths import CONCEPT_DEPENDENCIES
    
    # 按层次分组概念
    layers = {
        "第1层：数学基础": ["矩阵", "向量", "导数", "概率", "范数"],
        "第2层：数学工具": ["线性变换", "特征值", "特征向量", "SVD", "梯度", "链式法则", "熵", "条件概率"],
        "第3层：ML基础": ["梯度下降", "反向传播", "最小二乘", "交叉熵", "KL散度", "最大似然"],
        "第4层：优化算法": ["动量", "Adam", "学习率调度", "批归一化"],
        "第5层：正则化": ["L1正则化", "L2正则化", "权重衰减", "Dropout"],
        "第6层：约束优化": ["拉格朗日乘子", "KKT条件", "对偶问题", "凸优化"],
        "第7层：核方法": ["内积空间", "希尔伯特空间", "核函数", "核技巧", "SVM"],
        "第8层：深度架构": ["卷积", "池化", "残差连接", "注意力机制", "Transformer"],
        "第9层：信号处理": ["傅里叶变换", "卷积定理", "STFT", "小波变换"],
        "第10层：概率图": ["贝叶斯网络", "变分推断", "MCMC", "ELBO"],
        "第11层：生成模型": ["VAE", "GAN", "扩散模型", "归一化流"],
        "第12层：强化学习": ["MDP", "价值迭代", "策略梯度", "Q学习"],
        "第13层：图神经网络": ["图拉普拉斯", "谱图卷积", "消息传递", "图注意力"],
        "第14层：信息几何": ["费雪信息", "自然梯度", "KL球"],
        "第15层：泛化理论": ["PAC学习", "VC维", "Rademacher复杂度"],
        "第16层：因果推断": ["因果图", "Do算子", "反事实"],
        "第17层：最优传输": ["Wasserstein距离", "Kantorovich对偶", "Sinkhorn"],
        "第18层：博弈论": ["纳什均衡", "Stackelberg", "演化稳定"],
        "第19层：多模态": ["对比学习", "模态对齐", "CLIP"],
        "第20层：训练动力学": ["NTK", "双下降", "临界学习期"],
        "工程实践": ["缩放定律", "参数计算", "FLOPs估算", "显存估算"],
    }
    
    # 检查已掌握的概念
    if 'completed_modules' not in st.session_state:
        st.session_state['completed_modules'] = set()
    
    from learning_paths import get_learning_stats
    stats = get_learning_stats(st.session_state['completed_modules'])
    learned_concepts = set(stats.get('concepts', []))
    
    # 显示层次结构
    for layer_name, concepts in layers.items():
        with st.expander(f"📚 {layer_name} ({len(concepts)}个概念)", expanded=(layer_name in ["第1层：数学基础", "第2层：数学工具", "第3层：ML基础"])):
            # 统计该层掌握情况
            learned_in_layer = [c for c in concepts if c in learned_concepts]
            progress = len(learned_in_layer) / len(concepts) if concepts else 0
            
            if learned_in_layer:
                st.progress(progress, text=f"已掌握 {len(learned_in_layer)}/{len(concepts)} ({progress*100:.0f}%)")
            
            # 显示概念卡片
            cols = st.columns(4)
            for i, concept in enumerate(concepts):
                with cols[i % 4]:
                    is_learned = concept in learned_concepts
                    prereqs = CONCEPT_DEPENDENCIES.get(concept, [])
                    
                    # 检查前置条件是否满足
                    prereqs_met = all(p in learned_concepts for p in prereqs) if prereqs else True
                    
                    if is_learned:
                        status_icon = "✅"
                        color = "#22c55e"
                    elif prereqs_met:
                        status_icon = "🔓"
                        color = "#3b82f6"
                    else:
                        status_icon = "🔒"
                        color = "#94a3b8"
                    
                    st.markdown(f"""
                    <div style='padding: 0.8rem; background: rgba(26, 26, 46, 0.6); 
                                border-left: 3px solid {color}; border-radius: 8px; margin: 0.3rem 0;'>
                        <div style='font-weight: bold; color: white;'>{status_icon} {concept}</div>
                        <div style='font-size: 0.75rem; color: #888; margin-top: 0.3rem;'>
                            {f"需要: {', '.join(prereqs[:2])}{('...' if len(prereqs) > 2 else '')}" if prereqs else "基础概念"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def render_network_view():
    """渲染3D层次化网络视图"""
    from learning_paths import CONCEPT_DEPENDENCIES
    
    # 让用户选择过滤器
    st.markdown("#### 🎛️ 显示选项")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_3d = st.checkbox("3D层次视图", value=True, help="每层在不同高度，更清晰")
    
    with col2:
        show_mode = st.selectbox(
            "显示范围",
            ["完整网络", "基础层（1-5）", "高级层（6-10）", "前沿层（11+）"],
            help="过滤概念以简化视图"
        )
    
    with col3:
        show_labels = st.checkbox("显示标签", value=True, help="显示概念名称")
    
    # 使用networkx创建图
    G = nx.DiGraph()
    
    for concept, prerequisites in CONCEPT_DEPENDENCIES.items():
        for prereq in prerequisites:
            G.add_edge(prereq, concept)
    
    # 计算每个节点的层次深度
    def get_depth(node, memo={}):
        if node in memo:
            return memo[node]
        prereqs = CONCEPT_DEPENDENCIES.get(node, [])
        if not prereqs:
            memo[node] = 0
            return 0
        depth = 1 + max(get_depth(p, memo) for p in prereqs if p in CONCEPT_DEPENDENCIES)
        memo[node] = depth
        return depth
    
    depths = {node: get_depth(node) for node in G.nodes()}
    
    # 根据模式过滤节点
    if show_mode != "完整网络":
        if show_mode == "基础层（1-5）":
            nodes_to_keep = [n for n, d in depths.items() if d <= 5]
        elif show_mode == "高级层（6-10）":
            nodes_to_keep = [n for n, d in depths.items() if 5 < d <= 10]
        else:  # 前沿层
            nodes_to_keep = [n for n, d in depths.items() if d > 10]
        
        G = G.subgraph(nodes_to_keep).copy()
        depths = {k: v for k, v in depths.items() if k in nodes_to_keep}
    
    # 检查用户学习进度
    if 'completed_modules' not in st.session_state:
        st.session_state['completed_modules'] = set()
    
    from learning_paths import get_learning_stats
    stats = get_learning_stats(st.session_state['completed_modules'])
    learned_concepts = set(stats.get('concepts', []))
    
    if view_3d:
        render_3d_network(G, depths, learned_concepts, CONCEPT_DEPENDENCIES, show_labels)
    else:
        render_2d_network(G, depths, learned_concepts, CONCEPT_DEPENDENCIES, show_labels)


def render_3d_network(G, depths, learned_concepts, CONCEPT_DEPENDENCIES, show_labels):
    """渲染3D层次化网络"""
    import plotly.graph_objects as go
    import numpy as np
    
    # 检查是否有节点
    if len(G.nodes()) == 0 or len(depths) == 0:
        st.warning("⚠️ 当前过滤条件下没有概念可显示，请调整过滤选项。")
        return
    
    # 按层次分组
    layers = {}
    for node, depth in depths.items():
        if node in G.nodes():  # 确保节点在图中
            if depth not in layers:
                layers[depth] = []
            layers[depth].append(node)
    
    if not layers:
        st.warning("⚠️ 没有可显示的概念，请检查过滤条件。")
        return
    
    # 为每层创建圆形布局
    pos_3d = {}
    max_layer_size = max(len(nodes) for nodes in layers.values())
    
    for depth, nodes in layers.items():
        n = len(nodes)
        # 圆形布局
        radius = 1.5 + depth * 0.3  # 层次越高，圆越大
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = depth * 2  # Z轴高度代表层次
            pos_3d[node] = (x, y, z)
    
    # 创建边
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        if edge[0] in pos_3d and edge[1] in pos_3d:
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=2, color='rgba(150, 150, 180, 0.2)'),
        hoverinfo='none',
        showlegend=False
    )
    
    # 创建节点（按状态分类）
    traces = [edge_trace]
    
    for status, color, name, symbol in [
        ('learned', 'rgba(74, 222, 128, 0.9)', '已掌握', 'circle'),
        ('available', 'rgba(96, 165, 250, 0.8)', '可学习', 'circle'),
        ('locked', 'rgba(148, 163, 184, 0.5)', '需前置', 'circle')
    ]:
        node_x, node_y, node_z, node_text, hover_text = [], [], [], [], []
        
        for node in G.nodes():
            if node not in pos_3d:
                continue
            
            x, y, z = pos_3d[node]
            
            # 判断状态
            if status == 'learned' and node in learned_concepts:
                pass
            elif status == 'available' and node not in learned_concepts:
                prereqs = CONCEPT_DEPENDENCIES.get(node, [])
                if not all(p in learned_concepts for p in prereqs):
                    continue
            elif status == 'locked' and node not in learned_concepts:
                prereqs = CONCEPT_DEPENDENCIES.get(node, [])
                if all(p in learned_concepts for p in prereqs):
                    continue
            else:
                continue
            
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node)
            
            prereqs = CONCEPT_DEPENDENCIES.get(node, [])
            prereq_str = f"<br>前置: {', '.join(prereqs[:3])}" if prereqs else "<br>基础概念"
            if len(prereqs) > 3:
                prereq_str += "..."
            hover_text.append(f"<b>{node}</b><br>层次: {depths.get(node, 0)}{prereq_str}")
        
        if node_x:
            traces.append(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text' if show_labels else 'markers',
                text=node_text if show_labels else None,
                textposition='top center',
                textfont=dict(size=8, color='rgba(220, 220, 220, 0.9)'),
                marker=dict(
                    size=8,
                    color=color,
                    symbol=symbol,
                    line=dict(width=1, color='rgba(255, 255, 255, 0.3)')
                ),
                name=name,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text
            ))
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=dict(
            text="3D 知识层次结构（高度=学习层次）",
            font=dict(size=16, color='rgba(220, 220, 220, 0.9)'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                showbackground=False,
                title='',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                showbackground=False,
                title='',
                zeroline=False
            ),
            zaxis=dict(
                showgrid=True,
                gridcolor='rgba(100, 100, 120, 0.2)',
                showticklabels=True,
                showbackground=False,
                title=dict(text='学习层次', font=dict(size=12, color='rgba(200, 200, 200, 0.7)')),
                zeroline=False
            ),
            bgcolor='rgba(15, 15, 25, 0.3)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            font=dict(size=11, color='rgba(220, 220, 220, 0.9)'),
            bgcolor='rgba(30, 30, 50, 0.6)',
            bordercolor='rgba(100, 100, 120, 0.3)',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='rgba(30, 30, 50, 0.95)',
            font_size=11,
            font_color='white'
        ),
        height=800,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 统计信息
    display_network_stats(G, learned_concepts, CONCEPT_DEPENDENCIES)


def render_2d_network(G, depths, learned_concepts, CONCEPT_DEPENDENCIES, show_labels):
    """渲染2D网络（备用）"""
    import plotly.graph_objects as go
    
    # 检查是否有节点
    if len(G.nodes()) == 0:
        st.warning("⚠️ 当前过滤条件下没有概念可显示，请调整过滤选项。")
        return
    
    # 使用层次布局
    try:
        if len(G.nodes()) < 50:
            pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
        else:
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    except Exception as e:
        st.error(f"❌ 布局计算失败: {e}")
        return
    
    # 2D布局
    edge_x, edge_y = [], []
    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.8, color='rgba(120, 120, 150, 0.2)'),
        hoverinfo='none',
        showlegend=False
    )
    
    traces = [edge_trace]
    
    for status, color, name in [
        ('learned', 'rgba(74, 222, 128, 0.9)', '已掌握'),
        ('available', 'rgba(96, 165, 250, 0.8)', '可学习'),
        ('locked', 'rgba(148, 163, 184, 0.5)', '需前置')
    ]:
        node_x, node_y, node_text, hover_text = [], [], [], []
        
        for node in G.nodes():
            if node not in pos:
                continue
            x, y = pos[node]
            
            if status == 'learned' and node in learned_concepts:
                pass
            elif status == 'available' and node not in learned_concepts:
                prereqs = CONCEPT_DEPENDENCIES.get(node, [])
                if not all(p in learned_concepts for p in prereqs):
                    continue
            elif status == 'locked' and node not in learned_concepts:
                prereqs = CONCEPT_DEPENDENCIES.get(node, [])
                if all(p in learned_concepts for p in prereqs):
                    continue
            else:
                continue
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            prereqs = CONCEPT_DEPENDENCIES.get(node, [])
            prereq_str = f"<br>前置: {', '.join(prereqs[:3])}" if prereqs else "<br>基础概念"
            if len(prereqs) > 3:
                prereq_str += "..."
            hover_text.append(f"<b>{node}</b><br>层次: {depths.get(node, 0)}{prereq_str}")
        
        if node_x:
            traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if show_labels else 'markers',
                text=node_text if show_labels else None,
                textposition='top center',
                textfont=dict(size=7, color='rgba(200, 200, 200, 0.8)'),
                marker=dict(size=10, color=color, line=dict(width=1, color='rgba(255, 255, 255, 0.3)')),
                name=name,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text
            ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="2D 概念网络", font=dict(size=16, color='rgba(220, 220, 220, 0.9)'), x=0.5, xanchor='center'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                   font=dict(size=11, color='rgba(220, 220, 220, 0.9)'),
                   bgcolor='rgba(30, 30, 50, 0.6)', bordercolor='rgba(100, 100, 120, 0.3)', borderwidth=1),
        hovermode='closest',
        hoverlabel=dict(bgcolor='rgba(30, 30, 50, 0.95)', font_size=11, font_color='white'),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, showline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, showline=False),
        height=750,
        plot_bgcolor='rgba(20, 20, 35, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    display_network_stats(G, learned_concepts, CONCEPT_DEPENDENCIES)


def display_network_stats(G, learned_concepts, CONCEPT_DEPENDENCIES):
    """显示网络统计信息"""
    st.markdown("#### 📊 学习进度统计")
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(G.nodes())
    learned = len([n for n in G.nodes() if n in learned_concepts])
    available = len([n for n in G.nodes() if n not in learned_concepts and 
                     all(p in learned_concepts for p in CONCEPT_DEPENDENCIES.get(n, []))])
    locked = total - learned - available
    
    with col1:
        st.metric("总概念数", total, help="当前视图中的概念总数")
    with col2:
        st.metric("已掌握", learned, delta=f"{learned/total*100:.0f}%" if total > 0 else "0%", help="已完成学习")
    with col3:
        st.metric("可学习", available, delta=f"{available/total*100:.0f}%" if total > 0 else "0%", help="前置已满足")
    with col4:
        st.metric("待解锁", locked, delta=f"{locked/total*100:.0f}%" if total > 0 else "0%", help="需前置知识")
    
    st.info("💡 **3D视图**: 旋转拖动查看 | Z轴高度=学习层次 | 可过滤显示减少复杂度")


def render_list_view():
    """渲染列表视图"""
    from learning_paths import CONCEPT_DEPENDENCIES
    
    st.markdown("### 📋 完整概念索引")
    
    # 搜索功能
    search_term = st.text_input("🔍 搜索概念", placeholder="输入概念名称...")
    
    if 'completed_modules' not in st.session_state:
        st.session_state['completed_modules'] = set()
    
    from learning_paths import get_learning_stats
    stats = get_learning_stats(st.session_state['completed_modules'])
    learned_concepts = set(stats.get('concepts', []))
    
    # 过滤和排序
    concepts = list(CONCEPT_DEPENDENCIES.keys())
    if search_term:
        concepts = [c for c in concepts if search_term.lower() in c.lower()]
    
    concepts.sort()
    
    st.markdown(f"找到 **{len(concepts)}** 个概念")
    
    # 显示概念列表
    for concept in concepts:
        prereqs = CONCEPT_DEPENDENCIES.get(concept, [])
        is_learned = concept in learned_concepts
        prereqs_met = all(p in learned_concepts for p in prereqs) if prereqs else True
        
        if is_learned:
            icon, color = "✅", "#22c55e"
        elif prereqs_met:
            icon, color = "🔓", "#3b82f6"
        else:
            icon, color = "🔒", "#94a3b8"
        
        with st.expander(f"{icon} {concept}"):
            if prereqs:
                st.markdown(f"**前置概念**: {', '.join(prereqs)}")
                prereqs_status = [f"{'✅' if p in learned_concepts else '❌'} {p}" for p in prereqs]
                st.markdown("**前置状态**: " + " | ".join(prereqs_status))
            else:
                st.info("这是基础概念，无需前置知识")
            
            # 找出依赖此概念的后续概念
            dependents = [k for k, v in CONCEPT_DEPENDENCIES.items() if concept in v]
            if dependents:
                st.markdown(f"**后续概念** ({len(dependents)}个): {', '.join(dependents[:5])}{('...' if len(dependents) > 5 else '')}")
