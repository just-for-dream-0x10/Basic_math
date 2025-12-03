"""
性能优化工具
提供缓存、延迟加载等性能优化功能
"""

import streamlit as st
import functools
import time
from typing import Callable, Any, Optional, Dict
import hashlib
import json


def cache_data(ttl: Optional[int] = 3600, show_spinner: bool = True):
    """
    数据缓存装饰器（包装st.cache_data）
    
    Args:
        ttl: 缓存生存时间（秒），None表示永久缓存
        show_spinner: 是否显示加载动画
    
    使用方法：
    @cache_data(ttl=3600)
    def expensive_computation(x, y):
        # 耗时计算
        return result
    """
    def decorator(func: Callable) -> Callable:
        cached_func = st.cache_data(ttl=ttl, show_spinner=show_spinner)(func)
        return cached_func
    return decorator


def cache_resource(show_spinner: bool = True):
    """
    资源缓存装饰器（包装st.cache_resource）
    用于缓存模型、数据库连接等资源
    
    使用方法：
    @cache_resource()
    def load_model():
        # 加载大型模型
        return model
    """
    def decorator(func: Callable) -> Callable:
        cached_func = st.cache_resource(show_spinner=show_spinner)(func)
        return cached_func
    return decorator


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, operation_name: str, show_time: bool = False):
        """
        初始化性能监控器
        
        Args:
            operation_name: 操作名称
            show_time: 是否显示执行时间
        """
        self.operation_name = operation_name
        self.show_time = show_time
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        if self.show_time:
            if elapsed < 1:
                st.caption(f"⏱️ {self.operation_name}: {elapsed*1000:.0f}ms")
            else:
                st.caption(f"⏱️ {self.operation_name}: {elapsed:.2f}s")
        
        # 如果执行时间过长，显示警告
        if elapsed > 5:
            st.warning(f"⚠️ {self.operation_name}耗时较长 ({elapsed:.1f}s)，考虑优化或使用缓存")
        
        return False


def lazy_load(placeholder_text: str = "点击加载..."):
    """
    延迟加载装饰器
    用于大型可视化的按需加载
    
    使用方法：
    @lazy_load("加载3D可视化")
    def render_3d_plot():
        # 复杂的3D绘图
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_key = f"lazy_{func.__name__}"
            
            # 创建加载按钮
            if st.button(f"🔄 {placeholder_text}", key=func_key):
                with st.spinner(f"正在{placeholder_text}..."):
                    result = func(*args, **kwargs)
                return result
            else:
                st.info(f"💡 点击按钮{placeholder_text}")
                return None
        
        return wrapper
    return decorator


class BatchProcessor:
    """批处理器 - 用于处理大量数据时分批显示"""
    
    def __init__(self, items: list, batch_size: int = 10):
        """
        初始化批处理器
        
        Args:
            items: 要处理的项目列表
            batch_size: 每批处理的数量
        """
        self.items = items
        self.batch_size = batch_size
        self.total_batches = (len(items) + batch_size - 1) // batch_size
    
    def render_with_pagination(self, render_func: Callable):
        """
        分页渲染
        
        Args:
            render_func: 渲染函数，接收单个item作为参数
        """
        if not self.items:
            st.info("暂无数据")
            return
        
        # 分页控制
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            page = st.slider(
                "页码",
                min_value=1,
                max_value=self.total_batches,
                value=1,
                key="batch_page"
            )
        
        # 计算当前批次
        start_idx = (page - 1) * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.items))
        current_batch = self.items[start_idx:end_idx]
        
        # 显示当前范围
        st.caption(f"显示 {start_idx + 1}-{end_idx} / 共 {len(self.items)} 项")
        
        # 渲染当前批次
        for item in current_batch:
            render_func(item)


@cache_data(ttl=3600)
def generate_sample_data(n_samples: int, noise_level: float, seed: int = 42):
    """
    生成示例数据（带缓存）
    
    Args:
        n_samples: 样本数量
        noise_level: 噪声水平
        seed: 随机种子
    
    Returns:
        X, y: 特征和标签
    """
    import numpy as np
    np.random.seed(seed)
    
    X = np.linspace(-5, 5, n_samples)
    y = np.sin(X) + np.random.normal(0, noise_level, n_samples)
    
    return X, y


@cache_data(ttl=3600)
def compute_polynomial_features(X, degree: int):
    """
    计算多项式特征（带缓存）
    
    Args:
        X: 输入特征
        degree: 多项式阶数
    
    Returns:
        多项式特征矩阵
    """
    import numpy as np
    return np.column_stack([X**i for i in range(degree + 1)])


class PresetManager:
    """参数预设管理器"""
    
    def __init__(self, presets: Dict[str, Dict[str, Any]]):
        """
        初始化预设管理器
        
        Args:
            presets: 预设字典，格式为 {preset_name: {param_name: value}}
        """
        self.presets = presets
    
    def render_preset_selector(self, key_prefix: str = "preset") -> Optional[Dict[str, Any]]:
        """
        渲染预设选择器
        
        Args:
            key_prefix: session_state的key前缀
        
        Returns:
            选中的预设参数字典，如果选择"自定义"则返回None
        """
        preset_names = ["自定义"] + list(self.presets.keys())
        
        selected = st.selectbox(
            "📋 参数预设",
            options=preset_names,
            key=f"{key_prefix}_selector",
            help="选择预设参数或自定义"
        )
        
        if selected == "自定义":
            return None
        else:
            preset_params = self.presets[selected]
            st.info(f"💡 已加载预设：{selected}")
            with st.expander("查看预设参数"):
                for param, value in preset_params.items():
                    st.write(f"- **{param}**: {value}")
            return preset_params


def optimize_plotly_figure(fig, reduce_points: bool = True, max_points: int = 1000):
    """
    优化Plotly图表性能
    
    Args:
        fig: Plotly图表对象
        reduce_points: 是否减少数据点
        max_points: 最大数据点数
    
    Returns:
        优化后的图表
    """
    if reduce_points:
        for trace in fig.data:
            if hasattr(trace, 'x') and len(trace.x) > max_points:
                # 等间隔采样
                import numpy as np
                indices = np.linspace(0, len(trace.x) - 1, max_points, dtype=int)
                trace.x = [trace.x[i] for i in indices]
                trace.y = [trace.y[i] for i in indices]
    
    # 优化配置
    fig.update_layout(
        # 减少动画
        transition_duration=0,
        # 关闭不必要的交互
        hovermode='closest',
        # 优化渲染
        showlegend=True,
    )
    
    return fig


def measure_time(func: Callable) -> Callable:
    """
    测量函数执行时间的装饰器（用于调试）
    
    使用方法：
    @measure_time
    def slow_function():
        # 耗时操作
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        
        # 只在开发模式下显示
        if st.session_state.get('debug_mode', False):
            st.caption(f"🐛 DEBUG: {func.__name__} took {elapsed:.3f}s")
        
        return result
    
    return wrapper


# 通用预设配置
COMMON_PRESETS = {
    "推荐设置": {
        "描述": "适合大多数场景的平衡设置"
    },
    "快速演示": {
        "描述": "使用较少数据点，适合快速演示"
    },
    "高质量": {
        "描述": "使用更多数据点，适合生成高质量图表"
    },
    "极端情况": {
        "描述": "极端参数值，用于测试边界情况"
    }
}
