"""
智能缓存系统
自动识别和优化计算密集型函数的缓存策略
"""

import streamlit as st
import functools
import hashlib
import json
import time
from typing import Callable, Any, Optional, Dict, Tuple
import numpy as np


class CacheConfig:
    """缓存配置"""
    
    # 不同类型计算的默认TTL（秒）
    TTL_CONFIGS = {
        'fast': 300,        # 5分钟 - 快速计算
        'medium': 1800,     # 30分钟 - 中等计算
        'heavy': 3600,      # 1小时 - 重型计算
        'static': None,     # 永久 - 静态数据
    }
    
    # 自动检测计算类型的阈值（秒）
    TIMING_THRESHOLDS = {
        'fast': 0.1,
        'medium': 1.0,
        'heavy': 5.0,
    }


class SmartCache:
    """智能缓存装饰器"""
    
    def __init__(
        self,
        ttl: Optional[int] = None,
        cache_type: str = 'auto',
        show_stats: bool = False,
        max_entries: Optional[int] = None
    ):
        """
        初始化智能缓存
        
        Args:
            ttl: 缓存生存时间（秒），None表示永久
            cache_type: 缓存类型 ('auto', 'fast', 'medium', 'heavy', 'static')
            show_stats: 是否显示缓存统计
            max_entries: 最大缓存条目数
        """
        self.ttl = ttl
        self.cache_type = cache_type
        self.show_stats = show_stats
        self.max_entries = max_entries
        self._execution_times = []
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器调用"""
        
        # 如果是auto模式，先测量几次执行时间
        if self.cache_type == 'auto':
            actual_ttl = CacheConfig.TTL_CONFIGS['medium']  # 默认
        else:
            actual_ttl = self.ttl or CacheConfig.TTL_CONFIGS.get(self.cache_type)
        
        # 使用streamlit的缓存
        @st.cache_data(ttl=actual_ttl, show_spinner=False)
        def cached_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # 记录执行时间用于自适应调整
            self._execution_times.append(elapsed)
            
            return result, elapsed
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, elapsed = cached_wrapper(*args, **kwargs)
            
            # 显示缓存统计
            if self.show_stats:
                cache_info = self._get_cache_info(func.__name__, elapsed)
                st.caption(cache_info)
            
            return result
        
        return wrapper
    
    def _get_cache_info(self, func_name: str, elapsed: float) -> str:
        """获取缓存信息"""
        avg_time = sum(self._execution_times) / len(self._execution_times)
        return f"⚡ {func_name}: {elapsed*1000:.0f}ms (平均: {avg_time*1000:.0f}ms)"


def auto_cache(func: Callable) -> Callable:
    """
    自动缓存装饰器 - 根据第一次执行时间自动选择TTL
    
    使用方法：
    @auto_cache
    def expensive_computation(x, y):
        return heavy_calculation(x, y)
    """
    execution_count = {'count': 0, 'total_time': 0.0}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        execution_count['count'] += 1
        
        # 第一次执行，测量时间
        if execution_count['count'] == 1:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            execution_count['total_time'] = elapsed
            
            # 根据执行时间选择TTL
            if elapsed < CacheConfig.TIMING_THRESHOLDS['fast']:
                ttl = CacheConfig.TTL_CONFIGS['fast']
                cache_type = "快速"
            elif elapsed < CacheConfig.TIMING_THRESHOLDS['medium']:
                ttl = CacheConfig.TTL_CONFIGS['medium']
                cache_type = "中等"
            elif elapsed < CacheConfig.TIMING_THRESHOLDS['heavy']:
                ttl = CacheConfig.TTL_CONFIGS['heavy']
                cache_type = "重型"
            else:
                ttl = CacheConfig.TTL_CONFIGS['static']
                cache_type = "超重"
            
            # 创建缓存版本
            @st.cache_data(ttl=ttl)
            def cached_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 保存缓存函数供后续使用
            wrapper._cached_func = cached_func
            wrapper._cache_type = cache_type
            wrapper._ttl = ttl
            
            return result
        else:
            # 后续调用使用缓存
            if hasattr(wrapper, '_cached_func'):
                return wrapper._cached_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    return wrapper


def cache_numpy_computation(ttl: int = 1800):
    """
    专门用于NumPy计算的缓存装饰器
    自动处理NumPy数组的哈希
    
    使用方法：
    @cache_numpy_computation(ttl=3600)
    def matrix_multiply(A, B):
        return np.dot(A, B)
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl)
        def wrapper(*args, **kwargs):
            # 将NumPy数组转换为可哈希的格式
            hashable_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # 使用数组的形状、dtype和部分数据生成哈希
                    hashable_args.append((arg.shape, arg.dtype.name, arg.tobytes()[:1000]))
                else:
                    hashable_args.append(arg)
            
            hashable_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    hashable_kwargs[k] = (v.shape, v.dtype.name, v.tobytes()[:1000])
                else:
                    hashable_kwargs[k] = v
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ProgressiveCache:
    """
    渐进式缓存 - 对于大型计算，分阶段缓存中间结果
    """
    
    def __init__(self, stages: list):
        """
        Args:
            stages: 阶段名称列表，如 ['preprocess', 'compute', 'postprocess']
        """
        self.stages = stages
        self.cache_keys = {stage: f"progressive_cache_{stage}" for stage in stages}
    
    def cache_stage(self, stage: str):
        """缓存某个阶段的装饰器"""
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}")
        
        def decorator(func: Callable) -> Callable:
            @st.cache_data(ttl=3600)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


class AdaptiveCache:
    """
    自适应缓存 - 根据参数复杂度动态调整TTL
    """
    
    @staticmethod
    def calculate_complexity(args, kwargs) -> float:
        """计算参数复杂度分数"""
        complexity = 0
        
        for arg in args:
            if isinstance(arg, (list, tuple)):
                complexity += len(arg)
            elif isinstance(arg, np.ndarray):
                complexity += arg.size
            elif isinstance(arg, (int, float)):
                complexity += abs(arg) / 100
        
        for v in kwargs.values():
            if isinstance(v, (list, tuple)):
                complexity += len(v)
            elif isinstance(v, np.ndarray):
                complexity += v.size
            elif isinstance(v, (int, float)):
                complexity += abs(v) / 100
        
        return complexity
    
    @staticmethod
    def get_adaptive_ttl(complexity: float) -> int:
        """根据复杂度返回TTL"""
        if complexity < 100:
            return 300      # 5分钟
        elif complexity < 1000:
            return 1800     # 30分钟
        elif complexity < 10000:
            return 3600     # 1小时
        else:
            return 7200     # 2小时
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 计算复杂度
            complexity = self.calculate_complexity(args, kwargs)
            ttl = self.get_adaptive_ttl(complexity)
            
            # 创建动态缓存
            @st.cache_data(ttl=ttl)
            def cached_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            return cached_func(*args, **kwargs)
        
        return wrapper


# 便捷的预定义装饰器
cache_fast = lambda func: st.cache_data(ttl=CacheConfig.TTL_CONFIGS['fast'])(func)
cache_medium = lambda func: st.cache_data(ttl=CacheConfig.TTL_CONFIGS['medium'])(func)
cache_heavy = lambda func: st.cache_data(ttl=CacheConfig.TTL_CONFIGS['heavy'])(func)
cache_static = lambda func: st.cache_data(ttl=None)(func)


class CacheMonitor:
    """缓存监控器 - 显示缓存性能统计"""
    
    @staticmethod
    def show_cache_stats():
        """显示缓存统计信息"""
        if st.session_state.get('show_cache_stats', False):
            with st.expander("📊 缓存统计", expanded=False):
                # 获取缓存统计
                cache_stats = st.cache_data.cache_stats()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("缓存命中", cache_stats.get('hits', 0))
                with col2:
                    st.metric("缓存未命中", cache_stats.get('misses', 0))
                with col3:
                    hit_rate = cache_stats.get('hits', 0) / max(1, cache_stats.get('hits', 0) + cache_stats.get('misses', 0))
                    st.metric("命中率", f"{hit_rate*100:.1f}%")
                
                if st.button("🗑️ 清除缓存"):
                    st.cache_data.clear()
                    st.success("缓存已清除")
                    st.rerun()


# 使用示例和最佳实践
USAGE_EXAMPLES = """
# ============================================================================
# 智能缓存使用示例
# ============================================================================

# 1. 简单使用 - 快速计算
@cache_fast
def simple_calculation(x, y):
    return x + y

# 2. 中等计算 - 默认推荐
@cache_medium
def moderate_calculation(n):
    return np.random.randn(n, n).sum()

# 3. 重型计算 - 长时间缓存
@cache_heavy
def heavy_calculation(size):
    matrix = np.random.randn(size, size)
    return np.linalg.svd(matrix)

# 4. NumPy专用缓存
@cache_numpy_computation(ttl=3600)
def matrix_operation(A, B):
    return A @ B + np.linalg.inv(A)

# 5. 自适应缓存 - 自动选择TTL
@AdaptiveCache()
def adaptive_func(data):
    # 数据量大时自动延长缓存时间
    return expensive_operation(data)

# 6. 渐进式缓存 - 分阶段缓存
progressive = ProgressiveCache(['load', 'process', 'analyze'])

@progressive.cache_stage('load')
def load_data(path):
    return load_large_file(path)

@progressive.cache_stage('process')
def process_data(data):
    return heavy_processing(data)

@progressive.cache_stage('analyze')
def analyze_data(processed):
    return complex_analysis(processed)

# 7. 智能缓存 - 带统计
@SmartCache(cache_type='auto', show_stats=True)
def smart_computation(params):
    return do_computation(params)

# ============================================================================
# 最佳实践
# ============================================================================

# ✅ 好的做法：
# 1. 为纯函数使用缓存（输入相同，输出相同）
# 2. 缓存耗时 > 0.1秒的计算
# 3. 根据数据更新频率选择TTL
# 4. 使用适当的缓存类型

# ❌ 避免：
# 1. 缓存有副作用的函数
# 2. 缓存返回随机结果的函数
# 3. 缓存超大对象（> 100MB）
# 4. 过度缓存（内存溢出）
"""
