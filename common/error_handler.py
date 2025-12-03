"""
统一错误处理模块
为所有交互式模块提供统一的错误处理和用户友好的错误提示
"""

import streamlit as st
import functools
import traceback
from typing import Callable, Any


def safe_render(func: Callable) -> Callable:
    """
    装饰器：为render方法提供统一的错误处理
    
    使用方法：
    @safe_render
    def render():
        # 你的渲染代码
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            st.error(f"❌ 缺少依赖库: {str(e)}")
            st.info("💡 请运行: `pip install -r requirements.txt`")
        except ValueError as e:
            st.error(f"❌ 参数错误: {str(e)}")
            st.info("💡 请检查输入参数是否在有效范围内")
        except KeyError as e:
            st.error(f"❌ 配置错误: 缺少键 {str(e)}")
            st.info("💡 请检查配置文件或联系开发者")
        except Exception as e:
            st.error(f"❌ 发生错误: {str(e)}")
            with st.expander("🔍 查看详细错误信息"):
                st.code(traceback.format_exc())
            st.info("💡 建议：刷新页面或调整参数重试。如果问题持续，请报告此错误。")
    return wrapper


def safe_compute(func: Callable) -> Callable:
    """
    装饰器：为计算密集型函数提供错误处理
    
    使用方法：
    @safe_compute
    def complex_calculation(data):
        # 你的计算代码
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            st.warning("⚠️ 检测到除零错误，已使用默认值")
            return None
        except OverflowError:
            st.warning("⚠️ 数值溢出，请减小参数范围")
            return None
        except Exception as e:
            st.error(f"❌ 计算错误: {str(e)}")
            return None
    return wrapper


def validate_parameters(**constraints):
    """
    装饰器：验证参数范围
    
    使用方法：
    @validate_parameters(alpha=(0, 1), n_samples=(1, 1000))
    def train_model(alpha, n_samples):
        # 你的代码
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 验证参数
            for param_name, (min_val, max_val) in constraints.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not (min_val <= value <= max_val):
                        st.error(f"❌ 参数 {param_name}={value} 超出范围 [{min_val}, {max_val}]")
                        return None
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ErrorContext:
    """
    上下文管理器：用于代码块的错误处理
    
    使用方法：
    with ErrorContext("生成可视化"):
        # 你的代码
    """
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            st.error(f"❌ {self.operation_name}时发生错误: {str(exc_val)}")
            with st.expander("🔍 查看详细信息"):
                st.code(traceback.format_exc())
            return True  # 抑制异常
        return False


def show_warning_if(condition: bool, message: str):
    """显示条件警告"""
    if condition:
        st.warning(f"⚠️ {message}")


def show_info_if(condition: bool, message: str):
    """显示条件提示"""
    if condition:
        st.info(f"💡 {message}")
