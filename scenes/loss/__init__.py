# 损失函数模块初始化文件

from .least_squares import LeastSquaresScene
from .cross_entropy import CrossEntropyScene
from .gradient_descent_3d import Basic3DGradient

__all__ = [
    'LeastSquaresScene',
    'CrossEntropyScene',
    'Basic3DGradient'
]