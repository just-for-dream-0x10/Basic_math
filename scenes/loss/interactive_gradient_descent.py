"""
交互式3D梯度下降可视化
使用 matplotlib + plotly 实现实时交互
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InteractiveGradientDescent:
    """交互式梯度下降可视化类"""
    
    def __init__(self, loss_function='rosenbrock', learning_rate=0.01, 
                 start_x=-2.0, start_y=2.0, iterations=100):
        """
        初始化参数
        
        Args:
            loss_function: 损失函数类型 ('rosenbrock', 'sphere', 'beale', 'himmelblau')
            learning_rate: 学习率
            start_x: 起始点x坐标
            start_y: 起始点y坐标
            iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.start_point = np.array([start_x, start_y])
        self.iterations = iterations
        self.loss_function_name = loss_function
        
        # 设置损失函数和范围
        self.setup_loss_function(loss_function)
        
        # 运行梯度下降
        self.path, self.loss_history = self.gradient_descent()
    
    def setup_loss_function(self, name):
        """设置损失函数"""
        if name == 'rosenbrock':
            self.loss_fn = lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2
            self.x_range = (-2.5, 2.5)
            self.y_range = (-1, 3)
            self.title = "Rosenbrock函数（香蕉函数）"
        elif name == 'sphere':
            self.loss_fn = lambda x, y: x**2 + y**2
            self.x_range = (-3, 3)
            self.y_range = (-3, 3)
            self.title = "球面函数"
        elif name == 'beale':
            self.loss_fn = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
            self.x_range = (-4.5, 4.5)
            self.y_range = (-4.5, 4.5)
            self.title = "Beale函数"
        elif name == 'himmelblau':
            self.loss_fn = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            self.x_range = (-5, 5)
            self.y_range = (-5, 5)
            self.title = "Himmelblau函数"
        else:
            # 默认使用sphere
            self.loss_fn = lambda x, y: x**2 + y**2
            self.x_range = (-3, 3)
            self.y_range = (-3, 3)
            self.title = "球面函数"
    
    def compute_gradient(self, x, y):
        """数值计算梯度"""
        h = 1e-5
        grad_x = (self.loss_fn(x + h, y) - self.loss_fn(x - h, y)) / (2 * h)
        grad_y = (self.loss_fn(x, y + h) - self.loss_fn(x, y - h)) / (2 * h)
        return np.array([grad_x, grad_y])
    
    def gradient_descent(self):
        """执行梯度下降"""
        path = [self.start_point.copy()]
        loss_history = [self.loss_fn(self.start_point[0], self.start_point[1])]
        current_point = self.start_point.copy()
        
        for i in range(self.iterations):
            gradient = self.compute_gradient(current_point[0], current_point[1])
            current_point = current_point - self.learning_rate * gradient
            
            path.append(current_point.copy())
            loss_history.append(self.loss_fn(current_point[0], current_point[1]))
        
        return np.array(path), np.array(loss_history)
    
    def create_plotly_3d(self):
        """创建Plotly 3D交互图"""
        # 创建网格
        x = np.linspace(self.x_range[0], self.x_range[1], 100)
        y = np.linspace(self.y_range[0], self.y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.loss_fn(X, Y)
        
        # 限制Z值范围以便更好展示
        Z = np.minimum(Z, np.percentile(Z, 95))
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'scatter'}]],
            subplot_titles=(f'{self.title} - 3D曲面', '损失函数收敛曲线'),
            column_widths=[0.6, 0.4]
        )
        
        # 3D曲面
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                opacity=0.9,
                name='损失函数曲面'
            ),
            row=1, col=1
        )
        
        # 梯度下降路径
        path_z = [self.loss_fn(p[0], p[1]) for p in self.path]
        fig.add_trace(
            go.Scatter3d(
                x=self.path[:, 0],
                y=self.path[:, 1],
                z=path_z,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=4, color='red'),
                name='梯度下降路径'
            ),
            row=1, col=1
        )
        
        # 起点和终点标记
        fig.add_trace(
            go.Scatter3d(
                x=[self.path[0, 0]],
                y=[self.path[0, 1]],
                z=[path_z[0]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='diamond'),
                name='起始点'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=[self.path[-1, 0]],
                y=[self.path[-1, 1]],
                z=[path_z[-1]],
                mode='markers',
                marker=dict(size=8, color='blue', symbol='diamond'),
                name='终点'
            ),
            row=1, col=1
        )
        
        # 损失曲线
        fig.add_trace(
            go.Scatter(
                x=list(range(len(self.loss_history))),
                y=self.loss_history,
                mode='lines',
                line=dict(color='orange', width=2),
                name='损失值'
            ),
            row=1, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title=f'梯度下降可视化 (学习率={self.learning_rate}, 迭代={self.iterations})',
            height=600,
            showlegend=True,
            scene=dict(
                xaxis_title='w₁',
                yaxis_title='w₂',
                zaxis_title='Loss',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            )
        )
        
        fig.update_xaxes(title_text="迭代次数", row=1, col=2)
        fig.update_yaxes(title_text="损失值", row=1, col=2)
        
        return fig
    
    def create_matplotlib_contour(self):
        """创建Matplotlib等高线图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 创建网格
        x = np.linspace(self.x_range[0], self.x_range[1], 200)
        y = np.linspace(self.y_range[0], self.y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = self.loss_fn(X, Y)
        
        # 等高线图
        levels = np.logspace(np.log10(Z.min() + 1e-8), np.log10(Z.max()), 30)
        contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        ax1.clabel(contour, inline=True, fontsize=8)
        
        # 梯度下降路径
        ax1.plot(self.path[:, 0], self.path[:, 1], 'r-', linewidth=2, label='梯度下降路径')
        ax1.plot(self.path[:, 0], self.path[:, 1], 'ro', markersize=4)
        
        # 起点和终点
        ax1.plot(self.path[0, 0], self.path[0, 1], 'go', markersize=12, label='起始点')
        ax1.plot(self.path[-1, 0], self.path[-1, 1], 'b*', markersize=15, label='终点')
        
        ax1.set_xlabel('w₁', fontsize=12)
        ax1.set_ylabel('w₂', fontsize=12)
        ax1.set_title(f'{self.title} - 等高线图', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2.plot(self.loss_history, linewidth=2, color='orange')
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('损失值', fontsize=12)
        ax2.set_title('损失函数收敛曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, filename='gradient_descent.gif', fps=10):
        """创建动画GIF"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建网格
        x = np.linspace(self.x_range[0], self.x_range[1], 200)
        y = np.linspace(self.y_range[0], self.y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = self.loss_fn(X, Y)
        
        # 绘制等高线
        levels = np.logspace(np.log10(Z.min() + 1e-8), np.log10(Z.max()), 30)
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        
        # 初始化路径
        line, = ax.plot([], [], 'r-', linewidth=2, label='梯度下降路径')
        point, = ax.plot([], [], 'ro', markersize=10)
        
        ax.plot(self.path[0, 0], self.path[0, 1], 'go', markersize=12, label='起始点')
        ax.set_xlabel('w₁', fontsize=12)
        ax.set_ylabel('w₂', fontsize=12)
        ax.set_title(f'{self.title} - 梯度下降动画', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            line.set_data(self.path[:frame+1, 0], self.path[:frame+1, 1])
            point.set_data([self.path[frame, 0]], [self.path[frame, 1]])
            return line, point
        
        anim = FuncAnimation(fig, animate, frames=len(self.path), 
                           interval=1000//fps, blit=True, repeat=True)
        
        return anim, fig


# 使用示例
if __name__ == "__main__":
    # 创建可视化对象
    gd = InteractiveGradientDescent(
        loss_function='rosenbrock',
        learning_rate=0.003,
        start_x=-2.0,
        start_y=2.0,
        iterations=100
    )
    
    # 创建Plotly 3D图
    fig = gd.create_plotly_3d()
    fig.show()
    
    # 创建Matplotlib等高线图
    fig_contour = gd.create_matplotlib_contour()
    plt.show()
