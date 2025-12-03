import numpy as np
from manim import *
from common.base_scene import BaseScene

class LeastSquaresScene(BaseScene):
    """最小二乘法可视化场景"""
    
    def __init__(self, **kwargs):
        super().__init__(module_name="loss", **kwargs)
    
    def construct(self):
        title = self.add_title("最小二乘法", "像扔沙包砸靶子")
        
        # 创建坐标轴
        axes = self.create_axes()
        self.add(axes)
        
        # 创建数据点
        data_points = self.create_data_points()
        self.add(data_points)
        
        # 演示拟合过程
        self.demonstrate_fitting(data_points, axes)
        
        # 显示损失函数
        self.show_loss_function()
        
        self.wait(2)
    
    def create_axes(self):
        """创建坐标轴"""
        axes = Axes(
            x_range=[-2, 6, 1],
            y_range=[-2, 6, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": self.text_color, "stroke_width": 2},
            tips=False
        )
        
        # 添加坐标轴标签
        x_label = Text("x", color=self.text_color).scale(0.6).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", color=self.text_color).scale(0.6).next_to(axes.y_axis.get_end(), UP)
        
        return VGroup(axes, x_label, y_label)
    
    def create_data_points(self):
        """创建数据点"""
        # 示例数据点
        points_data = [
            (0, 1), (1, 2), (2, 3), (3, 3.5), (4, 4.5)
        ]
        
        points = VGroup()
        for x, y in points_data:
            point = Dot(point=[x, y, 0], color=RED, radius=0.1)
            points.add(point)
        
        return points
    
    def demonstrate_fitting(self, data_points, axes_group):
        """演示拟合过程"""
        axes = axes_group[0]
        
        # 初始的随机直线
        initial_line = self.create_fitted_line(axes, slope=0.5, intercept=0.2, color=GRAY)
        self.play(Create(initial_line))
        
        # 逐步优化直线
        slopes = [0.5, 0.7, 0.85, 0.95]
        intercepts = [0.2, 0.5, 0.8, 1.0]
        
        for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
            new_line = self.create_fitted_line(axes, slope, intercept, color=self.primary_color)
            
            # 显示误差
            errors = self.calculate_errors(data_points, slope, intercept)
            error_lines = self.show_error_lines(data_points, axes, slope, intercept)
            
            # 显示总误差
            total_error = sum(errors)
            error_text = Text(f"总误差: {total_error:.2f}", color=self.primary_color).scale(0.5)
            error_text.to_corner(UR, buff=1.0)
            
            self.play(
                Transform(initial_line, new_line),
                *[Create(error_line) for error_line in error_lines],
                Write(error_text),
                run_time=1
            )
            
            self.wait(0.5)
            
            # 清除误差线和文本
            self.play(
                *[FadeOut(error_line) for error_line in error_lines],
                FadeOut(error_text)
            )
        
        # 最终拟合线
        final_line = self.create_fitted_line(axes, 0.95, 1.0, color=GREEN)
        self.play(Transform(initial_line, final_line))
        
        # 添加拟合线标签
        fit_label = MathTex("y = 0.95x + 1.0", color=GREEN).scale(0.6)
        fit_label.next_to(final_line.get_center(), UR, buff=0.3)
        self.play(Write(fit_label))
    
    def create_fitted_line(self, axes, slope, intercept, color):
        """创建拟合直线"""
        return axes.plot(lambda x: slope * x + intercept, color=color, stroke_width=3)
    
    def calculate_errors(self, data_points, slope, intercept):
        """计算误差"""
        errors = []
        for point in data_points:
            x, y = point.get_center()[0], point.get_center()[1]
            predicted = slope * x + intercept
            error = abs(y - predicted)
            errors.append(error)
        return errors
    
    def show_error_lines(self, data_points, axes, slope, intercept):
        """显示误差线"""
        error_lines = VGroup()
        
        for point in data_points:
            x, y = point.get_center()[0], point.get_center()[1]
            predicted = slope * x + intercept
            
            # 创建垂直误差线
            error_line = Line(
                start=[x, y, 0],
                end=[x, predicted, 0],
                color=RED,
                stroke_width=2,
                stroke_opacity=0.7
            )
            error_lines.add(error_line)
        
        return error_lines
    
    def show_loss_function(self):
        """显示损失函数公式"""
        loss_formula = MathTex(
            "L = \sum_{i=1}^{n} (y_i - (ax_i + b))^2",
            color=self.text_color
        ).scale(0.8)
        
        loss_formula.to_corner(DL, buff=1.0)
        self.play(Write(loss_formula))
        
        # 添加解释
        explanation = Text(
            "最小化所有点到直线的垂直距离平方和",
            color=self.text_color
        ).scale(0.5)
        explanation.next_to(loss_formula, DOWN, buff=0.5)
        self.play(Write(explanation))

class CrossEntropyScene(BaseScene):
    """交叉熵损失可视化场景"""
    
    def construct(self):
        title = self.add_title("交叉熵损失", "测量预测有多烂的方法")
        
        # 创建概率分布
        self.create_probability_distributions()
        
        # 演示交叉熵计算
        self.demonstrate_cross_entropy()
        
        # 显示交叉熵公式
        self.show_cross_entropy_formula()
        
        self.wait(2)
    
    def create_probability_distributions(self):
        """创建概率分布可视化"""
        # 创建三个不同的概率分布
        distributions = [
            {"name": "高置信度正确", "probs": [0.9, 0.05, 0.05], "color": GREEN},
            {"name": "低置信度正确", "probs": [0.4, 0.3, 0.3], "color": ORANGE},
            {"name": "错误预测", "probs": [0.1, 0.8, 0.1], "color": RED}
        ]
        
        for i, dist in enumerate(distributions):
            self.create_single_distribution(dist, i)
    
    def create_single_distribution(self, distribution, index):
        """创建单个概率分布"""
        # 位置
        x_offset = (index - 1) * 4
        
        # 标题
        title = Text(distribution["name"], color=distribution["color"]).scale(0.5)
        title.move_to([x_offset, 3, 0])
        self.play(Write(title))
        
        # 概率柱状图
        bars = VGroup()
        categories = ["猫", "狗", "鸟"]
        
        for j, (prob, category) in enumerate(zip(distribution["probs"], categories)):
            # 创建柱子
            bar_height = prob * 2
            bar = Rectangle(
                width=0.6,
                height=bar_height,
                fill_color=distribution["color"],
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=1
            )
            bar.move_to([x_offset + (j - 1) * 0.8, bar_height/2 - 1, 0])
            bars.add(bar)
            
            # 添加概率标签
            prob_text = Text(f"{prob:.2f}", color=WHITE).scale(0.3)
            prob_text.move_to(bar.get_top() + UP * 0.2)
            self.play(Write(prob_text))
            
            # 添加类别标签
            category_text = Text(category, color=self.text_color).scale(0.4)
            category_text.move_to([x_offset + (j - 1) * 0.8, -1.5, 0])
            self.add(category_text)
        
        self.play(FadeIn(bars), run_time=0.5)
        
        # 计算并显示交叉熵（假设正确答案是猫）
        true_label = [1, 0, 0]  # 猫是正确答案
        cross_entropy = -np.sum(true_label * np.log(distribution["probs"]))
        
        ce_text = Text(f"交叉熵: {cross_entropy:.3f}", color=distribution["color"]).scale(0.4)
        ce_text.move_to([x_offset, -2, 0])
        self.play(Write(ce_text))
    
    def demonstrate_cross_entropy(self):
        """演示交叉熵的直观含义"""
        # 创建说明文本
        explanation = Text(
            "正确类别概率越小，交叉熵越大（惩罚越重）",
            color=self.text_color
        ).scale(0.6)
        explanation.to_edge(DOWN, buff=1.0)
        self.play(Write(explanation))
        
        # 添加比喻
        metaphor = Text(
            "就像做选择题：知道答案是A，却选B还说有99%把握 → 该狠狠扣分！",
            color=self.primary_color
        ).scale(0.5)
        metaphor.next_to(explanation, DOWN, buff=0.5)
        self.play(Write(metaphor))
        
        self.wait(1)
        self.play(FadeOut(VGroup(explanation, metaphor)))
    
    def show_cross_entropy_formula(self):
        """显示交叉熵公式"""
        # 二分类公式
        binary_ce = MathTex(
            "L = -[y \log(p) + (1-y) \log(1-p)]",
            color=self.text_color
        ).scale(0.7)
        
        # 多分类公式
        multi_ce = MathTex(
            "L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)",
            color=self.text_color
        ).scale(0.7)
        
        # 排列公式
        binary_ce.to_corner(UL, buff=1.0)
        multi_ce.next_to(binary_ce, DOWN, buff=1.0)
        
        # 添加标签
        binary_label = Text("二分类", color=self.primary_color).scale(0.5)
        binary_label.next_to(binary_ce, LEFT, buff=0.5)
        
        multi_label = Text("多分类", color=self.primary_color).scale(0.5)
        multi_label.next_to(multi_ce, LEFT, buff=0.5)
        
        self.play(
            Write(binary_ce), Write(binary_label),
            Write(multi_ce), Write(multi_label)
        )

class GradientDescentScene(BaseScene):
    """梯度下降可视化场景"""
    
    def construct(self):
        title = self.add_title("梯度下降", "沿着最陡峭方向下山")
        
        # 创建损失函数曲面
        surface = self.create_loss_surface()
        self.add(surface)
        
        # 演示梯度下降过程
        self.demonstrate_gradient_descent()
        
        # 显示梯度下降公式
        self.show_gradient_formula()
        
        self.wait(2)
    
    def create_loss_surface(self):
        """创建损失函数曲面"""
        # 创建3D网格（在2D平面上模拟3D效果）
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": self.text_color, "stroke_width": 1}
        )
        
        # 创建等高线
        contours = VGroup()
        for radius in np.linspace(0.5, 2.5, 5):
            circle = Circle(radius=radius, color=self.grid_color, stroke_width=1, stroke_opacity=0.5)
            contours.add(circle)
        
        # 添加最低点标记
        minimum = Dot(point=[0, 0, 0], color=GREEN, radius=0.15)
        min_label = Text("最小值", color=GREEN).scale(0.4)
        min_label.next_to(minimum, DOWN, buff=0.2)
        
        return VGroup(axes, contours, minimum, min_label)
    
    def demonstrate_gradient_descent(self):
        """演示梯度下降过程"""
        # 起始点
        start_point = Dot(point=[2, 2, 0], color=RED, radius=0.1)
        self.play(FadeIn(start_point))
        
        # 梯度下降路径
        path_points = [
            [2, 2], [1.5, 1.5], [1.0, 1.0], [0.5, 0.5], [0.2, 0.2], [0, 0]
        ]
        
        # 创建路径线
        path_lines = VGroup()
        current_point = start_point.get_center()
        
        for i, next_pos in enumerate(path_points[1:]):
            next_point = [next_pos[0], next_pos[1], 0]
            path_line = Line(current_point, next_point, color=self.primary_color, stroke_width=2)
            path_lines.add(path_line)
            
            # 移动的点
            moving_dot = Dot(point=current_point, color=RED, radius=0.1)
            
            # 显示梯度方向
            gradient_arrow = Arrow(
                start=current_point,
                end=current_point + (np.array(next_point) - np.array(current_point)) * 1.2,
                color=ORANGE,
                stroke_width=2,
                max_stroke_width_to_length_ratio=10
            )
            
            # 步数标签
            step_text = Text(f"Step {i+1}", color=self.text_color).scale(0.4)
            step_text.next_to(moving_dot, UP, buff=0.3)
            
            self.play(
                Create(path_line),
                Create(gradient_arrow),
                Write(step_text),
                moving_dot.animate.move_to(next_point),
                run_time=0.8
            )
            
            self.play(FadeOut(gradient_arrow), FadeOut(step_text))
            current_point = next_point
        
        # 最终点
        final_dot = Dot(point=[0, 0, 0], color=GREEN, radius=0.15)
        self.play(FadeOut(moving_dot), FadeIn(final_dot))
        
        # 添加收敛标记
        convergence_text = Text("收敛！", color=GREEN).scale(0.6)
        convergence_text.next_to(final_dot, UP, buff=0.5)
        self.play(Write(convergence_text))
    
    def show_gradient_formula(self):
        """显示梯度下降公式"""
        # 梯度下降更新公式
        update_formula = MathTex(
            "\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            color=self.text_color
        ).scale(0.7)
        
        update_formula.to_corner(UR, buff=1.0)
        
        # 添加符号说明
        explanations = VGroup(
            Text("θ: 参数", color=self.text_color).scale(0.4),
            Text("η: 学习率", color=self.text_color).scale(0.4),
            Text("∇L: 梯度", color=self.text_color).scale(0.4)
        )
        explanations.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        explanations.next_to(update_formula, DOWN, buff=0.5)
        
        self.play(Write(update_formula))
        self.play(Write(explanations))