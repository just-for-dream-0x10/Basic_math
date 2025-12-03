from manim import *
from config import COLORS, MANIM_CONFIG
from utils import get_module_color, format_math_label

class BaseScene(Scene):
    """基础场景类，提供通用的样式和功能"""
    
    def __init__(self, module_name="default", **kwargs):
        # 移除module_name参数，因为Scene类不接受它
        scene_kwargs = {k: v for k, v in kwargs.items() if k != 'module_name'}
        super().__init__(**scene_kwargs)
        self.module_name = module_name
        self.primary_color = get_module_color(module_name)
        self.text_color = COLORS["text"]
        self.grid_color = COLORS["grid"]
        self.background_color = COLORS["background"]
        
        # 设置背景
        self.camera.background_color = self.background_color
    
    def add_title(self, title, subtitle=""):
        """添加标题和副标题 - 改为左上角避免遮挡"""
        title_obj = Text(title, color=self.text_color, font_size=48).scale(0.8)
        if subtitle:
            subtitle_obj = Text(subtitle, color=self.primary_color, font_size=32).scale(0.7)
            subtitle_obj.next_to(title_obj, DOWN, buff=0.3)
            title_group = VGroup(title_obj, subtitle_obj)
        else:
            title_group = title_obj
            
        title_group.to_corner(UL, buff=1.0)
        self.add(title_group)
        return title_group
    
    def add_grid(self):
        """添加坐标网格"""
        grid = NumberPlane(
            background_line_style={
                "stroke_color": self.grid_color,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={
                "stroke_color": self.text_color,
                "stroke_width": 2,
                "stroke_opacity": 0.8
            }
        )
        self.add(grid)
        return grid
    
    def create_math_text(self, text, color=None, font_size=36):
        """创建数学文本"""
        if color is None:
            color = self.primary_color
        return MathTex(text, color=color, font_size=font_size)
    
    def create_text(self, text, color=None, font_size=36):
        """创建普通文本"""
        if color is None:
            color = self.text_color
        return Text(text, color=color, font_size=font_size)
    
    def highlight_element(self, element, duration=1.0):
        """高亮元素"""
        highlight = SurroundingRectangle(
            element, 
            color=self.primary_color,
            fill_opacity=0.1,
            stroke_width=2
        )
        self.play(Create(highlight), run_time=duration/2)
        self.wait(duration/2)
        self.play(FadeOut(highlight))
        return highlight
    
    def fade_in_element(self, element, duration=1.0):
        """淡入元素"""
        self.play(FadeIn(element), run_time=duration)
        return element
    
    def fade_out_element(self, element, duration=1.0):
        """淡出元素"""
        self.play(FadeOut(element), run_time=duration)
        return element
    
    def move_element(self, element, target_position, duration=1.0):
        """移动元素到目标位置"""
        self.play(element.animate.move_to(target_position), run_time=duration)
        return element
    
    def transform_element(self, element, target_element, duration=1.0):
        """变换元素为目标元素"""
        self.play(Transform(element, target_element), run_time=duration)
        return element
    
    def add_annotation(self, target, text, direction=RIGHT, arrow_scale=0.5):
        """添加注释"""
        arrow = Arrow(
            start=target.get_edge_center(direction),
            end=target.get_edge_center(direction) + direction * 1.5,
            color=self.primary_color,
            stroke_width=2,
            max_stroke_width_to_length_ratio=10,
            buff=0.1
        ).scale(arrow_scale)
        
        annotation = Text(text, color=self.text_color, font_size=24).scale(0.6)
        annotation.next_to(arrow.get_end(), direction, buff=0.2)
        
        self.play(Create(arrow), Write(annotation))
        return VGroup(arrow, annotation)
    
    def create_progress_indicator(self, total_steps, current_step=0):
        """创建进度指示器"""
        indicators = VGroup()
        for i in range(total_steps):
            circle = Circle(radius=0.1, color=self.grid_color, fill_opacity=0.3)
            if i < current_step:
                circle.set_color(self.primary_color)
                circle.set_fill(self.primary_color, opacity=0.8)
            indicators.add(circle)
        
        indicators.arrange(RIGHT, buff=0.3)
        indicators.to_edge(DOWN, buff=1.0)
        indicators.to_edge(RIGHT, buff=1.0)
        
        self.add(indicators)
        return indicators
    
    def update_progress(self, indicators, current_step):
        """更新进度指示器"""
        for i, circle in enumerate(indicators):
            if i <= current_step:
                circle.set_color(self.primary_color)
                circle.set_fill(self.primary_color, opacity=0.8)
            else:
                circle.set_color(self.grid_color)
                circle.set_fill(self.grid_color, opacity=0.3)
        
        return indicators

class InteractiveScene(BaseScene):
    """交互式场景基类，支持更复杂的动画"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interactive_elements = []
    
    def add_interactive_element(self, element, callback=None):
        """添加交互元素"""
        self.interactive_elements.append({
            "element": element,
            "callback": callback
        })
    
    def create_button(self, text, position=ORIGIN, size=(2, 1)):
        """创建按钮"""
        button = RoundedRectangle(
            width=size[0],
            height=size[1],
            corner_radius=0.2,
            fill_color=self.primary_color,
            fill_opacity=0.8,
            stroke_color=self.text_color,
            stroke_width=2
        )
        button.move_to(position)
        
        label = Text(text, color=self.text_color, font_size=24).scale(0.6)
        label.move_to(position)
        
        button_group = VGroup(button, label)
        return button_group
    
    def create_slider(self, min_val=0, max_val=1, initial_val=0.5, position=ORIGIN):
        """创建滑块"""
        # 滑轨
        track = Line(
            start=position + LEFT * 2,
            end=position + RIGHT * 2,
            color=self.grid_color,
            stroke_width=4
        )
        
        # 滑块
        slider = Circle(
            radius=0.2,
            color=self.primary_color,
            fill_opacity=1.0
        )
        slider_pos = position + LEFT * 2 + (initial_val - min_val) / (max_val - min_val) * RIGHT * 4
        slider.move_to(slider_pos)
        
        return VGroup(track, slider, Text(f"{initial_val:.2f}", color=self.text_color).scale(0.5).next_to(track, DOWN))