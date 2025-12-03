"""
字体工具 - 自动检测系统中文字体
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os


class FontManager:
    """中文字体管理器"""
    
    _chinese_font = None
    
    @classmethod
    def get_chinese_font(cls):
        """获取系统支持的中文字体"""
        if cls._chinese_font is not None:
            return cls._chinese_font
        
        system = platform.system()
        
        # 不同系统的中文字体列表（按优先级）
        font_candidates = {
            'Windows': [
                'Microsoft YaHei',
                'SimHei',
                'SimSun',
                'FangSong',
                'KaiTi',
            ],
            'Darwin': [  # macOS
                'PingFang SC',
                'Heiti SC',
                'STHeiti',
                'Songti SC',
                'STSong',
                'Arial Unicode MS',
            ],
            'Linux': [
                'Noto Sans CJK SC',
                'WenQuanYi Micro Hei',
                'WenQuanYi Zen Hei',
                'Droid Sans Fallback',
                'AR PL UMing CN',
                'AR PL UKai CN',
            ]
        }
        
        # 获取当前系统的候选字体
        candidates = font_candidates.get(system, [])
        
        # 获取系统所有可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 查找第一个可用的中文字体
        for font_name in candidates:
            if font_name in available_fonts:
                cls._chinese_font = font_name
                return font_name
        
        # 如果没找到预定义的字体，尝试查找任何包含中文的字体
        for font in fm.fontManager.ttflist:
            font_name = font.name
            # 检查字体名称是否包含中文相关关键词
            if any(keyword in font_name.lower() for keyword in 
                   ['chinese', 'cjk', 'han', 'hei', 'song', 'kai', 'fang']):
                cls._chinese_font = font_name
                return font_name
        
        # 最后的fallback
        print("警告：未找到合适的中文字体，使用默认字体（可能显示为方块）")
        cls._chinese_font = 'DejaVu Sans'
        return cls._chinese_font
    
    @classmethod
    def setup_matplotlib(cls):
        """配置matplotlib使用中文字体"""
        font_name = cls.get_chinese_font()
        
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        return font_name
    
    @classmethod
    def get_font_properties(cls, size=12):
        """获取字体属性对象"""
        font_name = cls.get_chinese_font()
        return fm.FontProperties(family=font_name, size=size)


def configure_chinese_font():
    """快捷函数：配置中文字体"""
    return FontManager.setup_matplotlib()


def get_system_info():
    """获取系统信息（用于调试）"""
    system = platform.system()
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in available_fonts if any(
        keyword in f.lower() for keyword in 
        ['chinese', 'cjk', 'han', 'hei', 'song', 'kai', 'fang', 'microsoft', 'pingfang', 'noto']
    )]
    
    return {
        'system': system,
        'total_fonts': len(available_fonts),
        'chinese_fonts': chinese_fonts[:10],  # 只显示前10个
        'selected_font': FontManager.get_chinese_font()
    }


if __name__ == "__main__":
    # 测试
    import json
    
    print("系统字体信息：")
    info = get_system_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    print(f"\n选择的中文字体: {configure_chinese_font()}")
    
    # 测试绘图
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y)
    ax.set_xlabel('横坐标 (测试中文)', fontsize=12)
    ax.set_ylabel('纵坐标 (测试中文)', fontsize=12)
    ax.set_title('中文标题测试 - 正弦函数', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.savefig('test_chinese_font.png', dpi=100, bbox_inches='tight')
    print("\n已保存测试图片: test_chinese_font.png")
