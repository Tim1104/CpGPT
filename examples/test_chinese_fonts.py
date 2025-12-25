#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试和修复matplotlib中文字体问题
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import sys

print("=" * 60)
print("matplotlib中文字体测试")
print("=" * 60)
print()

# 1. 检查matplotlib版本
print(f"[1] matplotlib版本: {matplotlib.__version__}")
print()

# 2. 检查字体缓存目录
cache_dir = Path(fm.get_cachedir())
print(f"[2] 字体缓存目录: {cache_dir}")
print(f"    缓存目录存在: {cache_dir.exists()}")
print()

# 3. 清除字体缓存
print("[3] 清除字体缓存...")
try:
    if cache_dir.exists():
        import shutil
        for cache_file in cache_dir.glob('*.cache'):
            try:
                cache_file.unlink()
                print(f"    ✓ 删除: {cache_file.name}")
            except Exception as e:
                print(f"    ✗ 删除失败: {cache_file.name} - {e}")
    print("    ✓ 缓存清除完成")
except Exception as e:
    print(f"    ✗ 缓存清除失败: {e}")
print()

# 4. 重新构建字体列表
print("[4] 重新构建字体列表...")
try:
    fm._rebuild()
    print("    ✓ 字体列表重建完成")
except Exception as e:
    print(f"    ✗ 字体列表重建失败: {e}")
print()

# 5. 检查系统中文字体文件
print("[5] 检查系统中文字体文件...")
font_paths = [
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
]

found_fonts = []
for font_path in font_paths:
    if Path(font_path).exists():
        print(f"    ✓ 找到: {font_path}")
        found_fonts.append(font_path)
    else:
        print(f"    ✗ 未找到: {font_path}")
print()

# 6. 检查matplotlib可用的中文字体
print("[6] 检查matplotlib可用的中文字体...")
chinese_fonts = [
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Noto Sans CJK SC',
    'Noto Sans CJK',
    'AR PL UMing CN',
    'Droid Sans Fallback',
    'SimHei',
    'STHeiti',
]

available_fonts = [f.name for f in fm.fontManager.ttflist]
found_mpl_fonts = []

for font in chinese_fonts:
    if font in available_fonts:
        print(f"    ✓ 可用: {font}")
        found_mpl_fonts.append(font)
    else:
        print(f"    ✗ 不可用: {font}")
print()

# 7. 测试字体
print("[7] 测试中文字体渲染...")

if found_fonts:
    # 使用字体文件路径
    test_font_path = found_fonts[0]
    print(f"    使用字体文件: {test_font_path}")
    
    try:
        from matplotlib.font_manager import FontProperties
        font_prop = FontProperties(fname=test_font_path)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, '中文测试 Chinese Test 123', 
                fontproperties=font_prop, fontsize=20, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        test_path = 'test_chinese_font.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ 测试图片已生成: {test_path}")
        print(f"    请打开图片检查中文是否正常显示")
    except Exception as e:
        print(f"    ✗ 测试失败: {e}")
        
elif found_mpl_fonts:
    # 使用字体名称
    test_font = found_mpl_fonts[0]
    print(f"    使用字体名称: {test_font}")
    
    try:
        plt.rcParams['font.sans-serif'] = [test_font]
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, '中文测试 Chinese Test 123', 
                fontsize=20, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        test_path = 'test_chinese_font.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ 测试图片已生成: {test_path}")
        print(f"    请打开图片检查中文是否正常显示")
    except Exception as e:
        print(f"    ✗ 测试失败: {e}")
else:
    print("    ✗ 未找到任何中文字体")
    print()
    print("=" * 60)
    print("建议安装中文字体：")
    print("=" * 60)
    print("sudo apt-get update")
    print("sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk")
    print("rm -rf ~/.cache/matplotlib")
    print()
    sys.exit(1)

print()
print("=" * 60)
print("✅ 测试完成")
print("=" * 60)
print()

if found_fonts or found_mpl_fonts:
    print("推荐配置：")
    if found_fonts:
        print(f"  字体文件路径: {found_fonts[0]}")
    if found_mpl_fonts:
        print(f"  字体名称: {found_mpl_fonts[0]}")
    print()
    print("现在可以运行主脚本：")
    print("  python 935k_enhanced_prediction.py")
else:
    print("⚠️  未找到可用的中文字体")
    print("请先安装中文字体，然后重新运行此测试脚本")

print()

