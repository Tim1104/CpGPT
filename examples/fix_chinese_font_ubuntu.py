#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ubuntu中文字体一键修复脚本
自动安装字体、清除缓存、测试字体
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行shell命令"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    print("=" * 60)
    print("Ubuntu中文字体一键修复脚本")
    print("=" * 60)
    
    # 1. 检查是否为root
    if subprocess.run("id -u", shell=True, capture_output=True).stdout.decode().strip() != "0":
        print("\n⚠️  需要sudo权限来安装字体")
        print("请使用: sudo python3 fix_chinese_font_ubuntu.py")
        sys.exit(1)
    
    # 2. 更新软件包列表
    run_command("apt-get update", "[1/6] 更新软件包列表")
    
    # 3. 安装文泉驿微米黑
    run_command("apt-get install -y fonts-wqy-microhei", "[2/6] 安装文泉驿微米黑字体")
    
    # 4. 安装Noto CJK
    run_command("apt-get install -y fonts-noto-cjk", "[3/6] 安装Noto CJK字体")
    
    # 5. 更新字体缓存
    run_command("fc-cache -fv", "[4/6] 更新系统字体缓存")
    
    # 6. 清除matplotlib缓存
    print("\n" + "="*60)
    print("[5/6] 清除matplotlib字体缓存")
    print("="*60)
    
    cache_dirs = [
        "/root/.cache/matplotlib",
        "/root/.matplotlib",
    ]
    
    # 清除所有用户的缓存
    import os
    for home_dir in Path("/home").glob("*"):
        if home_dir.is_dir():
            cache_dirs.append(str(home_dir / ".cache/matplotlib"))
            cache_dirs.append(str(home_dir / ".matplotlib"))
    
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                import shutil
                shutil.rmtree(cache_path)
                print(f"  ✓ 已删除: {cache_dir}")
            except Exception as e:
                print(f"  ✗ 删除失败: {cache_dir} - {e}")
    
    # 7. 验证字体
    print("\n" + "="*60)
    print("[6/6] 验证中文字体")
    print("="*60)
    
    result = subprocess.run("fc-list :lang=zh | grep -E 'WenQuanYi|Noto'", 
                          shell=True, capture_output=True, text=True)
    if result.stdout:
        print("已安装的中文字体:")
        for line in result.stdout.strip().split('\n')[:5]:
            print(f"  ✓ {line}")
    
    print("\n" + "="*60)
    print("✅ 字体安装完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 以普通用户身份运行测试脚本:")
    print("   python3 test_chinese_fonts.py")
    print("\n2. 如果测试通过，运行主脚本:")
    print("   python3 935k_enhanced_prediction.py")
    print()

if __name__ == "__main__":
    main()

