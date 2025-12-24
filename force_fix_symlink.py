#!/usr/bin/env python3
"""
强制修复符号链接
Force fix symbolic link
"""

import os
import shutil
from pathlib import Path

print("=" * 80)
print("强制修复符号链接")
print("Force Fix Symbolic Link")
print("=" * 80)

# 尝试所有可能的位置
possible_deps_dirs = [
    Path("/home/yc/CpGPT/dependencies"),
    Path("/home/yc/CpGPT/examples/dependencies"),
    Path("./dependencies"),
    Path("../dependencies"),
]

print("\n1. 查找 dependencies 目录...")
deps_dir = None
for d in possible_deps_dirs:
    if d.exists():
        print(f"   ✓ 找到: {d.resolve()}")
        deps_dir = d.resolve()
        break
    else:
        print(f"   ✗ 不存在: {d}")

if deps_dir is None:
    print("\n✗ 错误: 找不到 dependencies 目录")
    print("请先运行: inferencer.download_dependencies(species='human')")
    exit(1)

print(f"\n使用 dependencies 目录: {deps_dir}")

# 检查源目录
print("\n2. 检查源目录...")
human_source = deps_dir / "human" / "dna_embeddings" / "homo_sapiens"

if not human_source.exists():
    print(f"   ✗ 源目录不存在: {human_source}")
    print("\n请先运行:")
    print("  python -c \"from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer; ")
    print("  inferencer = CpGPTInferencer(dependencies_dir='./dependencies'); ")
    print("  inferencer.download_dependencies(species='human', overwrite=False)\"")
    exit(1)

print(f"   ✓ 源目录存在: {human_source}")

# 列出源目录内容
print("\n   源目录内容:")
for item in human_source.iterdir():
    print(f"     - {item.name}")

# 创建目标目录
print("\n3. 创建目标目录...")
dna_embeddings_dir = deps_dir / "dna_embeddings"
dna_embeddings_dir.mkdir(parents=True, exist_ok=True)
print(f"   ✓ 目标目录: {dna_embeddings_dir}")

# 检查并删除旧的链接/目录
print("\n4. 检查现有的 homo_sapiens...")
homo_sapiens_link = dna_embeddings_dir / "homo_sapiens"

if homo_sapiens_link.exists() or homo_sapiens_link.is_symlink():
    print(f"   已存在: {homo_sapiens_link}")
    
    if homo_sapiens_link.is_symlink():
        print(f"   类型: 符号链接 → {homo_sapiens_link.resolve()}")
        print("   删除旧符号链接...")
        homo_sapiens_link.unlink()
    elif homo_sapiens_link.is_dir():
        print("   类型: 目录")
        response = input("   是否删除现有目录? (y/n): ")
        if response.lower() == 'y':
            print("   删除中...")
            shutil.rmtree(homo_sapiens_link)
        else:
            print("   保留现有目录，退出")
            exit(0)
else:
    print("   不存在，将创建新的")

# 创建符号链接
print("\n5. 创建符号链接...")

try:
    # 使用绝对路径
    homo_sapiens_link.symlink_to(human_source, target_is_directory=True)
    print(f"   ✓ 符号链接创建成功")
    print(f"   {homo_sapiens_link} → {human_source}")
    method = "symlink"
except (OSError, NotImplementedError) as e:
    print(f"   ✗ 符号链接失败: {e}")
    print("   尝试复制文件...")
    
    try:
        shutil.copytree(human_source, homo_sapiens_link, dirs_exist_ok=True)
        print(f"   ✓ 文件复制成功")
        method = "copy"
    except Exception as e:
        print(f"   ✗ 复制失败: {e}")
        exit(1)

# 验证
print("\n6. 验证...")

if homo_sapiens_link.exists():
    print(f"   ✓ homo_sapiens 存在")
    
    if homo_sapiens_link.is_symlink():
        print(f"   ✓ 类型: 符号链接")
        print(f"   ✓ 目标: {homo_sapiens_link.resolve()}")
    else:
        print(f"   ✓ 类型: 目录（复制）")
    
    # 检查 nucleotide-transformer
    nt_dir = homo_sapiens_link / "nucleotide-transformer-v2-500m-multi-species"
    if nt_dir.exists():
        print(f"   ✓ nucleotide-transformer-v2-500m-multi-species 可访问")
        
        # 列出内容
        print("\n   目录内容:")
        for item in nt_dir.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"     - {item.name} ({size_mb:.1f} MB)")
            else:
                print(f"     - {item.name}/ (目录)")
    else:
        print(f"   ✗ nucleotide-transformer-v2-500m-multi-species 不可访问")
        exit(1)
else:
    print(f"   ✗ homo_sapiens 不存在")
    exit(1)

print("\n" + "=" * 80)
print("✓ 修复完成！")
print("✓ Fix completed!")
print("=" * 80)

print("\n现在可以运行预测脚本:")
print("Now you can run the prediction script:")
print("  cd /home/yc/CpGPT")
print("  python examples/935k_simple_prediction.py")
print()

