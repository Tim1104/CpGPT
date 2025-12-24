#!/usr/bin/env python3
"""
修复 examples/dependencies 下的符号链接
Fix symbolic link under examples/dependencies
"""

import shutil
from pathlib import Path

print("=" * 80)
print("修复 examples/dependencies 符号链接")
print("Fix examples/dependencies symbolic link")
print("=" * 80)

# 使用绝对路径
examples_deps = Path("/home/yc/CpGPT/examples/dependencies")
human_source = examples_deps / "human" / "dna_embeddings" / "homo_sapiens"
dna_embeddings_dir = examples_deps / "dna_embeddings"
homo_sapiens_link = dna_embeddings_dir / "homo_sapiens"

print(f"\n1. 检查源目录...")
print(f"   源目录: {human_source}")
if not human_source.exists():
    print(f"   ✗ 源目录不存在！")
    print(f"\n请先运行:")
    print(f"  cd /home/yc/CpGPT")
    print(f"  python examples/935k_simple_prediction.py")
    print(f"\n让它下载依赖到 examples/dependencies/")
    exit(1)

print(f"   ✓ 源目录存在")

# 检查 nucleotide-transformer
nt_source = human_source / "nucleotide-transformer-v2-500m-multi-species"
if not nt_source.exists():
    print(f"   ✗ nucleotide-transformer 不存在: {nt_source}")
    exit(1)

print(f"   ✓ nucleotide-transformer 存在")

print(f"\n2. 创建目标目录...")
dna_embeddings_dir.mkdir(parents=True, exist_ok=True)
print(f"   ✓ {dna_embeddings_dir}")

print(f"\n3. 检查现有链接...")
if homo_sapiens_link.exists() or homo_sapiens_link.is_symlink():
    print(f"   已存在: {homo_sapiens_link}")
    if homo_sapiens_link.is_symlink():
        print(f"   类型: 符号链接 → {homo_sapiens_link.resolve()}")
        print(f"   删除旧符号链接...")
        homo_sapiens_link.unlink()
    else:
        print(f"   类型: 目录")
        print(f"   删除旧目录...")
        shutil.rmtree(homo_sapiens_link)
else:
    print(f"   不存在，将创建新的")

print(f"\n4. 创建符号链接...")
try:
    homo_sapiens_link.symlink_to(human_source, target_is_directory=True)
    print(f"   ✓ 符号链接创建成功")
    print(f"   {homo_sapiens_link}")
    print(f"   → {human_source}")
except Exception as e:
    print(f"   ✗ 符号链接失败: {e}")
    print(f"   尝试复制文件...")
    shutil.copytree(human_source, homo_sapiens_link, dirs_exist_ok=True)
    print(f"   ✓ 文件复制成功")

print(f"\n5. 验证...")
nt_link = homo_sapiens_link / "nucleotide-transformer-v2-500m-multi-species"
if nt_link.exists():
    print(f"   ✓ nucleotide-transformer 可访问")
    print(f"   {nt_link}")
    
    # 列出文件
    print(f"\n   目录内容:")
    for item in nt_link.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"     - {item.name} ({size_mb:.1f} MB)")
        else:
            print(f"     - {item.name}/ (目录)")
else:
    print(f"   ✗ nucleotide-transformer 不可访问")
    exit(1)

print("\n" + "=" * 80)
print("✓ 修复完成！")
print("=" * 80)
print("\n现在可以运行:")
print("  cd /home/yc/CpGPT")
print("  python examples/935k_simple_prediction.py")
print()

