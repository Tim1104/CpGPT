#!/usr/bin/env python3
"""
诊断路径和目录结构
Diagnose paths and directory structure
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("路径诊断 / Path Diagnosis")
print("=" * 80)

# 1. 当前工作目录
print("\n1. 当前工作目录 / Current working directory:")
print(f"   {os.getcwd()}")

# 2. 脚本位置
print("\n2. 脚本位置 / Script location:")
script_path = Path(__file__).resolve()
print(f"   脚本文件: {script_path}")
print(f"   脚本目录: {script_path.parent}")
print(f"   项目根目录: {script_path.parent.parent if 'examples' in str(script_path) else script_path.parent}")

# 3. 检查 dependencies 目录
print("\n3. 检查 dependencies 目录 / Check dependencies directories:")

deps_locations = [
    "./dependencies",
    "../dependencies",
    "./examples/dependencies",
    "/home/yc/CpGPT/dependencies",
    "/home/yc/CpGPT/examples/dependencies",
]

for loc in deps_locations:
    path = Path(loc)
    exists = "✓ 存在" if path.exists() else "✗ 不存在"
    print(f"   {loc:40s} {exists}")
    if path.exists():
        # 检查子目录
        for subdir in ["human", "dna_embeddings", "model"]:
            subpath = path / subdir
            if subpath.exists():
                print(f"     └─ {subdir}/ ✓")

# 4. 检查 human/dna_embeddings/homo_sapiens
print("\n4. 检查 human/dna_embeddings/homo_sapiens:")

human_locations = [
    "./dependencies/human/dna_embeddings/homo_sapiens",
    "../dependencies/human/dna_embeddings/homo_sapiens",
    "./examples/dependencies/human/dna_embeddings/homo_sapiens",
    "/home/yc/CpGPT/dependencies/human/dna_embeddings/homo_sapiens",
    "/home/yc/CpGPT/examples/dependencies/human/dna_embeddings/homo_sapiens",
]

for loc in human_locations:
    path = Path(loc)
    exists = "✓ 存在" if path.exists() else "✗ 不存在"
    print(f"   {loc:70s} {exists}")
    if path.exists():
        # 检查 nucleotide-transformer
        nt_path = path / "nucleotide-transformer-v2-500m-multi-species"
        if nt_path.exists():
            print(f"     └─ nucleotide-transformer-v2-500m-multi-species/ ✓")

# 5. 检查 dna_embeddings/homo_sapiens（符号链接）
print("\n5. 检查 dna_embeddings/homo_sapiens (符号链接):")

link_locations = [
    "./dependencies/dna_embeddings/homo_sapiens",
    "../dependencies/dna_embeddings/homo_sapiens",
    "./examples/dependencies/dna_embeddings/homo_sapiens",
    "/home/yc/CpGPT/dependencies/dna_embeddings/homo_sapiens",
    "/home/yc/CpGPT/examples/dependencies/dna_embeddings/homo_sapiens",
]

for loc in link_locations:
    path = Path(loc)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            target = path.resolve()
            print(f"   {loc:70s} ✓ 符号链接 → {target}")
        else:
            print(f"   {loc:70s} ✓ 目录")
        
        # 检查 nucleotide-transformer
        nt_path = path / "nucleotide-transformer-v2-500m-multi-species"
        if nt_path.exists():
            print(f"     └─ nucleotide-transformer-v2-500m-multi-species/ ✓")
    else:
        print(f"   {loc:70s} ✗ 不存在")

# 6. 查找所有 nucleotide-transformer 目录
print("\n6. 查找所有 nucleotide-transformer 目录:")
print("   搜索中...")

search_roots = [
    Path("."),
    Path(".."),
    Path("/home/yc/CpGPT") if Path("/home/yc/CpGPT").exists() else None,
]

found_paths = []
for root in search_roots:
    if root is None:
        continue
    try:
        for path in root.rglob("nucleotide-transformer-v2-500m-multi-species"):
            if path.is_dir():
                found_paths.append(path.resolve())
    except Exception as e:
        print(f"   搜索 {root} 时出错: {e}")

if found_paths:
    for p in set(found_paths):
        print(f"   ✓ {p}")
else:
    print("   ✗ 未找到")

# 7. 模拟脚本的路径检测
print("\n7. 模拟 935k_simple_prediction.py 的路径检测:")

# 假设脚本在 examples/ 目录
script_dir = Path("/home/yc/CpGPT/examples").resolve()
project_root = script_dir.parent

print(f"   SCRIPT_DIR = {script_dir}")
print(f"   PROJECT_ROOT = {project_root}")

deps_dir = script_dir / "dependencies"
if not deps_dir.exists():
    deps_dir = project_root / "dependencies"

print(f"   DEPENDENCIES_DIR = {deps_dir}")
print(f"   存在: {'✓' if deps_dir.exists() else '✗'}")

if deps_dir.exists():
    human_source = deps_dir / "human" / "dna_embeddings" / "homo_sapiens"
    print(f"   human_source = {human_source}")
    print(f"   存在: {'✓' if human_source.exists() else '✗'}")
    
    homo_sapiens_link = deps_dir / "dna_embeddings" / "homo_sapiens"
    print(f"   homo_sapiens_link = {homo_sapiens_link}")
    print(f"   存在: {'✓' if homo_sapiens_link.exists() else '✗'}")
    
    if homo_sapiens_link.exists() or homo_sapiens_link.is_symlink():
        if homo_sapiens_link.is_symlink():
            print(f"   类型: 符号链接 → {homo_sapiens_link.resolve()}")
        else:
            print(f"   类型: 目录")

print("\n" + "=" * 80)
print("诊断完成 / Diagnosis completed")
print("=" * 80)

