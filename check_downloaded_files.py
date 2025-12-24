#!/usr/bin/env python3
"""
检查下载的文件
Check downloaded files
"""

from pathlib import Path
import json

print("=" * 80)
print("检查下载的文件")
print("Check Downloaded Files")
print("=" * 80)

deps_dir = Path("/home/yc/CpGPT/examples/dependencies")
human_dir = deps_dir / "human"

print(f"\n1. 检查 human 目录结构")
print(f"   路径: {human_dir}")

if not human_dir.exists():
    print(f"   ✗ 目录不存在")
    exit(1)

print(f"   ✓ 目录存在")

# 递归列出所有文件
print(f"\n2. 列出所有文件:")
for item in sorted(human_dir.rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(human_dir)
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"   {str(rel_path):70s} {size_mb:10.1f} MB")

# 检查是否有 metadata.json 或索引文件
print(f"\n3. 查找 metadata 或索引文件:")
metadata_files = list(human_dir.rglob("*metadata*"))
index_files = list(human_dir.rglob("*index*"))
json_files = list(human_dir.rglob("*.json"))

if metadata_files:
    print(f"   找到 {len(metadata_files)} 个 metadata 文件:")
    for f in metadata_files:
        print(f"     - {f.relative_to(human_dir)}")
        if f.suffix == '.json':
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                print(f"       内容: {json.dumps(data, indent=2)[:200]}...")
            except Exception as e:
                print(f"       读取失败: {e}")
else:
    print(f"   ✗ 没有找到 metadata 文件")

if index_files:
    print(f"\n   找到 {len(index_files)} 个 index 文件:")
    for f in index_files:
        print(f"     - {f.relative_to(human_dir)}")
else:
    print(f"\n   ✗ 没有找到 index 文件")

if json_files:
    print(f"\n   找到 {len(json_files)} 个 JSON 文件:")
    for f in json_files:
        print(f"     - {f.relative_to(human_dir)}")
else:
    print(f"\n   ✗ 没有找到 JSON 文件")

# 检查 nucleotide-transformer 目录
print(f"\n4. 检查 nucleotide-transformer 目录:")
nt_dir = human_dir / "dna_embeddings" / "homo_sapiens" / "nucleotide-transformer-v2-500m-multi-species"

if nt_dir.exists():
    print(f"   路径: {nt_dir}")
    print(f"   内容:")
    for item in sorted(nt_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"     - {item.name:50s} {size_mb:10.1f} MB")
        else:
            print(f"     - {item.name}/ (目录)")
else:
    print(f"   ✗ 目录不存在")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)

