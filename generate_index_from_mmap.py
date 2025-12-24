#!/usr/bin/env python3
"""
从 .mmap 文件生成索引
Generate index from .mmap file

注意：这个方法只有在有对应的位置列表文件时才能工作
"""

from pathlib import Path
import numpy as np
import json

print("=" * 80)
print("从 .mmap 文件生成索引")
print("Generate Index from .mmap File")
print("=" * 80)

deps_dir = Path("/home/yc/CpGPT/examples/dependencies")

print(f"\n问题分析:")
print(f"  下载的 .mmap 文件是预先生成的嵌入")
print(f"  但是没有对应的索引（位置 → 嵌入索引的映射）")
print(f"  索引通常在生成嵌入时创建")
print(f"")
print(f"  可能的解决方案:")
print(f"  1. 查找是否有预先生成的索引文件")
print(f"  2. 从头生成嵌入（需要几小时）")
print(f"  3. 使用不同的依赖目录（如果有完整的）")
print(f"")

# 检查是否有位置列表文件
print(f"\n1. 查找可能的索引/位置文件:")
human_dir = deps_dir / "human"

possible_index_files = [
    "dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/2001bp_locations.json",
    "dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/2001bp_index.json",
    "dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/metadata.json",
    "dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/locations.txt",
    "dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/index.txt",
]

found_files = []
for rel_path in possible_index_files:
    full_path = human_dir / rel_path
    if full_path.exists():
        print(f"  ✓ 找到: {rel_path}")
        found_files.append(full_path)
    else:
        print(f"  ✗ 不存在: {rel_path}")

if not found_files:
    print(f"\n  ✗ 没有找到索引文件")
    print(f"\n建议:")
    print(f"  1. 检查 S3 上是否有索引文件")
    print(f"  2. 或者使用项目根目录的 dependencies（如果有）")
    print(f"  3. 或者从头生成嵌入（需要很长时间）")
    print(f"")
    print(f"  让我检查项目根目录的 dependencies...")
    
    # 检查项目根目录
    root_deps = Path("/home/yc/CpGPT/dependencies")
    if root_deps.exists():
        print(f"\n  ✓ 找到项目根目录的 dependencies: {root_deps}")
        print(f"  检查是否有完整的索引...")
        
        root_metadata = root_deps / "ensembl_metadata.db"
        if root_metadata.exists():
            print(f"  ✓ 找到 ensembl_metadata.db")
            
            # 检查索引
            import sqlitedict
            with sqlitedict.SqliteDict(root_metadata, autocommit=True) as db:
                metadata = dict(db)
            
            if "homo_sapiens" in metadata:
                hs = metadata["homo_sapiens"]
                if "nucleotide-transformer-v2-500m-multi-species" in hs:
                    nt = hs["nucleotide-transformer-v2-500m-multi-species"]
                    if 2001 in nt:
                        index_size = len(nt[2001])
                        print(f"  ✓ 找到完整的索引！包含 {index_size} 个位置")
                        print(f"\n  建议: 复制项目根目录的 dependencies 到 examples/")
                        print(f"  或者修改脚本使用项目根目录的 dependencies")
                    else:
                        print(f"  ✗ 索引不完整（没有 2001）")
                else:
                    print(f"  ✗ 没有 nucleotide-transformer 索引")
            else:
                print(f"  ✗ 没有 homo_sapiens 条目")
        else:
            print(f"  ✗ 没有 ensembl_metadata.db")
    else:
        print(f"  ✗ 项目根目录的 dependencies 不存在")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)

print("\n最终建议:")
print("  CpGPT 的预先生成的嵌入文件似乎不包含索引")
print("  你有以下选择:")
print("")
print("  选项 1: 使用项目根目录的 dependencies（如果有完整索引）")
print("    修改 935k_simple_prediction.py:")
print("    DEPENDENCIES_DIR = PROJECT_ROOT / 'dependencies'")
print("")
print("  选项 2: 从头生成嵌入（需要几小时，需要 GPU）")
print("    这需要下载基因组文件和 DNA 语言模型")
print("")
print("  选项 3: 联系 CpGPT 开发者")
print("    询问是否有预先生成的索引文件")
print("")

