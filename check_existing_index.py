#!/usr/bin/env python3
"""
检查是否有已经生成好的索引
Check if there's an existing index
"""

from pathlib import Path
import sqlitedict

print("=" * 80)
print("检查现有索引")
print("Check Existing Index")
print("=" * 80)

deps_dir = Path("/home/yc/CpGPT/examples/dependencies")
metadata_db = deps_dir / "ensembl_metadata.db"

if not metadata_db.exists():
    print(f"\n✗ ensembl_metadata.db 不存在")
    exit(1)

print(f"\n✓ ensembl_metadata.db 存在")
print(f"  路径: {metadata_db}")
print(f"  大小: {metadata_db.stat().st_size / (1024*1024):.2f} MB")

# 加载索引
with sqlitedict.SqliteDict(metadata_db, autocommit=True) as db:
    metadata = dict(db)

print(f"\n物种列表:")
for species in metadata.keys():
    print(f"  - {species}")

if "homo_sapiens" not in metadata:
    print(f"\n✗ 没有 homo_sapiens 条目")
    exit(1)

print(f"\n✓ 找到 homo_sapiens")

hs = metadata["homo_sapiens"]
print(f"\n  科学名: {hs.get('scientific_name', 'N/A')}")
print(f"  基因组: {hs.get('assembly', 'N/A')}")

print(f"\n  DNA 语言模型:")
for llm in ["nucleotide-transformer-v2-500m-multi-species", "DNABERT-2-117M", "hyenadna-large-1m-seqlen-hf"]:
    if llm in hs:
        print(f"    - {llm}:")
        llm_data = hs[llm]
        if isinstance(llm_data, dict):
            for context_len, index_dict in llm_data.items():
                if isinstance(index_dict, dict):
                    print(f"      • {context_len}bp: {len(index_dict)} 个位置")
                else:
                    print(f"      • {context_len}bp: (不是字典)")
        else:
            print(f"      (不是字典)")

# 检查 2001bp 索引
if "nucleotide-transformer-v2-500m-multi-species" in hs:
    nt = hs["nucleotide-transformer-v2-500m-multi-species"]
    if 2001 in nt:
        index_2001 = nt[2001]
        print(f"\n✓ 找到 2001bp 索引！")
        print(f"  包含 {len(index_2001)} 个位置")
        
        # 显示前几个位置
        if len(index_2001) > 0:
            print(f"\n  前 5 个位置示例:")
            for i, (loc, idx) in enumerate(list(index_2001.items())[:5]):
                print(f"    {i+1}. {loc} -> 索引 {idx}")
    else:
        print(f"\n✗ 没有 2001bp 索引")
        print(f"  可用的上下文长度: {list(nt.keys())}")
else:
    print(f"\n✗ 没有 nucleotide-transformer 数据")

print("\n" + "=" * 80)

