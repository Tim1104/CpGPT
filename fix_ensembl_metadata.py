#!/usr/bin/env python3
"""
修复 ensembl_metadata.db
Fix ensembl_metadata.db
"""

from pathlib import Path
import sqlitedict
import json

print("=" * 80)
print("修复 ensembl_metadata.db")
print("Fix ensembl_metadata.db")
print("=" * 80)

deps_dir = Path("/home/yc/CpGPT/examples/dependencies")
metadata_db = deps_dir / "ensembl_metadata.db"

print(f"\n1. 检查 ensembl_metadata.db")
print(f"   路径: {metadata_db}")

if not metadata_db.exists():
    print(f"   ✗ 文件不存在")
    print(f"\n   需要重新下载依赖")
    exit(1)

print(f"   ✓ 文件存在")

print(f"\n2. 读取 ensembl_metadata.db")
try:
    with sqlitedict.SqliteDict(metadata_db, autocommit=True) as db:
        metadata = dict(db)
    print(f"   ✓ 成功读取")
    print(f"   包含 {len(metadata)} 个物种")
except Exception as e:
    print(f"   ✗ 读取失败: {e}")
    exit(1)

print(f"\n3. 检查 homo_sapiens 条目")
if "homo_sapiens" not in metadata:
    print(f"   ✗ homo_sapiens 不存在")
    exit(1)

print(f"   ✓ homo_sapiens 存在")

homo_sapiens = metadata["homo_sapiens"]
print(f"\n   homo_sapiens 结构:")
print(f"   类型: {type(homo_sapiens)}")

if isinstance(homo_sapiens, dict):
    print(f"   键: {list(homo_sapiens.keys())}")
    
    # 检查 nucleotide-transformer
    dna_llm = "nucleotide-transformer-v2-500m-multi-species"
    if dna_llm in homo_sapiens:
        print(f"\n   {dna_llm}:")
        print(f"   类型: {type(homo_sapiens[dna_llm])}")
        print(f"   值: {homo_sapiens[dna_llm]}")
        
        # 检查是否是字典
        if isinstance(homo_sapiens[dna_llm], dict):
            print(f"   键: {list(homo_sapiens[dna_llm].keys())}")
            
            # 检查 2001
            if 2001 in homo_sapiens[dna_llm]:
                print(f"\n   2001 条目:")
                print(f"   类型: {type(homo_sapiens[dna_llm][2001])}")
                if isinstance(homo_sapiens[dna_llm][2001], dict):
                    print(f"   包含 {len(homo_sapiens[dna_llm][2001])} 个位置")
                else:
                    print(f"   值: {homo_sapiens[dna_llm][2001]}")
            else:
                print(f"   ✗ 2001 不存在")
        else:
            print(f"   ✗ 错误: {dna_llm} 应该是字典，但是是 {type(homo_sapiens[dna_llm])}")
            print(f"\n   这是问题所在！需要重新下载依赖")
    else:
        print(f"   ✗ {dna_llm} 不存在")
else:
    print(f"   ✗ 错误: homo_sapiens 应该是字典，但是是 {type(homo_sapiens)}")

print(f"\n4. 建议")
print(f"   删除损坏的文件并重新下载:")
print(f"   rm {metadata_db}")
print(f"   rm {metadata_db}.tmp 2>/dev/null || true")
print(f"   rm -rf {deps_dir}/human")
print(f"   python -c \"from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer; inferencer = CpGPTInferencer(dependencies_dir='{deps_dir}'); inferencer.download_dependencies(species='human', overwrite=True)\"")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)

