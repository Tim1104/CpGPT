#!/usr/bin/env python3
"""
强制重建嵌入索引
Force rebuild embedding index
"""

from pathlib import Path
import numpy as np

print("=" * 80)
print("强制重建嵌入索引")
print("Force Rebuild Embedding Index")
print("=" * 80)

deps_dir = "/home/yc/CpGPT/examples/dependencies"

print(f"\n步骤 1: 删除现有的 .mmap 文件（强制重建）")

species = "homo_sapiens"
dna_llm = "nucleotide-transformer-v2-500m-multi-species"
dna_context_len = 2001

# 删除符号链接目录下的 .mmap 文件
mmap_file = Path(deps_dir) / "dna_embeddings" / species / dna_llm / f"{dna_context_len}bp_dna_embeddings.mmap"
if mmap_file.exists():
    print(f"  删除: {mmap_file}")
    mmap_file.unlink()
    print(f"  ✓ 已删除")
else:
    print(f"  - 文件不存在: {mmap_file}")

# 删除源目录下的 .mmap 文件
mmap_file_source = Path(deps_dir) / "human" / "dna_embeddings" / species / dna_llm / f"{dna_context_len}bp_dna_embeddings.mmap"
if mmap_file_source.exists():
    print(f"  删除: {mmap_file_source}")
    mmap_file_source.unlink()
    print(f"  ✓ 已删除")
else:
    print(f"  - 文件不存在: {mmap_file_source}")

# 删除所有 .tmp 文件
print(f"\n步骤 2: 删除所有 .tmp 文件")
for tmp_pattern in [
    Path(deps_dir) / "dna_embeddings" / species / dna_llm / "*.tmp",
    Path(deps_dir) / "human" / "dna_embeddings" / species / dna_llm / "*.tmp",
]:
    parent_dir = tmp_pattern.parent
    if parent_dir.exists():
        for tmp_file in parent_dir.glob("*.tmp"):
            print(f"  删除: {tmp_file}")
            tmp_file.unlink()

print(f"  ✓ 完成")

print(f"\n步骤 3: 删除 ensembl_metadata.db（强制重建）")
metadata_db = Path(deps_dir) / "ensembl_metadata.db"
if metadata_db.exists():
    print(f"  删除: {metadata_db}")
    metadata_db.unlink()
    print(f"  ✓ 已删除")

metadata_db_tmp = Path(deps_dir) / "ensembl_metadata.db.tmp"
if metadata_db_tmp.exists():
    print(f"  删除: {metadata_db_tmp}")
    metadata_db_tmp.unlink()

print(f"\n步骤 4: 重新下载依赖（这会重建所有索引）")
print(f"  这可能需要几分钟...")

try:
    from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
    
    inferencer = CpGPTInferencer(dependencies_dir=deps_dir)
    print(f"  ✓ 初始化 CpGPTInferencer")
    
    print(f"\n  开始下载...")
    inferencer.download_dependencies(species='human', overwrite=True)
    print(f"  ✓ 下载完成")
    
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n步骤 5: 验证索引")

try:
    from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
    
    embedder = DNALLMEmbedder(dependencies_dir=deps_dir)
    print(f"  ✓ 初始化 DNALLMEmbedder")
    
    # 检查索引
    if species in embedder.ensembl_metadata_dict:
        if dna_llm in embedder.ensembl_metadata_dict[species]:
            if dna_context_len in embedder.ensembl_metadata_dict[species][dna_llm]:
                index_size = len(embedder.ensembl_metadata_dict[species][dna_llm][dna_context_len])
                print(f"  ✓ 索引存在")
                print(f"    包含 {index_size} 个基因组位置")
            else:
                print(f"  ✗ {dna_context_len} 索引不存在")
                print(f"  当前键: {list(embedder.ensembl_metadata_dict[species][dna_llm].keys())}")
                exit(1)
        else:
            print(f"  ✗ {dna_llm} 不存在")
            exit(1)
    else:
        print(f"  ✗ {species} 不存在")
        exit(1)
    
    # 检查 .mmap 文件
    mmap_file = Path(deps_dir) / "dna_embeddings" / species / dna_llm / f"{dna_context_len}bp_dna_embeddings.mmap"
    if mmap_file.exists():
        size_mb = mmap_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ .mmap 文件存在 ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ .mmap 文件不存在: {mmap_file}")
        exit(1)
    
except Exception as e:
    print(f"  ✗ 验证失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ 索引重建完成！")
print("=" * 80)

print("\n现在可以运行预测脚本:")
print("  cd /home/yc/CpGPT")
print("  python examples/935k_simple_prediction.py")
print()

