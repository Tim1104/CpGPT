#!/usr/bin/env python3
"""
重建 DNA 嵌入索引
Rebuild DNA embeddings index
"""

from pathlib import Path
import sys

print("=" * 80)
print("重建 DNA 嵌入索引")
print("Rebuild DNA Embeddings Index")
print("=" * 80)

deps_dir = "/home/yc/CpGPT/examples/dependencies"

print(f"\n步骤 1: 初始化 DNALLMEmbedder")
try:
    from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
    
    embedder = DNALLMEmbedder(dependencies_dir=deps_dir)
    print(f"  ✓ DNALLMEmbedder 初始化成功")
except Exception as e:
    print(f"  ✗ 初始化失败: {e}")
    exit(1)

print(f"\n步骤 2: 检查现有的嵌入文件")
species = "homo_sapiens"
dna_llm = "nucleotide-transformer-v2-500m-multi-species"
dna_context_len = 2001

embeddings_file = (
    Path(deps_dir) / "dna_embeddings" / species / dna_llm / f"{dna_context_len}bp_dna_embeddings.mmap"
)

print(f"  嵌入文件: {embeddings_file}")

if not embeddings_file.exists():
    print(f"  ✗ 嵌入文件不存在")
    print(f"\n  需要先下载嵌入文件:")
    print(f"  python -c \"from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer; ")
    print(f"  inferencer = CpGPTInferencer(dependencies_dir='{deps_dir}'); ")
    print(f"  inferencer.download_dependencies(species='human', overwrite=True)\"")
    exit(1)

size_mb = embeddings_file.stat().st_size / (1024 * 1024)
print(f"  ✓ 嵌入文件存在 ({size_mb:.1f} MB)")

print(f"\n步骤 3: 检查 metadata 中的索引")
if species not in embedder.ensembl_metadata_dict:
    print(f"  ✗ {species} 不在 metadata 中")
    exit(1)

if dna_llm not in embedder.ensembl_metadata_dict[species]:
    print(f"  ✗ {dna_llm} 不在 metadata 中")
    exit(1)

current_index = embedder.ensembl_metadata_dict[species][dna_llm]
print(f"  当前索引: {current_index}")
print(f"  类型: {type(current_index)}")

if isinstance(current_index, dict):
    if dna_context_len in current_index:
        print(f"  ✓ {dna_context_len} 索引已存在")
        print(f"    包含 {len(current_index[dna_context_len])} 个位置")
        
        response = input("\n  索引已存在，是否重建? (y/n): ")
        if response.lower() != 'y':
            print("  取消重建")
            exit(0)
    else:
        print(f"  - {dna_context_len} 索引不存在，需要创建")
else:
    print(f"  ✗ 索引类型错误: {type(current_index)}")
    exit(1)

print(f"\n步骤 4: 重建索引")
print(f"  这可能需要几分钟...")

try:
    # 删除现有的 .tmp 文件
    tmp_file = embeddings_file.with_suffix('.mmap.tmp')
    if tmp_file.exists():
        print(f"  - 删除临时文件: {tmp_file}")
        tmp_file.unlink()
    
    # 重新解析嵌入文件
    # 这会重建索引
    print(f"  - 开始解析嵌入文件...")
    
    # 使用 CpGPTInferencer 重新下载，这会重建索引
    from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
    
    inferencer = CpGPTInferencer(dependencies_dir=deps_dir)
    print(f"  - 重新下载依赖（这会重建索引）...")
    inferencer.download_dependencies(species='human', overwrite=True)
    
    print(f"  ✓ 索引重建完成")
    
except Exception as e:
    print(f"  ✗ 重建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n步骤 5: 验证索引")
# 重新加载 embedder
embedder = DNALLMEmbedder(dependencies_dir=deps_dir)

if dna_context_len in embedder.ensembl_metadata_dict[species][dna_llm]:
    index_size = len(embedder.ensembl_metadata_dict[species][dna_llm][dna_context_len])
    print(f"  ✓ 索引验证成功")
    print(f"    包含 {index_size} 个基因组位置")
else:
    print(f"  ✗ 索引验证失败")
    exit(1)

print("\n" + "=" * 80)
print("✓ 索引重建完成！")
print("=" * 80)

print("\n现在可以运行预测脚本:")
print("  cd /home/yc/CpGPT")
print("  python examples/935k_simple_prediction.py")
print()

