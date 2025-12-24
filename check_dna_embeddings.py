#!/usr/bin/env python3
"""
检查 DNA 嵌入文件
Check DNA embedding files
"""

from pathlib import Path
import json

print("=" * 80)
print("检查 DNA 嵌入文件")
print("Check DNA Embedding Files")
print("=" * 80)

# 检查路径
nt_dir = Path("/home/yc/CpGPT/examples/dependencies/human/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species")

print(f"\n1. 检查目录: {nt_dir}")
if not nt_dir.exists():
    print("   ✗ 目录不存在")
    exit(1)

print("   ✓ 目录存在")

print("\n2. 列出所有文件:")
all_files = sorted(nt_dir.iterdir())
for f in all_files:
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name:50s} {size_mb:10.1f} MB")
    else:
        print(f"   - {f.name}/ (目录)")

print("\n3. 检查必需的文件:")

# 检查 metadata.json
metadata_file = nt_dir / "metadata.json"
if metadata_file.exists():
    print(f"   ✓ metadata.json 存在")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"     内容: {json.dumps(metadata, indent=2)}")
    except Exception as e:
        print(f"     ✗ 读取失败: {e}")
else:
    print(f"   ✗ metadata.json 不存在")

# 检查 .mmap 文件
mmap_files = list(nt_dir.glob("*.mmap"))
print(f"\n4. 检查 .mmap 文件:")
print(f"   找到 {len(mmap_files)} 个 .mmap 文件")
for mmap_file in mmap_files:
    size_mb = mmap_file.stat().st_size / (1024 * 1024)
    print(f"   - {mmap_file.name}: {size_mb:.1f} MB")

# 检查 .tmp 文件
tmp_files = list(nt_dir.glob("*.tmp"))
print(f"\n5. 检查 .tmp 文件:")
if tmp_files:
    print(f"   ⚠️  找到 {len(tmp_files)} 个 .tmp 文件（可能是未完成的下载）")
    for tmp_file in tmp_files:
        size_mb = tmp_file.stat().st_size / (1024 * 1024)
        print(f"   - {tmp_file.name}: {size_mb:.1f} MB")
    
    print("\n   建议:")
    print("   1. 删除 .tmp 文件")
    print("   2. 重新下载依赖")
    print("\n   运行以下命令:")
    print("   rm /home/yc/CpGPT/examples/dependencies/human/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/*.tmp")
    print("   python -c \"from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer; inferencer = CpGPTInferencer(dependencies_dir='/home/yc/CpGPT/examples/dependencies'); inferencer.download_dependencies(species='human', overwrite=True)\"")
else:
    print(f"   ✓ 没有 .tmp 文件")

# 检查预期的文件
print(f"\n6. 检查预期的文件:")
expected_files = [
    "2001bp_dna_embeddings.mmap",
    "metadata.json",
]

for expected in expected_files:
    expected_path = nt_dir / expected
    if expected_path.exists():
        size_mb = expected_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {expected:50s} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {expected:50s} (缺失)")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)

