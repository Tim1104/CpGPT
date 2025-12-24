#!/usr/bin/env python3
"""
完整修复脚本 - 重新下载所有依赖
Complete fix script - Re-download all dependencies
"""

import shutil
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("完整修复脚本")
print("Complete Fix Script")
print("=" * 80)

deps_dir = Path("/home/yc/CpGPT/examples/dependencies")

print(f"\n步骤 1: 备份现有依赖")
if deps_dir.exists():
    backup_name = f"dependencies.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir = deps_dir.parent / backup_name
    print(f"  备份到: {backup_dir}")
    shutil.move(str(deps_dir), str(backup_dir))
    print(f"  ✓ 备份完成")
else:
    print(f"  - 依赖目录不存在，跳过备份")

print(f"\n步骤 2: 创建新的依赖目录")
deps_dir.mkdir(parents=True, exist_ok=True)
print(f"  ✓ 创建: {deps_dir}")

print(f"\n步骤 3: 下载 DNA 嵌入依赖")
print(f"  这可能需要几分钟...")

try:
    from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
    
    inferencer = CpGPTInferencer(dependencies_dir=str(deps_dir))
    print(f"  ✓ 初始化 CpGPTInferencer")
    
    print(f"\n  开始下载...")
    inferencer.download_dependencies(species='human', overwrite=True)
    print(f"  ✓ DNA 嵌入依赖下载完成")
    
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    print(f"\n  恢复备份...")
    if backup_dir.exists():
        if deps_dir.exists():
            shutil.rmtree(deps_dir)
        shutil.move(str(backup_dir), str(deps_dir))
        print(f"  ✓ 已恢复备份")
    exit(1)

print(f"\n步骤 4: 创建符号链接")
human_source = deps_dir / "human" / "dna_embeddings" / "homo_sapiens"
dna_embeddings_dir = deps_dir / "dna_embeddings"
homo_sapiens_link = dna_embeddings_dir / "homo_sapiens"

if not human_source.exists():
    print(f"  ✗ 源目录不存在: {human_source}")
    exit(1)

print(f"  源目录: {human_source}")

dna_embeddings_dir.mkdir(parents=True, exist_ok=True)
print(f"  ✓ 创建目标目录: {dna_embeddings_dir}")

if homo_sapiens_link.exists() or homo_sapiens_link.is_symlink():
    if homo_sapiens_link.is_symlink():
        homo_sapiens_link.unlink()
    else:
        shutil.rmtree(homo_sapiens_link)

try:
    homo_sapiens_link.symlink_to(human_source, target_is_directory=True)
    print(f"  ✓ 符号链接创建成功")
except Exception as e:
    print(f"  - 符号链接失败，复制文件: {e}")
    shutil.copytree(human_source, homo_sapiens_link, dirs_exist_ok=True)
    print(f"  ✓ 文件复制成功")

print(f"\n步骤 5: 验证")

# 检查 nucleotide-transformer
nt_dir = homo_sapiens_link / "nucleotide-transformer-v2-500m-multi-species"
if not nt_dir.exists():
    print(f"  ✗ nucleotide-transformer 目录不存在")
    exit(1)

print(f"  ✓ nucleotide-transformer 目录存在")

# 检查 .mmap 文件
mmap_file = nt_dir / "2001bp_dna_embeddings.mmap"
if not mmap_file.exists():
    print(f"  ✗ 2001bp_dna_embeddings.mmap 不存在")
    exit(1)

size_mb = mmap_file.stat().st_size / (1024 * 1024)
print(f"  ✓ 2001bp_dna_embeddings.mmap 存在 ({size_mb:.1f} MB)")

# 检查 ensembl_metadata.db
metadata_db = deps_dir / "ensembl_metadata.db"
if not metadata_db.exists():
    print(f"  ✗ ensembl_metadata.db 不存在")
    exit(1)

print(f"  ✓ ensembl_metadata.db 存在")

# 检查 .tmp 文件
tmp_files = list(nt_dir.glob("*.tmp"))
if tmp_files:
    print(f"  ⚠️  发现 {len(tmp_files)} 个 .tmp 文件，删除中...")
    for tmp_file in tmp_files:
        tmp_file.unlink()
        print(f"    - 删除: {tmp_file.name}")

print("\n" + "=" * 80)
print("✓ 修复完成！")
print("=" * 80)

print("\n现在可以运行预测脚本:")
print("  cd /home/yc/CpGPT")
print("  python examples/935k_simple_prediction.py")
print()

# 删除备份（可选）
if backup_dir.exists():
    response = input("是否删除备份? (y/n): ")
    if response.lower() == 'y':
        shutil.rmtree(backup_dir)
        print(f"✓ 已删除备份: {backup_dir}")
    else:
        print(f"保留备份: {backup_dir}")

