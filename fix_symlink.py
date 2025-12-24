#!/usr/bin/env python3
"""
修复 DNA 嵌入目录结构
Fix DNA embeddings directory structure
"""

import os
import shutil
from pathlib import Path


def main():
    print("=" * 60)
    print("修复 DNA 嵌入目录结构")
    print("Fix DNA Embeddings Directory Structure")
    print("=" * 60)
    
    # 设置目录路径
    deps_dir = Path("dependencies")
    source_dir = deps_dir / "human" / "dna_embeddings" / "homo_sapiens"
    target_dir = deps_dir / "dna_embeddings"
    link_path = target_dir / "homo_sapiens"
    
    print("\n1. 检查源目录是否存在...")
    print("   Checking if source directory exists...")
    
    if not source_dir.exists():
        print(f"✗ 错误: 源目录不存在: {source_dir}")
        print(f"✗ Error: Source directory does not exist: {source_dir}")
        print("\n请先运行以下命令下载依赖：")
        print("Please download dependencies first:")
        print('  python -c "from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer; '
              'inferencer = CpGPTInferencer(dependencies_dir=\'./dependencies\'); '
              'inferencer.download_dependencies(species=\'human\', overwrite=False)"')
        return 1
    
    print(f"✓ 源目录存在: {source_dir}")
    print(f"✓ Source directory exists: {source_dir}")
    
    # 列出源目录内容
    print("\n   源目录内容 / Source directory contents:")
    for item in source_dir.iterdir():
        print(f"     - {item.name}")
    
    print("\n2. 创建目标目录...")
    print("   Creating target directory...")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 目标目录已创建: {target_dir}")
    print(f"✓ Target directory created: {target_dir}")
    
    print("\n3. 检查符号链接是否已存在...")
    print("   Checking if symbolic link already exists...")
    
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            print("   符号链接已存在，删除旧链接...")
            print("   Symbolic link exists, removing old link...")
            link_path.unlink()
        elif link_path.is_dir():
            print("   目录已存在（不是符号链接），删除...")
            print("   Directory exists (not a symlink), removing...")
            shutil.rmtree(link_path)
    
    print("\n4. 创建符号链接...")
    print("   Creating symbolic link...")
    
    try:
        # 使用相对路径创建符号链接
        relative_source = Path("..") / "human" / "dna_embeddings" / "homo_sapiens"
        link_path.symlink_to(relative_source, target_is_directory=True)
        print("✓ 符号链接创建成功（使用符号链接）")
        print("✓ Symbolic link created successfully")
        method = "symlink"
    except (OSError, NotImplementedError) as e:
        print(f"   符号链接失败: {e}")
        print(f"   Symbolic link failed: {e}")
        print("   尝试复制文件...")
        print("   Trying to copy files instead...")
        
        try:
            shutil.copytree(source_dir, link_path, dirs_exist_ok=True)
            print("✓ 文件复制成功")
            print("✓ Files copied successfully")
            method = "copy"
        except Exception as e:
            print(f"✗ 复制失败: {e}")
            print(f"✗ Copy failed: {e}")
            return 1
    
    print("\n   链接详情 / Link details:")
    if link_path.is_symlink():
        print(f"     类型: 符号链接 / Type: Symbolic link")
        print(f"     目标: {link_path.resolve()} / Target: {link_path.resolve()}")
    else:
        print(f"     类型: 目录（复制） / Type: Directory (copied)")
    
    print("\n5. 验证目录结构...")
    print("   Verifying directory structure...")
    
    nucleotide_dir = link_path / "nucleotide-transformer-v2-500m-multi-species"
    if nucleotide_dir.exists():
        print("✓ 验证成功: nucleotide-transformer-v2-500m-multi-species 目录可访问")
        print("✓ Verification successful: nucleotide-transformer-v2-500m-multi-species directory accessible")
        
        print("\n   目录内容 / Directory contents:")
        for item in nucleotide_dir.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"     - {item.name} ({size_mb:.1f} MB)")
            else:
                print(f"     - {item.name}/ (directory)")
    else:
        print("✗ 验证失败: 无法访问 nucleotide-transformer-v2-500m-multi-species 目录")
        print("✗ Verification failed: Cannot access nucleotide-transformer-v2-500m-multi-species directory")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ 修复完成！")
    print("✓ Fix completed!")
    print("=" * 60)
    
    print("\n现在可以运行预测脚本：")
    print("Now you can run the prediction script:")
    print("  python examples/935k_simple_prediction.py")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

