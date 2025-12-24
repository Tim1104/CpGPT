#!/bin/bash

echo "========================================="
echo "修复 DNA 嵌入目录结构"
echo "========================================="

# 设置目录路径
DEPS_DIR="dependencies"
SOURCE_DIR="$DEPS_DIR/human/dna_embeddings/homo_sapiens"
TARGET_DIR="$DEPS_DIR/dna_embeddings"
LINK_PATH="$TARGET_DIR/homo_sapiens"

echo ""
echo "1. 检查源目录是否存在..."
if [ ! -d "$SOURCE_DIR" ]; then
    echo "✗ 错误: 源目录不存在: $SOURCE_DIR"
    echo ""
    echo "请先运行以下命令下载依赖："
    echo "  python -c \"from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer; inferencer = CpGPTInferencer(dependencies_dir='./dependencies'); inferencer.download_dependencies(species='human', overwrite=False)\""
    exit 1
fi
echo "✓ 源目录存在: $SOURCE_DIR"

echo ""
echo "2. 创建目标目录..."
mkdir -p "$TARGET_DIR"
echo "✓ 目标目录已创建: $TARGET_DIR"

echo ""
echo "3. 检查符号链接是否已存在..."
if [ -L "$LINK_PATH" ]; then
    echo "符号链接已存在，删除旧链接..."
    rm "$LINK_PATH"
elif [ -d "$LINK_PATH" ]; then
    echo "目录已存在（不是符号链接），删除..."
    rm -rf "$LINK_PATH"
fi

echo ""
echo "4. 创建符号链接..."
# 使用相对路径
cd "$TARGET_DIR"
ln -s "../human/dna_embeddings/homo_sapiens" "homo_sapiens"
cd - > /dev/null

if [ -L "$LINK_PATH" ]; then
    echo "✓ 符号链接创建成功"
    echo ""
    echo "链接详情："
    ls -la "$LINK_PATH"
    echo ""
    echo "链接目标："
    readlink "$LINK_PATH"
else
    echo "✗ 符号链接创建失败"
    exit 1
fi

echo ""
echo "5. 验证目录结构..."
if [ -d "$LINK_PATH/nucleotide-transformer-v2-500m-multi-species" ]; then
    echo "✓ 验证成功: nucleotide-transformer-v2-500m-multi-species 目录可访问"
    echo ""
    echo "目录内容："
    ls -lh "$LINK_PATH/nucleotide-transformer-v2-500m-multi-species/"
else
    echo "✗ 验证失败: 无法访问 nucleotide-transformer-v2-500m-multi-species 目录"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ 修复完成！"
echo "========================================="
echo ""
echo "现在可以运行预测脚本："
echo "  python examples/935k_simple_prediction.py"
echo ""

