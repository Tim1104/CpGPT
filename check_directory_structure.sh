#!/bin/bash

echo "========================================="
echo "检查目录结构"
echo "========================================="

echo ""
echo "1. 检查 dependencies/human/ 目录："
ls -la dependencies/human/ 2>/dev/null || echo "目录不存在"

echo ""
echo "2. 检查 dependencies/human/dna_embeddings/ 目录："
ls -la dependencies/human/dna_embeddings/ 2>/dev/null || echo "目录不存在"

echo ""
echo "3. 检查 dependencies/human/dna_embeddings/homo_sapiens/ 目录："
ls -la dependencies/human/dna_embeddings/homo_sapiens/ 2>/dev/null || echo "目录不存在"

echo ""
echo "4. 检查 dependencies/dna_embeddings/ 目录："
ls -la dependencies/dna_embeddings/ 2>/dev/null || echo "目录不存在"

echo ""
echo "5. 检查 dependencies/dna_embeddings/homo_sapiens/ 是否存在："
if [ -L "dependencies/dna_embeddings/homo_sapiens" ]; then
    echo "是符号链接"
    ls -la dependencies/dna_embeddings/homo_sapiens
    echo "链接目标："
    readlink dependencies/dna_embeddings/homo_sapiens
elif [ -d "dependencies/dna_embeddings/homo_sapiens" ]; then
    echo "是普通目录"
    ls -la dependencies/dna_embeddings/homo_sapiens/
else
    echo "不存在"
fi

echo ""
echo "6. 查找所有 nucleotide-transformer 目录："
find dependencies/ -type d -name "*nucleotide*" 2>/dev/null

echo ""
echo "7. 检查完整路径是否存在："
if [ -d "dependencies/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species" ]; then
    echo "✓ dependencies/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species 存在"
    ls -la dependencies/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/
else
    echo "✗ dependencies/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species 不存在"
fi

echo ""
echo "========================================="

