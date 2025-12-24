#!/bin/bash

echo "================================================================================"
echo "修复 ensembl_metadata.db 并重新下载依赖"
echo "Fix ensembl_metadata.db and re-download dependencies"
echo "================================================================================"

DEPS_DIR="/home/yc/CpGPT/examples/dependencies"

echo ""
echo "步骤 1: 删除损坏的 ensembl_metadata.db"
if [ -f "$DEPS_DIR/ensembl_metadata.db" ]; then
    echo "  删除: $DEPS_DIR/ensembl_metadata.db"
    rm "$DEPS_DIR/ensembl_metadata.db"
fi

if [ -f "$DEPS_DIR/ensembl_metadata.db.tmp" ]; then
    echo "  删除: $DEPS_DIR/ensembl_metadata.db.tmp"
    rm "$DEPS_DIR/ensembl_metadata.db.tmp"
fi

echo "  ✓ 完成"

echo ""
echo "步骤 2: 删除临时文件"
NT_DIR="$DEPS_DIR/human/dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species"
if [ -d "$NT_DIR" ]; then
    TMP_COUNT=$(find "$NT_DIR" -name "*.tmp" 2>/dev/null | wc -l)
    if [ "$TMP_COUNT" -gt 0 ]; then
        echo "  找到 $TMP_COUNT 个 .tmp 文件，删除中..."
        find "$NT_DIR" -name "*.tmp" -delete
        echo "  ✓ 删除完成"
    else
        echo "  - 没有 .tmp 文件"
    fi
fi

echo ""
echo "步骤 3: 重新下载依赖并重建索引"
echo "  这可能需要几分钟..."
echo ""

python3 -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer

print('  初始化 CpGPTInferencer...')
inferencer = CpGPTInferencer(dependencies_dir='$DEPS_DIR')

print('  开始下载依赖（overwrite=True）...')
inferencer.download_dependencies(species='human', overwrite=True)

print('')
print('  ✓ 下载完成！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ 修复完成！"
    echo "================================================================================"
    echo ""
    echo "现在可以运行预测脚本:"
    echo "  cd /home/yc/CpGPT"
    echo "  python examples/935k_simple_prediction.py"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "✗ 修复失败"
    echo "================================================================================"
    echo ""
    exit 1
fi

