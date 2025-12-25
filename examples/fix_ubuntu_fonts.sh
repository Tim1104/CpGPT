#!/bin/bash
# Ubuntu中文字体修复脚本

echo "=========================================="
echo "Ubuntu中文字体修复脚本"
echo "=========================================="
echo ""

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  请使用sudo运行此脚本"
    echo "用法: sudo bash fix_ubuntu_fonts.sh"
    exit 1
fi

echo "[1/5] 更新软件包列表..."
apt-get update

echo ""
echo "[2/5] 安装文泉驿微米黑字体..."
apt-get install -y fonts-wqy-microhei

echo ""
echo "[3/5] 安装文泉驿正黑字体..."
apt-get install -y fonts-wqy-zenhei

echo ""
echo "[4/5] 安装Noto CJK字体..."
apt-get install -y fonts-noto-cjk

echo ""
echo "[5/5] 清除matplotlib字体缓存..."
# 清除所有用户的matplotlib缓存
for user_home in /home/*; do
    if [ -d "$user_home/.cache/matplotlib" ]; then
        echo "  清除 $user_home/.cache/matplotlib"
        rm -rf "$user_home/.cache/matplotlib"
    fi
done

# 清除root用户的缓存
if [ -d "/root/.cache/matplotlib" ]; then
    echo "  清除 /root/.cache/matplotlib"
    rm -rf "/root/.cache/matplotlib"
fi

echo ""
echo "=========================================="
echo "✅ 字体安装完成！"
echo "=========================================="
echo ""
echo "已安装的中文字体："
fc-list :lang=zh | grep -E "WenQuanYi|Noto" | head -5
echo ""
echo "请重新运行Python脚本："
echo "  python 935k_enhanced_prediction.py"
echo ""

