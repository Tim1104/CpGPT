#!/usr/bin/env python3
"""
CpGPT模型下载脚本
下载Web应用所需的所有预训练模型
"""

import sys
import os

print("=" * 80)
print("CpGPT 模型下载脚本")
print("=" * 80)
print()

# 检查依赖
print("📦 检查依赖...")
try:
    import boto3
    print("   ✅ boto3 已安装")
except ImportError:
    print("   ❌ boto3 未安装")
    print()
    print("请先安装boto3:")
    print("   pip3 install boto3")
    sys.exit(1)

try:
    from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
    print("   ✅ CpGPT 已安装")
except ImportError as e:
    print(f"   ❌ CpGPT 导入失败: {e}")
    print()
    print("请确保已安装CpGPT:")
    print("   pip3 install cpgpt")
    sys.exit(1)

print()

# 初始化inferencer
print("🔧 初始化CpGPT Inferencer...")
try:
    inferencer = CpGPTInferencer(dependencies_dir='./dependencies')
    print("   ✅ 初始化成功")
except Exception as e:
    print(f"   ❌ 初始化失败: {e}")
    sys.exit(1)

print()

# 下载依赖文件
print("📥 下载依赖文件...")
try:
    inferencer.download_dependencies()
    print("   ✅ 依赖文件下载完成")
except Exception as e:
    print(f"   ⚠️  依赖文件下载失败: {e}")
    print("   继续下载模型...")

print()

# 下载模型
models = ['age_cot', 'cancer', 'clock_proxies', 'proteins']

print(f"📥 下载 {len(models)} 个预训练模型...")
print()

for i, model_name in enumerate(models, 1):
    print(f"[{i}/{len(models)}] 下载模型: {model_name}")
    try:
        inferencer.download_model(model_name)
        print(f"   ✅ {model_name} 下载完成")
    except Exception as e:
        print(f"   ❌ {model_name} 下载失败: {e}")
        print(f"   错误详情: {type(e).__name__}")
    print()

print("=" * 80)
print("✅ 模型下载完成！")
print("=" * 80)
print()
print("下载的模型:")
for model in models:
    model_path = f"./dependencies/model/{model}"
    if os.path.exists(model_path):
        print(f"   ✅ {model}")
    else:
        print(f"   ❌ {model} (未找到)")

print()
print("现在可以启动Web服务器:")
print("   bash webapp/start_server.sh")
print()
print("或直接启动:")
print("   cd webapp")
print("   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000")
print()

