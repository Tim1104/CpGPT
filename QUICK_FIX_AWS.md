# 🔧 快速修复：下载CpGPT模型

## 问题
```
ModuleNotFoundError: No module named 'boto3'
或
NoCredentialsError: Unable to locate credentials
```

## 快速解决方案

### 步骤1: 安装AWS CLI和配置凭证

```bash
# 1. 安装AWS CLI
brew install awscli

# 2. 配置AWS凭证（需要AWS账户）
aws configure
```

输入您的AWS凭证：
- AWS Access Key ID: `你的访问密钥`
- AWS Secret Access Key: `你的秘密密钥`
- Default region: `us-east-1`
- Default output format: `json`

### 步骤2: 下载模型

```bash
# 运行下载脚本
python3 download_models.py
```

---

## 如果您没有AWS账户

### 选项1: 注册AWS免费套餐

1. 访问 https://aws.amazon.com
2. 注册免费套餐（需要信用卡验证）
3. 在IAM控制台创建访问密钥
4. 使用上述步骤配置和下载

### 选项2: 使用匿名访问（如果支持）

尝试设置匿名凭证：

```bash
# 创建虚拟凭证文件
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
EOF

cat > ~/.aws/config << EOF
[default]
region = us-east-1
output = json
EOF
```

然后尝试下载：
```bash
python3 download_models.py
```

**注意**: 这可能不起作用，因为S3存储桶可能需要真实凭证。

### 选项3: 联系项目维护者

在CpGPT GitHub仓库提issue：
- https://github.com/lucascamillo/cpgpt/issues
- 询问是否有其他下载方式（如Hugging Face、Google Drive等）

---

## 验证模型是否已下载

```bash
# 检查模型目录
ls -la ./dependencies/model/

# 应该看到：
# age_cot/
# cancer/
# clock_proxies/
# proteins/
```

---

## 临时方案：先运行Web界面

即使没有模型，Web服务器也可以启动：

```bash
cd webapp
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000 查看界面。

稍后下载模型后，重启服务器即可使用完整功能。

---

## 需要帮助？

查看详细指南：
- `MODEL_DOWNLOAD_GUIDE.md` - 完整的模型下载指南
- `FINAL_SETUP_SUMMARY.md` - 安装总结

---

**最后更新**: 2025-11-07

