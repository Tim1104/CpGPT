# CpGPT 模型下载指南

## 问题说明

下载CpGPT模型时遇到错误：
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**原因**: CpGPT的预训练模型存储在AWS S3上，需要AWS凭证才能下载。

---

## 解决方案

### 方案1: 配置AWS凭证（推荐）

CpGPT使用的是公开的S3存储桶（`cpgpt-lucascamillo-public`），但仍需要AWS凭证来访问。

#### 步骤1: 安装AWS CLI

```bash
# macOS
brew install awscli

# 或使用pip
pip3 install awscli
```

#### 步骤2: 配置AWS凭证

```bash
aws configure
```

系统会提示输入：
- **AWS Access Key ID**: 您的AWS访问密钥ID
- **AWS Secret Access Key**: 您的AWS秘密访问密钥
- **Default region name**: 输入 `us-east-1`
- **Default output format**: 输入 `json`

**注意**: 
- 如果您没有AWS账户，需要先在 https://aws.amazon.com 注册
- 免费套餐足够下载模型使用
- 访问密钥可以在AWS控制台的IAM服务中创建

#### 步骤3: 下载模型

配置好凭证后，运行：

```bash
python3 download_models.py
```

---

### 方案2: 使用环境变量

如果您已有AWS凭证，可以通过环境变量设置：

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"

python3 download_models.py
```

---

### 方案3: 手动下载（如果可用）

如果CpGPT提供了其他下载方式（如直接HTTP链接），可以手动下载模型文件。

检查CpGPT官方文档：
- GitHub: https://github.com/lucascamillo/cpgpt
- 文档: 查看是否有替代下载方法

---

### 方案4: 使用已有模型（如果您已下载过）

如果您之前已经下载过模型，可以：

1. **查找现有模型**:
   ```bash
   find ~ -name "age_cot" -type d 2>/dev/null
   find ~ -name "cancer" -type d 2>/dev/null
   ```

2. **复制到项目目录**:
   ```bash
   # 假设找到了模型在 ~/old_cpgpt/dependencies/model/
   mkdir -p ./dependencies/model
   cp -r ~/old_cpgpt/dependencies/model/* ./dependencies/model/
   ```

3. **验证模型**:
   ```bash
   ls -la ./dependencies/model/
   ```

   应该看到：
   ```
   age_cot/
   cancer/
   clock_proxies/
   proteins/
   ```

---

## 验证安装

### 检查模型是否存在

```bash
ls -la ./dependencies/model/
```

应该看到4个模型目录：
- `age_cot/` - 年龄预测模型
- `cancer/` - 癌症预测模型
- `clock_proxies/` - 表观遗传时钟模型
- `proteins/` - 蛋白质预测模型

### 测试模型加载

```python
python3 -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir='./dependencies')
print('✅ 模型加载成功')
"
```

---

## 常见问题

### Q1: 我没有AWS账户怎么办？

**选项A**: 注册AWS免费套餐
- 访问 https://aws.amazon.com
- 注册免费套餐（需要信用卡，但下载模型不会产生费用）
- 创建IAM用户和访问密钥

**选项B**: 联系CpGPT作者
- 在GitHub上提issue询问是否有其他下载方式
- 查看是否有Hugging Face或其他镜像

### Q2: 下载速度很慢怎么办？

```bash
# 使用AWS CLI直接下载（可能更快）
aws s3 sync s3://cpgpt-lucascamillo-public/dependencies/ ./dependencies/ --request-payer requester
aws s3 sync s3://cpgpt-lucascamillo-public/model/age_cot/ ./dependencies/model/age_cot/ --request-payer requester
aws s3 sync s3://cpgpt-lucascamillo-public/model/cancer/ ./dependencies/model/cancer/ --request-payer requester
aws s3 sync s3://cpgpt-lucascamillo-public/model/clock_proxies/ ./dependencies/model/clock_proxies/ --request-payer requester
aws s3 sync s3://cpgpt-lucascamillo-public/model/proteins/ ./dependencies/model/proteins/ --request-payer requester
```

### Q3: 模型文件有多大？

预计总大小：
- 依赖文件: ~100-500 MB
- 每个模型: ~500 MB - 2 GB
- 总计: ~3-5 GB

确保有足够的磁盘空间。

### Q4: 可以只下载部分模型吗？

可以！如果您只需要某些功能，可以只下载对应的模型：

```python
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer

inferencer = CpGPTInferencer(dependencies_dir='./dependencies')

# 只下载年龄预测模型
inferencer.download_model('age_cot')

# 只下载癌症预测模型
inferencer.download_model('cancer')
```

但是，Web应用需要所有4个模型才能完整运行。

---

## 临时解决方案：在没有模型的情况下运行

如果暂时无法下载模型，Web服务器仍然可以启动，但分析功能会失败。

您可以：
1. 先启动服务器测试界面
2. 稍后配置AWS凭证并下载模型
3. 重启服务器即可使用完整功能

```bash
# 启动服务器（即使没有模型）
cd webapp
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 下一步

### 如果成功下载模型

1. 验证模型文件存在
2. 启动Web服务器
3. 上传测试数据进行分析

### 如果无法下载模型

1. 检查CpGPT GitHub仓库的Issues
2. 查看是否有其他用户遇到相同问题
3. 联系项目维护者询问替代下载方式
4. 考虑使用Docker镜像（如果提供）

---

## 相关资源

- **CpGPT GitHub**: https://github.com/lucascamillo/cpgpt
- **AWS免费套餐**: https://aws.amazon.com/free/
- **AWS CLI文档**: https://docs.aws.amazon.com/cli/
- **boto3文档**: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

---

**最后更新**: 2025-11-07  
**状态**: 需要AWS凭证才能下载模型

