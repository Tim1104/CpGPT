# 935k 零样本推理内存优化指南

## 问题诊断

你遇到的错误信息：
```
RuntimeError: MPS backend out of memory (MPS allocated: 19.05 GB, other allocations: 8.02 GB, max allowed: 20.40 GB)
```

**重要说明：**
- ✅ **你的 Mac GPU (MPS) 已经在使用了** - 不是 CPU 问题
- ❌ **问题是内存不足** - 模型太大，超过了 Mac 的统一内存限制（20.4GB）
- ⚠️ CUDA 警告只是警告，不影响实际运行（因为 Mac 不支持 CUDA）

## 解决方案

### 方案1：使用 CPU 推理（推荐，已应用）

**优点：**
- ✅ 稳定可靠，不会内存溢出
- ✅ 可以处理更大的模型和数据
- ✅ 适合一次性推理任务

**缺点：**
- ⏱️ 速度较慢（约为 GPU 的 1/5 到 1/10）

**配置：**
```python
USE_CPU = True  # 在脚本第63行
MAX_INPUT_LENGTH = 15000  # 在脚本第62行（从30000降低）
```

**预计时间：**
- 年龄预测：约 5-15 分钟（取决于样本数）
- 癌症预测：约 5-15 分钟
- 时钟预测：约 5-15 分钟
- 总计：约 15-45 分钟

### 方案2：优化 MPS GPU 使用（高级用户）

**优点：**
- ⚡ 速度快（约为 CPU 的 5-10 倍）

**缺点：**
- ⚠️ 可能仍然内存溢出
- ⚠️ 需要更多调试

**配置：**
```python
USE_CPU = False  # 在脚本第63行
MAX_INPUT_LENGTH = 8000  # 大幅降低（从30000降到8000）
```

**额外优化（在脚本开头添加）：**
```python
import os
# 允许 MPS 使用更多内存（可能导致系统不稳定）
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
```

## 已应用的优化

### 1. 降低 MAX_INPUT_LENGTH
```python
# 修改前
MAX_INPUT_LENGTH = 30000

# 修改后
MAX_INPUT_LENGTH = 15000  # 减少内存使用
```

### 2. 添加 CPU/GPU 选择开关
```python
USE_CPU = True  # 设置为True使用CPU，False使用MPS GPU
```

### 3. 自动设备检测和提示
脚本现在会显示：
- 可用的计算设备（CUDA/MPS/CPU）
- 当前使用的设备
- 内存优化建议

### 4. 智能 Trainer 配置
```python
if USE_CPU:
    trainer = CpGPTTrainer(accelerator="cpu", precision="32")
else:
    trainer = CpGPTTrainer(precision="16-mixed")  # GPU 使用混合精度
```

## 运行建议

### 首次运行（推荐）
1. 使用 CPU 模式（`USE_CPU = True`）
2. 确保脚本能够完整运行
3. 获得基准结果

### 如果需要加速
1. 设置 `USE_CPU = False`
2. 降低 `MAX_INPUT_LENGTH` 到 8000 或更低
3. 监控内存使用情况
4. 如果仍然溢出，返回 CPU 模式

### 监控内存使用

**在运行脚本时，打开另一个终端监控内存：**
```bash
# 每2秒更新一次内存使用情况
while true; do
    echo "=== $(date) ==="
    ps aux | grep python | grep 935k
    sleep 2
done
```

**或使用 Activity Monitor（活动监视器）：**
1. 打开"活动监视器"应用
2. 查看"内存"标签
3. 找到 Python 进程
4. 观察内存使用情况

## 其他优化技巧

### 1. 分批处理样本
如果样本数很多，可以分批处理：
```python
# 将数据分成多个小批次
batch_size = 10  # 每次处理10个样本
for i in range(0, len(df_935k), batch_size):
    batch_df = df_935k.iloc[i:i+batch_size]
    # 处理这个批次...
```

### 2. 使用更小的模型
```python
# 使用 small 模型而不是 age_cot/cancer
models_to_download = ["small"]  # 而不是 ["age_cot", "cancer", "clock_proxies"]
```

### 3. 减少特征数量
```python
# 只使用最重要的特征（需要领域知识）
top_features = vocab_age["input"][:10000]  # 只使用前10000个特征
df_filtered = df_935k[available_features[:10000]]
```

## 常见问题

### Q1: 为什么 CPU 模式这么慢？
A: 深度学习模型在 CPU 上运行确实较慢，但对于一次性推理任务是可以接受的。如果需要频繁推理，建议使用配备更大内存的 GPU 服务器。

### Q2: 可以使用云 GPU 吗？
A: 可以！推荐使用：
- Google Colab（免费 GPU）
- AWS SageMaker
- Azure ML
- 阿里云 PAI

### Q3: 降低 MAX_INPUT_LENGTH 会影响准确性吗？
A: 可能会，但影响通常不大。935k 数据的特征数量可能不需要 30000 的长度。建议：
- 先用较小的值（如 15000）测试
- 比较结果质量
- 根据需要调整

### Q4: 如何知道我的数据需要多大的 MAX_INPUT_LENGTH？
A: 检查过滤后的特征数量：
```python
print(f"可用特征数: {len(available_features)}")
# MAX_INPUT_LENGTH 应该 >= 可用特征数
```

## 性能对比

| 配置 | 速度 | 内存使用 | 稳定性 | 推荐场景 |
|------|------|----------|--------|----------|
| CPU + 15000 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 一次性推理，稳定优先 |
| MPS + 15000 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 中等数据量 |
| MPS + 8000 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 小数据量，速度优先 |
| MPS + 30000 | ⭐⭐⭐ | ⭐ | ⭐ | ❌ 不推荐（内存溢出） |

## 总结

**当前配置（推荐）：**
- ✅ `USE_CPU = True`
- ✅ `MAX_INPUT_LENGTH = 15000`
- ✅ 稳定可靠，适合首次运行

**如果需要加速：**
- 尝试 `USE_CPU = False` + `MAX_INPUT_LENGTH = 8000`
- 监控内存使用
- 如果溢出，返回 CPU 模式

**长期解决方案：**
- 使用云 GPU（更大内存）
- 优化模型（量化、剪枝）
- 分批处理数据

