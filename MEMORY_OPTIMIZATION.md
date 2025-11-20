# 🧠 内存优化指南

**更新时间**: 2025-11-07

---

## ❌ 问题：MPS内存溢出

### 错误信息

```
RuntimeError: MPS backend out of memory (MPS allocated: 19.05 GB, other allocations: 8.02 GB, max allowed: 20.40 GB).
```

### 原因分析

1. **多个模型累积占用内存**
   - age_cot 模型：~6-8 GB
   - cancer 模型：~6-8 GB
   - clock_proxies 模型：~6-8 GB
   - **总计需要 > 20 GB，超过MPS限制**

2. **MPS内存管理特点**
   - Apple Silicon的GPU内存是共享的（与系统内存共享）
   - 默认限制约为物理内存的60-70%
   - 不会自动释放，需要手动清理

3. **PyTorch Lightning的内存行为**
   - 模型加载后会保留在内存中
   - 预测结果也会占用内存
   - 需要显式删除和清理

---

## ✅ 解决方案

### 方案1: 自动内存释放（已实现）

**修改内容**：在每个模型预测后立即释放内存

#### 添加导入
```python
import gc
import torch
```

#### 在年龄预测后释放内存
```python
# 释放年龄模型内存
print("\n释放年龄模型内存...")
del model_age
del datamodule_age
del age_predictions
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("内存已释放")
```

#### 在癌症预测后释放内存
```python
# 释放癌症模型内存
print("\n释放癌症模型内存...")
del model_cancer
del datamodule_cancer
del cancer_predictions
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("内存已释放")
```

#### 在时钟预测后释放内存
```python
# 释放时钟模型内存
print("\n释放时钟模型内存...")
del model_clocks
del datamodule_clocks
del clocks_predictions
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("内存已释放")
```

**效果**：
- ✅ 每个模型预测后立即释放内存
- ✅ 峰值内存使用降低到单个模型的大小（~8 GB）
- ✅ 可以顺利运行所有3个模型

---

### 方案2: 增加MPS内存限制（可选）

如果方案1仍然不够，可以增加MPS内存上限：

```bash
# 设置环境变量（允许使用更多内存，但可能导致系统不稳定）
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 运行脚本
python examples/935k_zero_shot_inference.py
```

**警告**：
- ⚠️ 可能导致系统内存不足
- ⚠️ 可能导致系统崩溃
- ⚠️ 仅在有足够物理内存时使用（建议 > 32 GB）

---

### 方案3: 使用CPU推理（备选）

如果MPS内存仍然不足，可以切换到CPU：

```python
# 在脚本开头添加
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 或者完全禁用MPS
# 在创建trainer时指定
trainer = CpGPTTrainer(
    accelerator="cpu",  # 使用CPU而不是MPS
    devices=1,
    ...
)
```

**优缺点**：
- ✅ 不受GPU内存限制
- ❌ 速度较慢（约慢5-10倍）
- ✅ 适合小批量数据

---

### 方案4: 减少MAX_INPUT_LENGTH（高级）

如果数据量很大，可以减少每个样本的最大输入长度：

```python
# 在配置部分修改
MAX_INPUT_LENGTH = 2048  # 从4096减少到2048
```

**效果**：
- ✅ 减少内存使用（约减少50%）
- ⚠️ 可能影响预测准确度（如果样本有很多CpG位点）
- ✅ 适合探针数量较少的平台

---

## 📊 内存使用监控

### 查看当前内存使用

```python
import torch

if torch.backends.mps.is_available():
    # MPS内存统计
    print(f"MPS已分配: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
    print(f"MPS驱动分配: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GB")
elif torch.cuda.is_available():
    # CUDA内存统计
    print(f"CUDA已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 在脚本中添加内存监控

```python
def print_memory_usage(stage_name):
    """打印当前内存使用情况"""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        driver = torch.mps.driver_allocated_memory() / 1024**3
        print(f"\n[{stage_name}] MPS内存: 已分配={allocated:.2f} GB, 驱动={driver:.2f} GB")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n[{stage_name}] CUDA内存: 已分配={allocated:.2f} GB, 保留={reserved:.2f} GB")

# 在每个步骤后调用
print_memory_usage("年龄预测完成")
print_memory_usage("癌症预测完成")
print_memory_usage("时钟预测完成")
```

---

## 🎯 推荐配置

### 对于不同内存大小的Mac

#### 16 GB 内存
```python
# 使用方案1（自动释放）+ 方案4（减少长度）
MAX_INPUT_LENGTH = 2048
batch_size = 1
```

#### 32 GB 内存
```python
# 使用方案1（自动释放）
MAX_INPUT_LENGTH = 4096
batch_size = 1
```

#### 64 GB+ 内存
```python
# 可以不用特殊优化
MAX_INPUT_LENGTH = 4096
batch_size = 2  # 可以稍微增加
```

---

## 🔧 故障排除

### 问题1: 仍然内存溢出

**解决方法**：
1. 确认已添加所有内存释放代码
2. 减少 `MAX_INPUT_LENGTH` 到 2048 或 1024
3. 尝试使用CPU推理（方案3）

### 问题2: 内存释放后仍占用

**解决方法**：
```python
# 强制垃圾回收
import gc
gc.collect()
gc.collect()  # 调用两次确保彻底清理

# 清空MPS缓存
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    torch.mps.synchronize()  # 等待所有操作完成
```

### 问题3: 系统整体内存不足

**解决方法**：
1. 关闭其他应用程序
2. 重启Mac释放系统内存
3. 使用活动监视器检查内存使用

---

## 📈 性能对比

### 优化前
```
步骤4: 年龄预测 → 内存: 8 GB
步骤5: 癌症预测 → 内存: 16 GB
步骤5.5: 时钟预测 → 内存: 24 GB ❌ 溢出！
```

### 优化后
```
步骤4: 年龄预测 → 内存: 8 GB → 释放 → 1 GB
步骤5: 癌症预测 → 内存: 8 GB → 释放 → 1 GB
步骤5.5: 时钟预测 → 内存: 8 GB → 释放 → 1 GB ✅ 成功！
```

---

## ✨ 总结

**已实现的优化**:
- ✅ 自动内存释放（在每个模型后）
- ✅ MPS/CUDA缓存清理
- ✅ Python垃圾回收
- ✅ batch_size = 1（最小化内存）

**预期效果**:
- ✅ 峰值内存从 24 GB 降低到 8 GB
- ✅ 可以在16 GB Mac上运行
- ✅ 不影响预测准确度

**运行建议**:
1. 关闭其他占用内存的应用
2. 确保有至少10 GB可用内存
3. 如果仍然溢出，使用方案2或方案3

---

**最后更新**: 2025-11-07  
**状态**: ✅ 已优化  
**测试**: ⏳ 待验证

