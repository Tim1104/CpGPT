# ✅ prefetch_factor 参数修复

**错误**: `ValueError: prefetch_factor option could only be specified in multiprocessing`

**修复时间**: 2025-11-07

---

## 🔍 问题分析

### 错误信息
```python
ValueError: prefetch_factor option could only be specified in multiprocessing.
let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.
```

### 错误位置
```python
File "cpgpt/data/components/dna_llm_embedder.py", line 503
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # = 0
        ...
        prefetch_factor=2,  # ❌ 错误：num_workers=0时不能设置
        persistent_workers=num_workers > 0,
    )
```

### 根本原因

**PyTorch DataLoader规则**:
- `prefetch_factor`: 预取因子，用于多进程数据加载
- **要求**: 只能在 `num_workers > 0` 时使用
- **当 `num_workers=0`**: 必须设置 `prefetch_factor=None`

**为什么会出错**:
1. 我们将 `num_workers` 改为 `0` 以避免序列化问题
2. 但 `prefetch_factor=2` 仍然硬编码在代码中
3. PyTorch检测到冲突并抛出错误

---

## ✅ 解决方案

### 修改代码

**文件**: `cpgpt/data/components/dna_llm_embedder.py`  
**行号**: 510

```python
# 修改前 ❌
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    prefetch_factor=2,  # ❌ 硬编码
    persistent_workers=num_workers > 0,
)

# 修改后 ✅
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    prefetch_factor=2 if num_workers > 0 else None,  # ✅ 条件设置
    persistent_workers=num_workers > 0,
)
```

### 逻辑说明

```python
prefetch_factor=2 if num_workers > 0 else None
```

- **当 `num_workers > 0`**: `prefetch_factor=2` (使用预取)
- **当 `num_workers = 0`**: `prefetch_factor=None` (不使用预取)

---

## 🧪 验证修复

### 运行测试
```bash
cd /Users/wulianghua/Documents/GitHub/CpGPT
python examples/935k_zero_shot_inference.py
```

### 预期输出
```
生成DNA序列嵌入...
总共识别到 XXXX 个基因组位置
Generating embeddings: 0%|          | 0/XXXX
```

**不应该再出现**:
- ❌ `ValueError: prefetch_factor option could only be specified in multiprocessing`
- ❌ `TypeError: cannot pickle 'BufferedReader' instances`

---

## 📝 完整修复清单

### 已修复的两个问题

#### 问题1: 多进程序列化错误
- **文件**: `examples/935k_zero_shot_inference.py`
- **修改**: `num_workers=4` → `num_workers=0`
- **行号**: 464

#### 问题2: prefetch_factor参数错误
- **文件**: `cpgpt/data/components/dna_llm_embedder.py`
- **修改**: `prefetch_factor=2` → `prefetch_factor=2 if num_workers > 0 else None`
- **行号**: 510

---

## 🎯 DataLoader参数最佳实践

### macOS + Python 3.13

```python
# 推荐配置
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=0,                                    # 避免序列化问题
    shuffle=False,
    pin_memory=True,                                  # MPS可以使用
    prefetch_factor=None,                             # num_workers=0时必须为None
    persistent_workers=False,                         # num_workers=0时必须为False
)
```

### Linux + CUDA

```python
# 可以使用多进程
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,                                    # 使用多进程
    shuffle=False,
    pin_memory=True,                                  # CUDA推荐
    prefetch_factor=2,                                # 预取2个batch
    persistent_workers=True,                          # 保持worker进程
)
```

### 通用跨平台代码

```python
import platform

# 根据平台自动配置
is_macos = platform.system() == 'Darwin'
num_workers = 0 if is_macos else 4

dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=2 if num_workers > 0 else None,
    persistent_workers=num_workers > 0,
)
```

---

## 📚 PyTorch DataLoader参数说明

### num_workers
- **类型**: int
- **默认**: 0
- **说明**: 数据加载的子进程数量
- **0**: 在主进程中加载数据
- **>0**: 创建多个worker进程并行加载

### prefetch_factor
- **类型**: int or None
- **默认**: None (当num_workers=0时)
- **默认**: 2 (当num_workers>0时)
- **说明**: 每个worker预取的batch数量
- **要求**: 只能在 `num_workers > 0` 时设置

### persistent_workers
- **类型**: bool
- **默认**: False
- **说明**: 是否在epoch之间保持worker进程
- **要求**: 只能在 `num_workers > 0` 时设置为True

### pin_memory
- **类型**: bool
- **默认**: False
- **说明**: 是否将数据固定在内存中
- **CUDA**: 推荐True（加速CPU→GPU传输）
- **MPS**: 可以使用True
- **CPU**: 设置为False

---

## 🐛 相关错误

### 错误1: prefetch_factor with num_workers=0
```python
ValueError: prefetch_factor option could only be specified in multiprocessing
```
**解决**: `prefetch_factor=2 if num_workers > 0 else None`

### 错误2: persistent_workers with num_workers=0
```python
ValueError: persistent_workers option needs num_workers > 0
```
**解决**: `persistent_workers=num_workers > 0`

### 错误3: pickle错误
```python
TypeError: cannot pickle 'BufferedReader' instances
```
**解决**: `num_workers=0`

---

## ✨ 总结

**问题**: 
1. ❌ 多进程序列化错误
2. ❌ prefetch_factor参数冲突

**修复**:
1. ✅ 设置 `num_workers=0`
2. ✅ 条件设置 `prefetch_factor=2 if num_workers > 0 else None`

**文件**:
1. ✅ `examples/935k_zero_shot_inference.py` (Line 464)
2. ✅ `cpgpt/data/components/dna_llm_embedder.py` (Line 510)

**状态**: ✅ 已修复

---

**最后更新**: 2025-11-07  
**修复状态**: ✅ 完成  
**测试状态**: ⏳ 待验证

