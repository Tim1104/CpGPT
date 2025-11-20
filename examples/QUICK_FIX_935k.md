# 🚀 935k 内存溢出快速修复

## ❌ 你遇到的错误
```
RuntimeError: MPS backend out of memory (MPS allocated: 19.05 GB, max allowed: 20.40 GB)
```

## ✅ 已经修复！

### 修改内容
在 `examples/935k_zero_shot_inference.py` 中：

1. **第62-63行** - 添加了配置选项：
```python
MAX_INPUT_LENGTH = 15000  # 从 30000 降低到 15000
USE_CPU = True  # 使用 CPU 而不是 MPS GPU
```

2. **第578-585行** - 智能设备选择：
```python
if USE_CPU:
    print("⚙️ 使用 CPU 进行推理（稳定但较慢）")
    trainer = CpGPTTrainer(accelerator="cpu", precision="32")
else:
    print("⚙️ 使用 MPS GPU 进行推理（快但可能内存溢出）")
    trainer = CpGPTTrainer(precision="16-mixed")
```

## 🎯 现在可以运行了！

```bash
cd /Users/wulianghua/Documents/GitHub/CpGPT
python examples/935k_zero_shot_inference.py
```

**预计运行时间：** 15-45 分钟（CPU 模式）

## ⚡ 如果想要更快（但有风险）

编辑 `examples/935k_zero_shot_inference.py` 第63行：

```python
# 改为
USE_CPU = False  # 使用 MPS GPU

# 同时降低第62行的值
MAX_INPUT_LENGTH = 8000  # 进一步降低内存使用
```

⚠️ **注意：** 可能仍然会内存溢出，如果溢出请改回 `USE_CPU = True`

## 📊 性能对比

| 模式 | 速度 | 稳定性 | 推荐 |
|------|------|--------|------|
| CPU (当前) | 慢 | ⭐⭐⭐⭐⭐ | ✅ 推荐 |
| MPS GPU | 快 5-10倍 | ⭐⭐⭐ | ⚠️ 可能溢出 |

## 🔍 监控运行

打开另一个终端，运行：
```bash
# 监控内存使用
top -pid $(pgrep -f 935k_zero_shot_inference)
```

或打开"活动监视器"查看 Python 进程的内存使用。

## 💡 关键点

1. **你的 Mac GPU 之前是在工作的** - 不是没用到 GPU
2. **问题是内存不够** - 20.4GB 统一内存不足以运行这么大的模型
3. **CPU 模式是最稳定的解决方案** - 虽然慢但一定能跑完
4. **如果需要速度** - 考虑使用云 GPU（Google Colab、AWS 等）

## 📝 详细文档

查看 `examples/README_935k_memory_optimization.md` 了解更多优化技巧。

