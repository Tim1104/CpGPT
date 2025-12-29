# DNA 嵌入目录修复说明

## 🐛 发现的问题

根据你的错误日志：

```
FileNotFoundError: Species directory is missing: dependencies/dna_embeddings/homo_sapiens
Dependencies for species 'homo_sapiens' were not downloaded or are incomplete.
```

### 问题原因

**目录结构不匹配**：

1. **S3 下载位置**：
   ```
   dependencies/human/dna_embeddings/homo_sapiens/
   ```

2. **代码期望位置**：
   ```
   dependencies/dna_embeddings/homo_sapiens/
   ```

**为什么会这样？**

- `download_dependencies(species="human")` 从 S3 下载 `dependencies/human/` 目录
- S3 上的结构是：`dependencies/human/dna_embeddings/homo_sapiens/...`
- 但 `DNALLMEmbedder` 期望的是：`dependencies/dna_embeddings/homo_sapiens/...`
- 多了一层 `human/` 目录

## ✅ 已修复

### 修复方案：创建符号链接

在 `examples/935k_simple_prediction.py` 的步骤 2 中添加了自动修复：

```python
# 修复目录结构：创建符号链接
import os
dna_embeddings_dir = Path(DEPENDENCIES_DIR) / "dna_embeddings"
dna_embeddings_dir.mkdir(parents=True, exist_ok=True)

homo_sapiens_link = dna_embeddings_dir / "homo_sapiens"
human_source = Path(DEPENDENCIES_DIR) / "human" / "dna_embeddings" / "homo_sapiens"

if human_source.exists() and not homo_sapiens_link.exists():
    try:
        # 创建符号链接
        homo_sapiens_link.symlink_to(human_source.resolve(), target_is_directory=True)
    except OSError:
        # 如果符号链接失败（如 Windows 无管理员权限），复制文件
        import shutil
        shutil.copytree(human_source, homo_sapiens_link, dirs_exist_ok=True)
```

### 工作原理

1. **首选方案**：创建符号链接
   - 不占用额外空间
   - 速度快
   - Linux/Mac 默认支持

2. **备用方案**：复制文件
   - 如果符号链接失败（Windows 需要管理员权限）
   - 会占用双倍空间
   - 但保证能工作

## 📁 目录结构

### 下载后的实际结构

```
dependencies/
├── human/
│   └── dna_embeddings/
│       └── homo_sapiens/
│           ├── nucleotide-transformer-v2-500m-multi-species/
│           │   └── 512bp_dna_embeddings.mmap
│           └── ...
└── model/
    └── ...
```

### 代码期望的结构

```
dependencies/
├── dna_embeddings/
│   └── homo_sapiens/  ← 需要这个
│       ├── nucleotide-transformer-v2-500m-multi-species/
│       │   └── 512bp_dna_embeddings.mmap
│       └── ...
└── model/
    └── ...
```

### 修复后的结构

```
dependencies/
├── human/
│   └── dna_embeddings/
│       └── homo_sapiens/  ← 实际文件
│           └── nucleotide-transformer-v2-500m-multi-species/
├── dna_embeddings/
│   └── homo_sapiens/  ← 符号链接到 ../human/dna_embeddings/homo_sapiens/
│       └── nucleotide-transformer-v2-500m-multi-species/
└── model/
    └── ...
```

## 🚀 现在可以使用了

### 重新运行预测脚本

```bash
python examples/935k_simple_prediction.py
```

**期望输出**：

```
================================================================================
935k/EPICv2 甲基化数据预测
================================================================================

[1/6] 初始化组件...

[2/6] 检查并下载依赖和模型...
  - 下载 DNA 嵌入依赖...
  - 创建符号链接...  ← 新增
  - 下载 4 个模型...
    • age_cot
    • cancer
    • clock_proxies
    • proteins

[3/6] 准备数据...
  - 检测到 2 个样本

[4/6] 数据预处理...

[5/6] 运行预测...
  [1/4] 年龄预测...
  ✓ 完成

[6/6] 保存结果...
✓ 预测完成！
```

## 🔍 手动修复（如果需要）

如果自动修复失败，可以手动创建符号链接：

### Linux/Mac

```bash
cd dependencies
mkdir -p dna_embeddings
cd dna_embeddings
ln -s ../human/dna_embeddings/homo_sapiens homo_sapiens
```

### Windows（需要管理员权限）

```cmd
cd dependencies
mkdir dna_embeddings
cd dna_embeddings
mklink /D homo_sapiens ..\human\dna_embeddings\homo_sapiens
```

### Windows（无管理员权限）

```bash
# 复制文件（会占用双倍空间）
cp -r dependencies/human/dna_embeddings/homo_sapiens dependencies/dna_embeddings/homo_sapiens
```

## 📊 验证修复

### 检查符号链接

```bash
ls -la dependencies/dna_embeddings/
```

应该看到：
```
homo_sapiens -> ../human/dna_embeddings
```

### 检查文件存在

```bash
ls dependencies/dna_embeddings/homo_sapiens/
```

应该看到：
```
nucleotide-transformer-v2-500m-multi-species/
```

### Python 验证

```python
from pathlib import Path

# 检查目录存在
dna_embeddings = Path("dependencies/dna_embeddings/homo_sapiens")
print(f"目录存在: {dna_embeddings.exists()}")

# 检查是否是符号链接
print(f"是符号链接: {dna_embeddings.is_symlink()}")

# 检查实际路径
if dna_embeddings.is_symlink():
    print(f"链接到: {dna_embeddings.resolve()}")
```

## ❓ 常见问题

### Q1: 为什么需要符号链接？

**A**: 因为 S3 的目录结构和代码期望的不一致：
- S3: `dependencies/human/dna_embeddings/homo_sapiens/`
- 代码: `dependencies/dna_embeddings/homo_sapiens/`

### Q2: 符号链接会占用空间吗？

**A**: 不会！符号链接只是一个指针，不占用实际空间。

### Q3: Windows 上符号链接失败怎么办？

**A**: 脚本会自动回退到复制文件。虽然占用双倍空间，但能保证工作。

### Q4: 可以直接修改代码吗？

**A**: 可以，但不推荐。因为：
- 需要修改 CpGPT 库的源码
- 升级后会丢失修改
- 符号链接是更优雅的解决方案

### Q5: 为什么不在下载时就修复？

**A**: `download_dependencies()` 是 CpGPT 库的函数，我们不能修改它。只能在下载后修复目录结构。

## 🎯 总结

### 问题

- ✅ S3 下载到 `dependencies/human/dna_embeddings/homo_sapiens/`
- ❌ 代码期望 `dependencies/dna_embeddings/homo_sapiens/`

### 解决方案

- ✅ 自动创建符号链接
- ✅ 如果失败，自动复制文件
- ✅ 对用户透明，无需手动操作

### 现在的工作流程

```bash
# 1. 转换数据
python examples/convert_935k_format.py "./data/Sample251212.csv"

# 2. 运行预测（会自动修复目录结构）
python examples/935k_simple_prediction.py
```

**应该可以正常工作了！** 🚀

---

**如果还有问题，请提供新的错误日志。**

