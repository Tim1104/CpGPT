# Ubuntu中文字体修复指南

## 🐛 问题描述

在Ubuntu系统上运行脚本时，出现大量警告：
```
UserWarning: Glyph 54 (6) missing from font(s) Droid Sans Fallback.
```

导致PDF中的中文和数字显示为乱码或方框。

---

## ✅ 解决方案

### 方案1：自动修复（推荐）

#### 步骤1：运行字体测试脚本
```bash
cd examples
python3 test_chinese_fonts.py
```

这个脚本会：
- ✅ 检查系统中文字体
- ✅ 清除matplotlib字体缓存
- ✅ 重建字体列表
- ✅ 生成测试图片验证字体

#### 步骤2：如果测试失败，安装中文字体
```bash
sudo bash fix_ubuntu_fonts.sh
```

或手动安装：
```bash
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk
rm -rf ~/.cache/matplotlib
```

#### 步骤3：重新运行测试
```bash
python3 test_chinese_fonts.py
```

#### 步骤4：运行主脚本
```bash
python3 935k_enhanced_prediction.py
```

---

### 方案2：手动修复

#### 1. 检查已安装的中文字体
```bash
fc-list :lang=zh
```

应该看到类似输出：
```
/usr/share/fonts/truetype/wqy/wqy-microhei.ttc: WenQuanYi Micro Hei
/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc: WenQuanYi Zen Hei
```

#### 2. 如果没有中文字体，安装
```bash
# 文泉驿微米黑（推荐）
sudo apt-get install fonts-wqy-microhei

# 文泉驿正黑
sudo apt-get install fonts-wqy-zenhei

# Noto CJK字体
sudo apt-get install fonts-noto-cjk

# 全部安装
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk
```

#### 3. 清除matplotlib字体缓存
```bash
rm -rf ~/.cache/matplotlib
```

#### 4. 验证字体文件存在
```bash
ls -lh /usr/share/fonts/truetype/wqy/
```

应该看到：
```
-rw-r--r-- 1 root root 4.0M wqy-microhei.ttc
-rw-r--r-- 1 root root 8.5M wqy-zenhei.ttc
```

#### 5. 测试matplotlib
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 清除缓存并重建
fm._rebuild()

# 检查可用字体
fonts = [f.name for f in fm.fontManager.ttflist if 'WenQuanYi' in f.name]
print(fonts)
```

---

## 🔧 代码修复说明

脚本已更新，现在会：

### 1. 自动清除字体缓存
```python
cache_dir = Path(fm.get_cachedir())
if cache_dir.exists():
    for cache_file in cache_dir.glob('*.cache'):
        cache_file.unlink()
```

### 2. 重新构建字体列表
```python
fm._rebuild()
```

### 3. 优先使用字体文件路径
```python
font_paths = [
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]

for font_path in font_paths:
    if Path(font_path).exists():
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        break
```

### 4. 回退到字体名称
如果字体文件路径不存在，尝试使用字体名称：
```python
chinese_fonts = [
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Noto Sans CJK SC',
]
```

---

## 📊 推荐字体

| 字体 | 包名 | 文件路径 | 优先级 |
|------|------|---------|--------|
| 文泉驿微米黑 | fonts-wqy-microhei | /usr/share/fonts/truetype/wqy/wqy-microhei.ttc | ⭐⭐⭐ |
| 文泉驿正黑 | fonts-wqy-zenhei | /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc | ⭐⭐ |
| Noto Sans CJK | fonts-noto-cjk | /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc | ⭐⭐⭐ |

**推荐**：文泉驿微米黑（WenQuanYi Micro Hei）
- 文件小（4MB）
- 渲染清晰
- 兼容性好

---

## 🧪 测试步骤

### 1. 运行测试脚本
```bash
python3 test_chinese_fonts.py
```

### 2. 检查输出
```
[1] matplotlib版本: 3.x.x
[2] 字体缓存目录: /home/user/.cache/matplotlib
[3] 清除字体缓存...
    ✓ 缓存清除完成
[4] 重新构建字体列表...
    ✓ 字体列表重建完成
[5] 检查系统中文字体文件...
    ✓ 找到: /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
[6] 检查matplotlib可用的中文字体...
    ✓ 可用: WenQuanYi Micro Hei
[7] 测试中文字体渲染...
    ✓ 测试图片已生成: test_chinese_font.png
```

### 3. 查看测试图片
```bash
xdg-open test_chinese_font.png
```

检查中文是否正常显示。

---

## ⚠️ 常见问题

### 问题1：安装字体后仍然报错
**解决方案**：
```bash
# 清除所有matplotlib缓存
rm -rf ~/.cache/matplotlib
rm -rf ~/.matplotlib

# 重新运行测试
python3 test_chinese_fonts.py
```

### 问题2：字体文件存在但matplotlib找不到
**解决方案**：
```bash
# 更新字体缓存
sudo fc-cache -fv

# 清除matplotlib缓存
rm -rf ~/.cache/matplotlib

# 重建字体列表
python3 -c "import matplotlib.font_manager as fm; fm._rebuild()"
```

### 问题3：数字也显示为方框
**原因**：字体文件损坏或不完整

**解决方案**：
```bash
# 重新安装字体
sudo apt-get remove fonts-wqy-microhei
sudo apt-get install fonts-wqy-microhei

# 清除缓存
rm -rf ~/.cache/matplotlib
```

### 问题4：PDF中文正常但数字乱码
**原因**：字体不支持某些字符

**解决方案**：使用Noto Sans CJK字体
```bash
sudo apt-get install fonts-noto-cjk
rm -rf ~/.cache/matplotlib
```

---

## 📝 验证清单

运行主脚本前，确保：

- [ ] 已安装中文字体（至少一个）
- [ ] 已清除matplotlib缓存
- [ ] 测试脚本运行成功
- [ ] 测试图片中文显示正常
- [ ] 控制台显示"✓ 使用中文字体: xxx"

---

## 🚀 快速修复命令

```bash
# 一键修复（需要sudo权限）
sudo apt-get update && \
sudo apt-get install -y fonts-wqy-microhei fonts-noto-cjk && \
rm -rf ~/.cache/matplotlib && \
python3 test_chinese_fonts.py
```

---

## 📚 相关文档

- **test_chinese_fonts.py** - 字体测试脚本
- **fix_ubuntu_fonts.sh** - 自动修复脚本
- **PDF_GENERATION_FIX.md** - PDF生成问题修复

---

**按照以上步骤操作后，PDF中的中文应该能正常显示了！** ✅

