# Ubuntu字体问题快速修复 ⚡

## 🐛 问题
PDF生成时出现大量警告，中文和数字显示为乱码：
```
UserWarning: Glyph 22270 missing from font(s) DejaVu Sans
UserWarning: Glyph 20248 missing from font(s) DejaVu Sans
```

## ✅ 一键修复（推荐）

### 步骤1：运行自动修复脚本
```bash
cd examples
sudo python3 fix_chinese_font_ubuntu.py
```

这个脚本会自动：
- ✅ 安装文泉驿微米黑字体
- ✅ 安装Noto CJK字体
- ✅ 更新系统字体缓存
- ✅ 清除matplotlib缓存

### 步骤2：测试字体
```bash
python3 test_chinese_fonts.py
```

### 步骤3：运行主脚本
```bash
python3 935k_enhanced_prediction.py
```

---

## 🔧 手动修复

如果自动脚本失败，可以手动执行：

```bash
# 1. 安装字体
sudo apt-get update
sudo apt-get install -y fonts-wqy-microhei fonts-noto-cjk

# 2. 更新字体缓存
sudo fc-cache -fv

# 3. 清除matplotlib缓存
rm -rf ~/.cache/matplotlib
rm -rf ~/.matplotlib

# 4. 验证字体
fc-list :lang=zh | grep -E 'WenQuanYi|Noto'

# 5. 测试
python3 test_chinese_fonts.py
```

---

## ✅ 成功标志

运行主脚本时，应该看到：
```
[7/7] 生成PDF报告...
  配置中文字体...
  ✓ 使用中文字体文件: /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
  ✓ 字体名称: WenQuanYi Micro Hei
  生成样本 000536 的PDF报告...
    ✓ PDF报告已生成: results/935k_enhanced_predictions/report_000536.pdf
```

**没有**看到大量的 `UserWarning: Glyph xxx missing` 警告。

---

## 📋 代码已修复

脚本已更新，现在会：

1. **全局字体配置**：只在第一次生成PDF时配置一次字体
2. **优先使用字体文件路径**：直接加载字体文件，而不是依赖字体名称
3. **自动测试字体**：配置后会测试字体是否真的可用
4. **详细的错误提示**：如果字体配置失败，会给出明确的安装建议

---

## 🔍 故障排除

### 问题1：仍然报字体错误
```bash
# 完全清除缓存
sudo rm -rf /root/.cache/matplotlib
rm -rf ~/.cache/matplotlib
rm -rf ~/.matplotlib

# 重新安装字体
sudo apt-get remove fonts-wqy-microhei
sudo apt-get install fonts-wqy-microhei

# 重新运行
python3 935k_enhanced_prediction.py
```

### 问题2：PDF数据不全
检查是否所有预测都已启用：
```python
# 在脚本中确认这些都是True
PREDICT_AGE = True
PREDICT_CANCER = True
PREDICT_CLOCKS = True
PREDICT_PROTEINS = True
CALCULATE_ORGAN_HEALTH = True
```

### 问题3：找不到字体文件
```bash
# 查找字体文件位置
find /usr/share/fonts -name "*wqy*" -o -name "*Noto*CJK*"

# 如果找到了，记下路径，然后修改脚本中的font_paths列表
```

---

## 📚 相关文档

- **test_chinese_fonts.py** - 字体测试脚本
- **fix_chinese_font_ubuntu.py** - 自动修复脚本
- **UBUNTU_FONT_FIX.md** - 详细修复指南

---

**按照以上步骤操作后，PDF应该能正常显示中文了！** ✅

