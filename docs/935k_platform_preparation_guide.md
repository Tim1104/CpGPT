# 935k平台CpGPT微调准备指南

## 一、前期准备清单

### 1.1 硬件要求
- **GPU**: 至少8GB显存（小模型）或24GB+显存（大模型）
- **内存**: 建议32GB+
- **存储**: 至少100GB可用空间（用于数据、模型和依赖）

### 1.2 软件环境
```bash
# Python 3.10+
# Poetry (依赖管理)
# AWS CLI (下载预训练模型和依赖)
```

## 二、数据准备步骤

### 步骤1：获取935k平台Manifest文件

**需要的文件格式**：
- 文件名示例: `935k.hg38.manifest.tsv.gz`
- 必需列:
  - `Probe_ID`: 探针ID
  - `CpG_chrm`: 染色体编号（如"1", "2", "X"等）
  - `CpG_beg`: CpG位点的基因组起始位置

**获取途径**：
1. Illumina官方网站
2. GEO数据库的平台信息
3. 芯片制造商提供的注释文件

**文件存放位置**：
```
dependencies/manifests/Anno/935k/935k.hg38.manifest.tsv.gz
```

### 步骤2：修改代码以支持935k平台

需要修改 `cpgpt/data/components/illumina_methylation_prober.py` 文件：

```python
# 在 _collect_files_to_process 方法中添加935k平台
def _collect_files_to_process(
    self,
    human: bool = True,
    mammalian: bool = False,
) -> list[tuple[str, str, bool]]:
    """Collect all manifest files that need to be processed."""
    all_files = []

    # Process Homo sapiens arrays
    if human:
        human_files = [
            "Anno/EPICv2/EPICv2.hg38.manifest.tsv.gz",
            "Anno/EPIC+/EPIC+.hg38.manifest.tsv.gz",
            "Anno/EPIC/EPIC.hg38.manifest.tsv.gz",
            "Anno/HM27/HM27.hg38.manifest.tsv.gz",
            "Anno/HM450/HM450.hg38.manifest.tsv.gz",
            "Anno/MSA/MSA.hg38.manifest.tsv.gz",
            "Anno/935k/935k.hg38.manifest.tsv.gz",  # 添加这一行
        ]
        # ... 其余代码
```

### 步骤3：准备935k甲基化数据

**数据格式要求**：
- **格式**: Arrow或Feather格式
- **结构**: 
  - 行：样本
  - 列：探针ID或基因组位置
  - 值：Beta值（0-1之间）或M值

**必需的元数据列**：
- `species`: 物种名称（如"homo_sapiens"）
- 如果要预测表型，还需要相应的标签列（如"age", "cancer_status"等）

**示例数据结构**：
```python
import pandas as pd
import pyarrow.feather as feather

# 示例数据框
df = pd.DataFrame({
    'sample_id': ['sample1', 'sample2', 'sample3'],
    'species': ['homo_sapiens', 'homo_sapiens', 'homo_sapiens'],
    'age': [45, 52, 38],  # 可选：如果要预测年龄
    'cg00000029': [0.85, 0.82, 0.88],  # 探针ID作为列名
    'cg00000165': [0.12, 0.15, 0.10],
    # ... 更多探针
})

# 保存为Arrow格式
feather.write_feather(df, 'data/935k_samples.arrow')
```

### 步骤4：数据集划分

建议按照以下比例划分数据：
- **训练集**: 70-80%
- **验证集**: 10-15%
- **测试集**: 10-15%

**最小样本量建议**：
- 简单任务（如单一表型预测）: 50-100个样本
- 复杂任务: 500+个样本
- 最佳性能: 1000+个样本

## 三、环境配置步骤

### 3.1 安装CpGPT

```bash
# 克隆仓库
git clone https://github.com/lcamillo/CpGPT.git
cd CpGPT

# 使用Poetry安装依赖
pip install poetry
poetry install
```

### 3.2 配置AWS CLI（下载预训练模型）

```bash
# 安装AWS CLI
# macOS/Linux:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# 配置AWS凭证
aws configure
# 输入Access Key ID和Secret Access Key
# Region: us-east-1
# Output format: json

# 测试连接
aws s3 ls s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw/ --request-payer requester
```

### 3.3 下载依赖文件

```bash
# 创建Python脚本下载依赖
python << EOF
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer

# 初始化inferencer
inferencer = CpGPTInferencer(
    dependencies_dir="./dependencies",
    data_dir="./data"
)

# 下载人类物种的依赖（DNA嵌入等）
inferencer.download_dependencies(species="human", overwrite=False)

# 下载预训练模型（选择small或large）
inferencer.download_model(model_name="small", overwrite=False)
EOF
```

## 四、数据预处理流程

### 4.1 创建数据处理脚本

```python
# preprocess_935k_data.py
from pathlib import Path
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber

# 配置路径
DEPENDENCIES_DIR = "./dependencies"
DATA_DIR = "./data"
RAW_DATA_PATH = "./data/935k_samples.arrow"  # 您的935k数据
PROCESSED_DIR = "./data/935k_processed"

# 初始化组件
embedder = DNALLMEmbedder(dependencies_dir=DEPENDENCIES_DIR)
prober = IlluminaMethylationProber(dependencies_dir=DEPENDENCIES_DIR, embedder=embedder)

# 创建数据保存器
datasaver = CpGPTDataSaver(
    data_paths=RAW_DATA_PATH,
    processed_dir=PROCESSED_DIR,
    metadata_cols=["age"],  # 如果要预测年龄，添加此列
)

# 处理文件
datasaver.process_files(
    prober=prober,
    embedder=embedder,
    check_methylation_pattern=False  # 首次运行设为False
)

# 获取所有基因组位置
all_genomic_locations = datasaver.all_genomic_locations.get("homo_sapiens", set())
print(f"Total genomic locations: {len(all_genomic_locations)}")

# 生成DNA嵌入
embedder.parse_dna_embeddings(
    genomic_locations=sorted(all_genomic_locations),
    species="homo_sapiens",
    dna_llm="nucleotide-transformer-v2-500m-multi-species",
    dna_context_len=2001,
    batch_size=8,  # 根据GPU内存调整
    num_workers=4,
)

# 保存处理后的数据
datasaver.save_processed_data(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
```

### 4.2 运行预处理

```bash
# 激活Poetry环境
poetry shell

# 运行预处理脚本
python preprocess_935k_data.py
```

**预期输出目录结构**：
```
data/935k_processed/
├── train/
│   ├── dataset_0/
│   │   ├── X.mmap          # 甲基化数据
│   │   ├── var.mmap        # 基因组位置
│   │   ├── obsm.mmap       # 元数据（如年龄）
│   │   ├── var_names.npy   # 位置名称
│   │   ├── species.npy     # 物种信息
│   │   └── obsm_names.npy  # 元数据列名
├── val/
│   └── dataset_0/
│       └── ...
└── test/
    └── dataset_0/
        └── ...
```

## 五、配置微调实验

### 5.1 创建实验配置文件

```bash
# 复制模板配置
cp configs/experiment/template.yaml configs/experiment/935k_finetuning.yaml
```

### 5.2 编辑配置文件

```yaml
# configs/experiment/935k_finetuning.yaml
# @package _global_

# ===== Basic Configuration =====
defaults:
  - override /model/net: small          # 或 large
  - override /model/optimizer: adamwschedulefree
  - override /model/scheduler: constant
  - override /logger: csv               # 或 wandb, tensorboard
  - _self_

tags: ["935k_finetuning", "age_prediction", "v1"]

seed: 42

# ===== Model Configuration =====
model:
  training:
    binarize_input: false
    condition_decoder_loss: mae  # mae用于回归，ce用于分类
    generative_splits: 2  # 2为默认，3+会更慢但可能更好
    
    loss_weights:
      condition_loss: 0.1  # 根据验证集调整
      m_mae: 1.0           # 重建损失权重

  optimizer:
    lr: 0.0001  # 可能需要调整

  net:
    use_condition_decoder: true  # 如果预测表型设为true
    condition_size: 1  # 预测变量数量（如年龄=1）

# ===== Training Configuration =====
trainer:
  min_steps: 2000    # 预热步数
  max_steps: 100000  # 根据数据量调整
  precision: "16-mixed"  # 重要：必须使用混合精度

# ===== Data Configuration =====
data:
  _target_: cpgpt.data.cpgpt_datamodule.CpGPTDataModule
  
  batch_size: 16  # 根据GPU内存调整
  
  train_dir: ${paths.data_dir}/935k_processed/train
  val_dir: ${paths.data_dir}/935k_processed/val
  test_dir: ${paths.data_dir}/935k_processed/test
  dependencies_dir: ${paths.dependencies_dir}/human
  
  max_length: 20000  # 935k可能需要更大的值
  sorting_strategy: "random"  # 或 "sorted_chromosome"

# ===== Callback Configuration =====
callbacks:
  model_checkpoint:
    monitor: "val/condition_loss"  # 或 "val/loss"
    filename: "step_{step:06d}"
    mode: "min"

# ===== Checkpoint Configuration =====
strict_load: false  # 加载预训练模型时设为false

# 预训练模型路径
model_ckpt_path: ${paths.dependencies_dir}/model/weights/small.ckpt

# ===== Hydra Configuration =====
hydra:
  run:
    dir: ${paths.log_dir}/experiments/${tags[0]}/${now:%Y-%m-%d_%H-%M-%S}
```

## 六、特征选择（可选但推荐）

对于935k这样的大规模平台，可以先进行特征选择以加快训练：

```python
# feature_selection.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_feather("data/935k_samples.arrow")

# 分离特征和标签
y = df['age'].values  # 或其他目标变量
X = df.drop(['sample_id', 'species', 'age'], axis=1).values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge回归特征选择
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# 选择top N个特征
n_features = 10000  # 根据需要调整
top_indices = np.argsort(np.abs(ridge.coef_))[-n_features:]
selected_probes = df.drop(['sample_id', 'species', 'age'], axis=1).columns[top_indices]

print(f"Selected {len(selected_probes)} features")

# 保存选择的特征
np.save("data/selected_935k_probes.npy", selected_probes)

# 创建过滤后的数据集
df_filtered = df[['sample_id', 'species', 'age'] + list(selected_probes)]
df_filtered.to_feather("data/935k_samples_filtered.arrow")
```

## 七、训练步数估算

根据FAQ建议，理想的训练步数计算：

```
理想步数 = (样本数 × CpG位点数 × 50) / 批次大小

示例：
- 样本数: 100
- CpG位点数: 10,000（特征选择后）
- 批次大小: 10

理想步数 = (100 × 10,000 × 50) / 10 = 5,000,000 步

实际可以从较少步数开始（如100,000步），观察验证集性能
```

## 八、常见问题处理

### 问题1：内存不足
**解决方案**：
- 减小batch_size
- 减小max_length
- 使用特征选择减少CpG位点数
- 使用梯度累积

### 问题2：训练过慢
**解决方案**：
- 使用small模型而非large
- 减少generative_splits（设为1或2）
- 使用特征选择
- 确保使用GPU和混合精度训练

### 问题3：验证损失不下降
**解决方案**：
- 调整学习率（增大或减小）
- 调整loss_weights中的condition_loss权重
- 增加训练步数
- 检查数据质量和标签

### 问题4：DNA嵌入生成失败
**解决方案**：
- 检查基因组位置格式（应为"chr:position"）
- 确保染色体编号正确（1-22, X, Y, MT）
- 减小batch_size
- 检查GPU内存

## 九、验证清单

在开始微调前，确保：

- [ ] 已获取935k平台的manifest文件
- [ ] 已修改代码支持935k平台
- [ ] 已准备好Arrow格式的甲基化数据
- [ ] 数据包含必需的"species"列
- [ ] 已安装所有依赖（poetry install）
- [ ] 已配置AWS CLI并下载预训练模型
- [ ] 已下载DNA嵌入依赖
- [ ] 已运行数据预处理脚本
- [ ] 已创建并配置实验YAML文件
- [ ] GPU可用且有足够显存
- [ ] 已设置precision为"16-mixed"

## 十、下一步

完成所有准备后，即可开始微调：

```bash
# 激活环境
poetry shell

# 开始训练
cpgpt-train experiment=935k_finetuning

# 训练完成后，最佳模型保存在：
# logs/experiments/935k_finetuning/YYYY-MM-DD_HH-MM-SS/checkpoints/
```

