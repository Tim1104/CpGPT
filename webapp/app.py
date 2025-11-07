"""
CpGPT Web Application - FastAPI Backend
提供935k甲基化数据分析的Web服务
"""

import asyncio
import json
import logging
import shutil
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from lightning import seed_everything
from pydantic import BaseModel

from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

# GPU工具模块
try:
    from webapp.gpu_utils import initialize_device, get_current_device, get_optimal_precision
except ModuleNotFoundError:
    # 当从webapp目录运行时，使用相对导入
    from gpu_utils import initialize_device, get_current_device, get_optimal_precision

# ============================================================================
# 日志配置
# ============================================================================

# 创建日志目录（在导入后定义WEBAPP_DIR之前需要临时处理）
# 这部分会在后面重新定义，这里先用相对路径
_temp_log_dir = Path(__file__).parent / "logs"
_temp_log_dir.mkdir(parents=True, exist_ok=True)
LOG_DIR = _temp_log_dir

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"cpgpt_web_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cpgpt_web")

# ============================================================================
# 配置
# ============================================================================

# 获取当前文件所在目录（webapp目录）
WEBAPP_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = WEBAPP_DIR.parent

DEPENDENCIES_DIR = str(PROJECT_ROOT / "dependencies")
UPLOAD_DIR = str(WEBAPP_DIR / "uploads")
RESULTS_DIR = str(WEBAPP_DIR / "results")
STATIC_DIR = str(WEBAPP_DIR / "static")
RANDOM_SEED = 42
MAX_INPUT_LENGTH = 30000

# 创建必要的目录
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(STATIC_DIR).mkdir(parents=True, exist_ok=True)

# 任务状态存储
tasks: Dict[str, Dict] = {}

# GPU设备信息（在启动时初始化）
DEVICE_INFO = None

# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="CpGPT 935k Methylation Analysis",
    description="Web interface for 935k methylation data analysis using CpGPT",
    version="2.0.0",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# 全局异常处理
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "服务器内部错误",
            "detail": str(exc) if app.debug else "请查看服务器日志获取详细信息",
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
        }
    )


# ============================================================================
# 启动和关闭事件
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    global DEVICE_INFO

    logger.info("=" * 80)
    logger.info("CpGPT Web Application Starting...")
    logger.info(f"Dependencies directory: {DEPENDENCIES_DIR}")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info("=" * 80)

    # 初始化GPU设备
    DEVICE_INFO = initialize_device()

    logger.info("=" * 80)
    logger.info("Application Ready!")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("CpGPT Web Application Shutting Down...")
    # 清理临时文件等


# ============================================================================
# 数据模型
# ============================================================================


class TaskStatus(BaseModel):
    """任务状态模型"""

    task_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    report_url: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# 核心分析函数
# ============================================================================


async def analyze_935k_data(task_id: str, file_path: str):
    """
    异步分析935k甲基化数据

    Args:
        task_id: 任务ID
        file_path: 上传的文件路径
    """
    logger.info(f"[Task {task_id}] Starting analysis for file: {file_path}")

    try:
        # 更新任务状态
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 5
        tasks[task_id]["message"] = "初始化环境..."
        logger.info(f"[Task {task_id}] Initializing environment...")

        # 获取设备信息
        device_info = get_current_device()
        logger.info(f"[Task {task_id}] Using device: {device_info['device_type'].upper()}")
        logger.info(f"[Task {task_id}] Device name: {device_info['device_name']}")
        logger.info(f"[Task {task_id}] Precision: {device_info['precision']}")

        # 设置随机种子
        seed_everything(RANDOM_SEED, workers=True)

        # 设置设备特定的优化（已在startup时设置，这里跳过）

        # 初始化inferencer
        tasks[task_id]["progress"] = 10
        tasks[task_id]["message"] = "初始化CpGPT推理器..."
        logger.info(f"[Task {task_id}] Initializing CpGPT inferencer...")
        inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR, data_dir=UPLOAD_DIR)

        # 检查并转换文件格式
        tasks[task_id]["progress"] = 15
        tasks[task_id]["message"] = "检查数据格式..."
        logger.info(f"[Task {task_id}] Checking data format...")

        if file_path.endswith(".csv"):
            logger.info(f"[Task {task_id}] Converting CSV to Arrow format...")
            df_csv = pd.read_csv(file_path)
            logger.info(f"[Task {task_id}] Loaded CSV with shape: {df_csv.shape}")
            arrow_path = file_path.replace(".csv", ".arrow")
            df_csv.to_feather(arrow_path)
            data_path = arrow_path
            logger.info(f"[Task {task_id}] Converted to Arrow: {arrow_path}")
        else:
            data_path = file_path
            logger.info(f"[Task {task_id}] Using Arrow file directly: {data_path}")

        # 创建任务专用目录
        task_dir = Path(RESULTS_DIR) / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = task_dir / "processed"
        figures_dir = task_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Task {task_id}] Created task directory: {task_dir}")

        # 初始化组件
        tasks[task_id]["progress"] = 20
        tasks[task_id]["message"] = "初始化DNA嵌入器..."
        logger.info(f"[Task {task_id}] Initializing DNA embedder and prober...")
        embedder = DNALLMEmbedder(dependencies_dir=DEPENDENCIES_DIR)
        prober = IlluminaMethylationProber(dependencies_dir=DEPENDENCIES_DIR, embedder=embedder)

        # 处理数据
        tasks[task_id]["progress"] = 25
        tasks[task_id]["message"] = "处理甲基化数据..."
        logger.info(f"[Task {task_id}] Processing methylation data...")
        datasaver = CpGPTDataSaver(
            data_paths=data_path,
            processed_dir=str(processed_dir),
            metadata_cols=None,
        )
        datasaver.process_files(prober=prober, embedder=embedder, check_methylation_pattern=False)
        logger.info(f"[Task {task_id}] Data processing complete")

        # 生成DNA嵌入
        tasks[task_id]["progress"] = 35
        tasks[task_id]["message"] = "生成DNA序列嵌入..."
        all_genomic_locations = datasaver.all_genomic_locations.get("homo_sapiens", set())
        logger.info(f"[Task {task_id}] Generating DNA embeddings for {len(all_genomic_locations)} locations...")
        embedder.parse_dna_embeddings(
            genomic_locations=sorted(all_genomic_locations),
            species="homo_sapiens",
            dna_llm="nucleotide-transformer-v2-500m-multi-species",
            dna_context_len=2001,
            batch_size=8,
            num_workers=4,
        )
        logger.info(f"[Task {task_id}] DNA embeddings generated")

        # 读取数据
        df_935k = pd.read_feather(data_path)
        sample_ids = df_935k.iloc[:, 0] if "GSM_ID" not in df_935k.columns else df_935k["GSM_ID"]
        logger.info(f"[Task {task_id}] Loaded data with {len(sample_ids)} samples")

        # 年龄预测
        tasks[task_id]["progress"] = 40
        tasks[task_id]["message"] = "执行年龄预测..."
        logger.info(f"[Task {task_id}] Starting age prediction...")
        age_results = await predict_age(
            task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober
        )
        logger.info(f"[Task {task_id}] Age prediction complete. Mean age: {age_results['predicted_age'].mean():.1f}")

        # 癌症预测
        tasks[task_id]["progress"] = 50
        tasks[task_id]["message"] = "执行癌症预测..."
        logger.info(f"[Task {task_id}] Starting cancer prediction...")
        cancer_results = await predict_cancer(
            task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober
        )
        cancer_rate = (cancer_results["cancer_prediction"].sum() / len(cancer_results)) * 100
        logger.info(f"[Task {task_id}] Cancer prediction complete. Cancer rate: {cancer_rate:.1f}%")

        # 表观遗传时钟预测
        tasks[task_id]["progress"] = 60
        tasks[task_id]["message"] = "执行表观遗传时钟预测..."
        logger.info(f"[Task {task_id}] Starting epigenetic clocks prediction...")
        clocks_results = await predict_clocks(
            task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober
        )
        logger.info(f"[Task {task_id}] Epigenetic clocks prediction complete")

        # 蛋白质水平预测
        tasks[task_id]["progress"] = 70
        tasks[task_id]["message"] = "执行蛋白质水平预测..."
        logger.info(f"[Task {task_id}] Starting proteins prediction...")
        proteins_results = await predict_proteins(
            task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober
        )
        logger.info(f"[Task {task_id}] Proteins prediction complete")

        # 合并结果
        tasks[task_id]["progress"] = 80
        tasks[task_id]["message"] = "生成分析报告..."
        logger.info(f"[Task {task_id}] Merging results and generating report...")
        combined_results = pd.merge(age_results, cancer_results, on="sample_id")
        combined_results = pd.merge(combined_results, clocks_results, on="sample_id")
        combined_results = pd.merge(combined_results, proteins_results, on="sample_id")

        # 生成可视化和报告
        from webapp.report_generator import generate_html_report, create_visualizations

        logger.info(f"[Task {task_id}] Creating visualizations...")
        create_visualizations(
            combined_results, age_results, cancer_results,
            clocks_results, proteins_results, str(figures_dir)
        )

        logger.info(f"[Task {task_id}] Generating HTML report...")
        report_path = task_dir / "analysis_report.html"
        generate_html_report(combined_results, str(report_path), str(figures_dir))

        # 保存结果CSV
        combined_results.to_csv(task_dir / "combined_predictions.csv", index=False)
        logger.info(f"[Task {task_id}] Saved results to CSV")

        # 完成
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "分析完成！"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        tasks[task_id]["report_url"] = f"/results/{task_id}/analysis_report.html"

        logger.info(f"[Task {task_id}] Analysis completed successfully!")
        logger.info(f"[Task {task_id}] Report URL: {tasks[task_id]['report_url']}")

    except Exception as e:
        error_msg = f"分析失败: {str(e)}"
        logger.error(f"[Task {task_id}] {error_msg}")
        logger.error(f"[Task {task_id}] Traceback:\n{traceback.format_exc()}")

        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["message"] = error_msg
        tasks[task_id]["completed_at"] = datetime.now().isoformat()

        # 不再抛出异常，避免中断异步任务


async def predict_age(task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober):
    """年龄预测"""
    model_name = "age_cot"
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/configs/{model_name}.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/{model_name}.ckpt",
        strict_load=True,
    )

    # 过滤特征
    vocab = json.load(open(f"{DEPENDENCIES_DIR}/model/vocabs/{model_name}.json", "r"))
    available_features = [col for col in df_935k.columns if col in vocab["input"]]
    
    df_filtered = df_935k[available_features]
    filtered_path = f"{processed_dir}_age_filtered.arrow"
    df_filtered.to_feather(filtered_path)

    # 重新处理
    datasaver_age = CpGPTDataSaver(
        data_paths=filtered_path, processed_dir=f"{processed_dir}_age", metadata_cols=None
    )
    datasaver_age.process_files(prober=prober, embedder=embedder)

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=f"{processed_dir}_age",
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )

    # 预测 - 使用动态精度
    device_info = get_current_device()
    precision = device_info["precision"]
    trainer = CpGPTTrainer(precision=precision, enable_progress_bar=False)
    predictions = trainer.predict(
        model=model, datamodule=datamodule, predict_mode="forward", return_keys=["pred_conditions"]
    )

    return pd.DataFrame(
        {"sample_id": sample_ids, "predicted_age": predictions["pred_conditions"].flatten()}
    )


async def predict_cancer(task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober):
    """癌症预测"""
    model_name = "cancer"
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/configs/{model_name}.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/{model_name}.ckpt",
        strict_load=True,
    )

    # 过滤特征
    vocab = json.load(open(f"{DEPENDENCIES_DIR}/model/vocabs/{model_name}.json", "r"))
    available_features = [col for col in df_935k.columns if col in vocab["input"]]
    
    df_filtered = df_935k[available_features]
    filtered_path = f"{processed_dir}_cancer_filtered.arrow"
    df_filtered.to_feather(filtered_path)

    # 重新处理
    datasaver_cancer = CpGPTDataSaver(
        data_paths=filtered_path, processed_dir=f"{processed_dir}_cancer", metadata_cols=None
    )
    datasaver_cancer.process_files(prober=prober, embedder=embedder)

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=f"{processed_dir}_cancer",
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )

    # 预测 - 使用动态精度
    device_info = get_current_device()
    precision = device_info["precision"]
    trainer = CpGPTTrainer(precision=precision, enable_progress_bar=False)
    predictions = trainer.predict(
        model=model, datamodule=datamodule, predict_mode="forward", return_keys=["pred_conditions"]
    )

    cancer_logits = predictions["pred_conditions"].flatten()
    cancer_probabilities = torch.sigmoid(torch.tensor(cancer_logits)).numpy()

    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cancer_logit": cancer_logits,
            "cancer_probability": cancer_probabilities,
            "cancer_prediction": (cancer_probabilities > 0.5).astype(int),
        }
    )


async def predict_clocks(task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober):
    """表观遗传时钟预测 - 5种时钟代理"""
    model_name = "clock_proxies"
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/configs/{model_name}.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/{model_name}.ckpt",
        strict_load=True,
    )

    # 过滤特征
    vocab = json.load(open(f"{DEPENDENCIES_DIR}/model/vocabs/{model_name}.json", "r"))
    available_features = [col for col in df_935k.columns if col in vocab["input"]]

    df_filtered = df_935k[available_features]
    filtered_path = f"{processed_dir}_clocks_filtered.arrow"
    df_filtered.to_feather(filtered_path)

    # 重新处理
    datasaver_clocks = CpGPTDataSaver(
        data_paths=filtered_path, processed_dir=f"{processed_dir}_clocks", metadata_cols=None
    )
    datasaver_clocks.process_files(prober=prober, embedder=embedder)

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=f"{processed_dir}_clocks",
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )

    # 预测 - 使用动态精度
    device_info = get_current_device()
    precision = device_info["precision"]
    trainer = CpGPTTrainer(precision=precision, enable_progress_bar=False)
    predictions = trainer.predict(
        model=model, datamodule=datamodule, predict_mode="forward", return_keys=["pred_conditions"]
    )

    # 5种表观遗传时钟: altumage, dunedinpace, grimage2, hrsinchphenoage, pchorvath2013
    clock_names = ["altumage", "dunedinpace", "grimage2", "hrsinchphenoage", "pchorvath2013"]
    clock_values = predictions["pred_conditions"]

    result_dict = {"sample_id": sample_ids}
    for i, clock_name in enumerate(clock_names):
        result_dict[clock_name] = clock_values[:, i]

    return pd.DataFrame(result_dict)


async def predict_proteins(task_id, inferencer, df_935k, sample_ids, processed_dir, embedder, prober):
    """蛋白质水平预测 - 血浆蛋白"""
    model_name = "proteins"
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/configs/{model_name}.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/{model_name}.ckpt",
        strict_load=True,
    )

    # 过滤特征
    vocab = json.load(open(f"{DEPENDENCIES_DIR}/model/vocabs/{model_name}.json", "r"))
    available_features = [col for col in df_935k.columns if col in vocab["input"]]

    df_filtered = df_935k[available_features]
    filtered_path = f"{processed_dir}_proteins_filtered.arrow"
    df_filtered.to_feather(filtered_path)

    # 重新处理
    datasaver_proteins = CpGPTDataSaver(
        data_paths=filtered_path, processed_dir=f"{processed_dir}_proteins", metadata_cols=None
    )
    datasaver_proteins.process_files(prober=prober, embedder=embedder)

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=f"{processed_dir}_proteins",
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )

    # 预测 - 使用动态精度
    device_info = get_current_device()
    precision = device_info["precision"]
    trainer = CpGPTTrainer(precision=precision, enable_progress_bar=False)
    predictions = trainer.predict(
        model=model, datamodule=datamodule, predict_mode="forward", return_keys=["pred_conditions"]
    )

    # 蛋白质水平（标准化值）
    protein_values = predictions["pred_conditions"]

    # 创建蛋白质结果DataFrame
    result_dict = {"sample_id": sample_ids}

    # 根据实际蛋白质数量创建列
    num_proteins = protein_values.shape[1] if len(protein_values.shape) > 1 else 1
    if num_proteins == 1:
        result_dict["protein_level"] = protein_values.flatten()
    else:
        # 使用通用命名，实际应该有具体的蛋白质名称
        for i in range(min(num_proteins, 20)):  # 限制最多20个蛋白质
            result_dict[f"protein_{i+1}"] = protein_values[:, i]

    return pd.DataFrame(result_dict)


# ============================================================================
# API 端点
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    html_file = Path(STATIC_DIR) / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse(content="<h1>CpGPT 935k Analysis</h1><p>Please create index.html</p>")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传935k甲基化数据文件

    支持的格式: CSV, Arrow/Feather
    """
    logger.info(f"Received file upload request: {file.filename}")

    try:
        # 验证文件格式
        if not (file.filename.endswith(".csv") or file.filename.endswith(".arrow") or file.filename.endswith(".feather")):
            logger.warning(f"Invalid file format: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="不支持的文件格式。请上传CSV或Arrow格式的文件。",
            )

        # 验证文件大小（最大500MB）
        MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
        file.file.seek(0, 2)  # 移动到文件末尾
        file_size = file.file.tell()
        file.file.seek(0)  # 重置到开头

        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} bytes")
            raise HTTPException(
                status_code=400,
                detail=f"文件过大。最大支持500MB，当前文件大小: {file_size / 1024 / 1024:.1f}MB",
            )

        logger.info(f"File size: {file_size / 1024 / 1024:.2f}MB")

        # 生成任务ID
        task_id = str(uuid.uuid4())
        logger.info(f"Generated task ID: {task_id}")

        # 保存上传的文件
        file_extension = Path(file.filename).suffix
        file_path = Path(UPLOAD_DIR) / f"{task_id}{file_extension}"

        logger.info(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved successfully: {file_path}")

        # 创建任务记录
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "任务已创建，等待处理...",
            "created_at": datetime.now().isoformat(),
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
        }

        # 启动异步分析任务
        logger.info(f"Starting analysis task for {task_id}")
        asyncio.create_task(analyze_935k_data(task_id, str(file_path)))

        return JSONResponse(
            content={
                "success": True,
                "task_id": task_id,
                "message": "文件上传成功，分析任务已启动",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in tasks:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    return JSONResponse(content=tasks[task_id])


@app.get("/api/tasks")
async def list_tasks():
    """
    列出所有任务
    """
    return JSONResponse(content={"tasks": list(tasks.values())})


@app.get("/results/{task_id}/analysis_report.html")
async def get_report(task_id: str):
    """
    获取HTML分析报告
    """
    report_path = Path(RESULTS_DIR) / task_id / "analysis_report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")

    return FileResponse(report_path, media_type="text/html")


@app.get("/results/{task_id}/figures/{filename}")
async def get_figure(task_id: str, filename: str):
    """
    获取图表文件
    """
    figure_path = Path(RESULTS_DIR) / task_id / "figures" / filename
    if not figure_path.exists():
        raise HTTPException(status_code=404, detail="图表不存在")

    return FileResponse(figure_path)


@app.get("/api/download/{task_id}/pdf")
async def download_pdf(task_id: str):
    """
    下载PDF格式的报告
    """
    logger.info(f"PDF download requested for task: {task_id}")

    if task_id not in tasks:
        logger.warning(f"Task not found for PDF download: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    if tasks[task_id]["status"] != "completed":
        logger.warning(f"Task not completed for PDF download: {task_id}")
        raise HTTPException(status_code=400, detail="报告尚未生成完成")

    try:
        # 生成PDF（可选功能）
        try:
            try:
                from webapp.pdf_generator import generate_pdf_report
            except ModuleNotFoundError:
                from pdf_generator import generate_pdf_report
            pdf_available = True
        except ImportError as e:
            logger.warning(f"PDF generation not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="PDF导出功能不可用。请安装系统依赖：brew install pango cairo gdk-pixbuf libffi glib"
            )

        pdf_path = Path(RESULTS_DIR) / task_id / "analysis_report.pdf"
        html_path = Path(RESULTS_DIR) / task_id / "analysis_report.html"

        if not pdf_path.exists():
            logger.info(f"Generating PDF for task: {task_id}")
            generate_pdf_report(str(html_path), str(pdf_path))
            logger.info(f"PDF generated successfully: {pdf_path}")
        else:
            logger.info(f"Using existing PDF: {pdf_path}")

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"cpgpt_analysis_{task_id}.pdf",
        )
    except Exception as e:
        logger.error(f"PDF generation failed for task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF生成失败: {str(e)}")


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务及其相关文件
    """
    logger.info(f"Delete request for task: {task_id}")

    if task_id not in tasks:
        logger.warning(f"Task not found for deletion: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    try:
        # 删除文件
        task_dir = Path(RESULTS_DIR) / task_id
        if task_dir.exists():
            logger.info(f"Deleting task directory: {task_dir}")
            shutil.rmtree(task_dir)

        # 删除上传的文件
        if "file_path" in tasks[task_id]:
            file_path = Path(tasks[task_id]["file_path"])
            if file_path.exists():
                logger.info(f"Deleting uploaded file: {file_path}")
                file_path.unlink()

        # 删除任务记录
        del tasks[task_id]
        logger.info(f"Task deleted successfully: {task_id}")

        return JSONResponse(content={"success": True, "message": "任务已删除"})

    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@app.get("/health")
async def health_check():
    """
    健康检查 - 支持CUDA和MPS
    """
    active_tasks = len([t for t in tasks.values() if t["status"] == "processing"])
    device_info = get_current_device()

    # 检查PDF功能是否可用
    pdf_available = False
    try:
        try:
            from webapp.pdf_generator import generate_pdf_report
        except ModuleNotFoundError:
            from pdf_generator import generate_pdf_report
        pdf_available = True
    except ImportError:
        pass

    health_info = {
        "status": "healthy",
        "platform": device_info["platform"],
        "machine": device_info["machine"],
        "device_type": device_info["device_type"],
        "device_name": device_info["device_name"],
        "gpu_available": device_info["gpu_available"],
        "cuda_available": device_info["cuda_available"],
        "mps_available": device_info["mps_available"],
        "precision": device_info["precision"],
        "active_tasks": active_tasks,
        "total_tasks": len(tasks),
        "dependencies_dir_exists": Path(DEPENDENCIES_DIR).exists(),
        "pdf_export_available": pdf_available,
    }

    # CUDA特定信息
    if device_info["cuda_available"]:
        health_info["gpu_count"] = device_info["gpu_count"]
        health_info["gpu_memory"] = device_info["gpu_memory"]
        health_info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"

    # MPS特定信息
    if device_info["mps_available"]:
        health_info["pytorch_version"] = torch.__version__

    logger.debug(f"Health check: {health_info}")

    return JSONResponse(content=health_info)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

