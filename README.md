# PN-Search

正负反馈驱动的复杂学术论文搜索系统。Reasoner 通过多样化查询提高召回，双边界 Listwise Reranker 将候选论文划分为 `SELECT`、`BORDERLINE`、`REJECT`，并将正选概念和负选模式共同反馈到下一轮查询演化。

## 已实现功能

- 复杂查询解析：纳入、偏好、排除和元数据条件；
- Semantic Scholar 与 OpenAlex 学术搜索 API；
- 多 API 合并、DOI/标题去重、硬条件过滤和粗排；
- 高召回迭代 Search Reasoner；
- 基于标题和摘要的 Listwise 三分类 Reranker；
- 正选扩展、负选纠偏、边界论文复核反馈；
- API 预算、重复率、拒绝率和边际收益停止策略；
- OpenAI-compatible Qwen/vLLM 模型接口；
- 无模型服务时可运行的启发式后端；
- CLI、FastAPI、训练数据构建、LoRA SFT 和评测脚本。

## 本地安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

启发式模式无需模型权重：

```bash
pnsearch search "寻找使用LLM Agent进行学术论文检索并支持查询演化的工作" \
  --mode heuristic \
  --output outputs/example.json
```

真实 Qwen/vLLM 模式：

```bash
cp .env.example .env
export PNSEARCH_MODE=llm
export PNSEARCH_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PNSEARCH_REASONER_MODEL=Qwen/Qwen3-8B
export PNSEARCH_RERANKER_MODEL=Qwen/Qwen3-Reranker-4B

pnsearch search "complex academic query" --mode llm
```

## HTTP API

```bash
pip install -e '.[api]'
uvicorn pnsearch.api:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://127.0.0.1:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"LLM agents for comprehensive academic paper search"}'
```

## 超算数据目录

Git 仓库仅保存代码。数据、模型和训练产物统一存放在：

```text
/file_storage01/home/juanliu/25_ymj/search_data/
├── raw/          # PaSa、AstaBench 原始数据
├── processed/    # 统一格式 train/validation/test
├── candidates/   # API 宽召回候选与教师标注
├── indexes/      # PaSa 离线论文检索索引
├── models/       # 基础模型和 LoRA checkpoint
└── outputs/      # 推理结果、日志和评测报告
```

如果服务器可以直接访问 Hugging Face，可一键准备 PaSa/AstaBench：

```bash
cd /file_storage01/home/juanliu/25_ymj/search
bash scripts/prepare_server_data.sh
```

当前超算登录节点不能直接访问 Hugging Face。本项目提供本地临时中转脚本；数据只在临时目录存在，传输成功或失败后都会自动删除：

```bash
# 公开 LitSearch 查询与标题/摘要语料
DATASETS=litsearch bash scripts/relay_datasets_to_server.sh

# PaSa/AstaBench 获得 Hugging Face 访问权限后再执行
DATASETS=pasa,asta bash scripts/relay_datasets_to_server.sh
```

PaSa 和 AstaBench 的 Hugging Face 仓库当前均为受限访问，需要先在各自数据集页面申请权限。

详细数据设计参见 [DATASETS.md](./DATASETS.md)，整体模型方案参见 [MODEL_DESIGN.md](./MODEL_DESIGN.md)。

## 数据构建流程

```bash
export PYTHONPATH=src
DATA_ROOT=/file_storage01/home/juanliu/25_ymj/search_data

# 1. 下载官方数据
python scripts/download_datasets.py --data-root "$DATA_ROOT"

# 2. 统一 PaSa/AstaBench 查询格式
python scripts/prepare_benchmarks.py \
  --pasa-root "$DATA_ROOT/raw/pasa" \
  --asta "$DATA_ROOT/raw/asta-bench/tasks/paper_finder_bench/validation_*.json" \
  --output "$DATA_ROOT/processed"

# 3. 宽召回候选挖掘
python scripts/mine_candidates.py \
  --queries "$DATA_ROOT/processed/queries_train.jsonl" \
  --output "$DATA_ROOT/candidates/train_raw.jsonl"

# 4. 使用教师 Reranker 标注 SELECT/BORDERLINE/REJECT
python scripts/label_candidates.py \
  --queries "$DATA_ROOT/processed/queries_train.jsonl" \
  --candidates "$DATA_ROOT/candidates/train_raw.jsonl" \
  --output "$DATA_ROOT/candidates/train_labeled.jsonl"

# 5. 生成 query-grouped Listwise 数据
python scripts/build_listwise_dataset.py \
  --queries "$DATA_ROOT/processed/queries_train.jsonl" \
  --candidates "$DATA_ROOT/candidates/train_labeled.jsonl" \
  --output "$DATA_ROOT/processed/reranker_train.jsonl"
```

### PaSa + MiMo 教师数据闭环

MiMo 使用 OpenAI-compatible API。系统 Controller 变量可直接复用；代码也兼容
`LLM_MODEL_URL`、`LLM_API_KEY`、`LLM_MODEL_NAME`。凭据只通过环境变量注入，不提交到 Git：

```bash
export CL_GISM_CONTROLLER_BASE_URL='.../v1'
export CL_GISM_CONTROLLER_API_KEY='...'
export CL_GISM_CONTROLLER_MODEL='mimo-v2.5-pro'
```

构建离线索引与第一批训练数据：

```bash
sbatch scripts/slurm_build_pasa_index.sh

PYTHONPATH=src python3 scripts/prepare_pasa_sft.py \
  --input "$DATA_ROOT/raw/pasa/sft_selector/train.jsonl" \
  --pointwise-output "$DATA_ROOT/processed/pasa/reranker_pointwise_train.jsonl" \
  --listwise-output "$DATA_ROOT/processed/pasa/reranker_listwise_train.jsonl"

sbatch scripts/slurm_mine_pasa_candidates.sh

PYTHONPATH=src python3 scripts/label_candidates.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  --candidates "$DATA_ROOT/candidates/pasa_train_shards/raw_0.jsonl" \
  --candidates "$DATA_ROOT/candidates/pasa_train_shards/raw_1.jsonl" \
  --candidates "$DATA_ROOT/candidates/pasa_train_shards/raw_2.jsonl" \
  --candidates "$DATA_ROOT/candidates/pasa_train_shards/raw_3.jsonl" \
  --candidates "$DATA_ROOT/candidates/pasa_train_shards/raw_4.jsonl" \
  --output "$DATA_ROOT/candidates/pasa_train_mimo.jsonl" \
  --limit 100 --concurrency 2 --resume

PYTHONPATH=src python3 scripts/build_listwise_dataset.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  --candidates "$DATA_ROOT/candidates/pasa_train_mimo.jsonl" \
  --output "$DATA_ROOT/processed/pasa/reranker_mimo_train.jsonl"

PYTHONPATH=src python3 scripts/generate_reasoner_trajectories.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  --index "$DATA_ROOT/indexes/pasa.sqlite" \
  --output "$DATA_ROOT/processed/pasa/reasoner_mimo_train.jsonl" \
  --preferences-output "$DATA_ROOT/processed/pasa/reasoner_mimo_preferences.jsonl" \
  --teacher llm --limit 100 --resume
```

也可以使用可恢复的教师流水线。默认先标注 100 个查询作为成本和格式校验；确认质量后再逐步提高
`LABEL_LIMIT`。设置 `RUN_REASONER=1` 才会额外生成多轮搜索轨迹：

```bash
LABEL_LIMIT=100 RUN_REASONER=0 bash scripts/run_mimo_teacher_pipeline.sh
```

服务器上已配置 Controller 时，推荐提交可追踪日志的 CPU Slurm 作业；作业默认每次断点续跑
100 个新查询，并把本次 API/Token 成本写入 `pasa_train_mimo.summary.json`：

```bash
sbatch --export=ALL,LABEL_LIMIT=100,MIMO_CONCURRENCY=2 scripts/slurm_label_mimo.sh
```

做教师提示词消融时应使用独立输出，避免不同版本标签混在一起：

```bash
sbatch --export=ALL,LABEL_LIMIT=20,MIMO_CONCURRENCY=2,\
MIMO_OUTPUT="$DATA_ROOT/candidates/pasa_train_mimo_v2.jsonl",\
MIMO_LISTWISE_OUTPUT="$DATA_ROOT/processed/pasa/reranker_mimo_v2_train.jsonl" \
scripts/slurm_label_mimo.sh
```

## 模型训练

```bash
pip install -e '.[train]'

python scripts/train_sft.py \
  --task reranker \
  --model Qwen/Qwen3-8B \
  --train "$DATA_ROOT/processed/reranker_train.jsonl" \
  --validation "$DATA_ROOT/processed/reranker_validation.jsonl" \
  --output "$DATA_ROOT/models/pnsearch-reranker-lora"
```

Reasoner 使用相同入口，将 `--task` 改为 `reasoner` 并传入搜索轨迹数据。

超算训练环境和 Slurm 作业：

```bash
# 首次执行会创建环境骨架；不会盲目安装 CUDA/PyTorch
bash scripts/setup_training_env.sh

# GPU 配额恢复并确认计算节点驱动后，明确指定兼容 wheel 源
INSTALL_TORCH=1 TORCH_INDEX_URL='<compatible-pytorch-index>' \
  bash scripts/setup_training_env.sh

# 检查脚本和路径后提交
sbatch scripts/slurm_train_reranker.sh
sbatch scripts/slurm_train_reasoner.sh
```

默认使用服务器已有的 `Qwen3-Coder-30B-A3B-Instruct` 和 8 张 GPU，通过 LoRA + DeepSpeed ZeRO-3 训练。正式实验可将 `MODEL_PATH` 替换为更适合相关性判定的 Qwen Instruct/Reranker 权重。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 评测

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --queries "$DATA_ROOT/processed/queries_test.jsonl" \
  --predictions "$DATA_ROOT/outputs/predictions.jsonl"
```

输出包括 Precision、Recall、F1、Recall@20/50/100 和 nDCG。
