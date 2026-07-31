# 超算部署状态

更新时间：2026-07-31

## 路径

```text
代码：/file_storage01/home/juanliu/25_ymj/search
数据：/file_storage01/home/juanliu/25_ymj/search_data
基础模型：/file_storage01/home/juanliu/25_ymj/model
Python 环境：/file_storage01/home/juanliu/25_ymj/search_data/envs/pnsearch
```

Git 分支为 `main`，远端为 `git@github.com:yangCode-res/search.git`。Git 仅同步代码；数据、索引、模型和日志只保存在超算。

## 已完成

- PaSa、AstaBench、LitSearch 原始数据已下载到超算；
- PaSa 已规范化为 train 33,551、validation 1,000、test 1,050 条查询；
- AstaBench 已规范化为 validation 66、test 267 条查询；
- PaSa 论文库已构建 SQLite FTS 索引：555,197 篇去重论文，索引约 1.3 GB；
- PaSa Selector 官方 SFT 已转换为 19,826 条 pointwise 样本和 5,530 条 listwise 训练样本；
- 前 5,000 条 PaSa 训练查询已完成宽召回，得到 249,193 个 query-paper 候选，保存为 5 个分片；
- LitSearch 已生成 Reranker Listwise 和 Reasoner bootstrap 数据；
- 训练环境已安装，PyTorch 2.13.0+cu130 可识别 NVIDIA A800-SXM4-80GB；
- 本地单元测试共 14 项通过；
- 4 GPU、Qwen3-Coder-30B-A3B-Instruct、LoRA + DeepSpeed ZeRO-3 冒烟训练正在验证。

## 关键数据文件

```text
search_data/
├── raw/
│   ├── pasa/
│   ├── asta-bench/
│   └── litsearch-data/
├── indexes/
│   └── pasa.sqlite
├── candidates/
│   └── pasa_train_shards/
│       ├── raw_0.jsonl
│       ├── raw_1.jsonl
│       ├── raw_2.jsonl
│       ├── raw_3.jsonl
│       └── raw_4.jsonl
└── processed/
    ├── pasa/
    └── litsearch/
```

候选分片行数分别为 49,868、49,943、49,847、49,799、49,736，共 249,193 行。

## 下一阶段

1. 使用 MiMo 对候选池做 listwise 教师标注，输出 SELECT、BORDERLINE、REJECT 以及证据；
2. 将教师标签与 PaSa gold 合并，生成正式 Reranker listwise 数据；
3. 运行 Reasoner 多轮检索，收集搜索动作、边际召回奖励和偏好对；
4. 先完成小步数冒烟训练，再提交完整 Reranker SFT；
5. 用微调后的 Reranker 重放轨迹，继续训练 Reasoner；
6. 在 PaSa、LitSearch、AstaBench 上评测 F1、Recall@K、API/Token 成本与延迟。

## MiMo 配置边界

代码支持以下环境变量，按顺序读取：

```text
PNSEARCH_LLM_*
CL_GISM_CONTROLLER_*
LLM_MODEL_URL / LLM_API_KEY / LLM_MODEL_NAME
```

凭据不要提交到 Git。若凭据位于其他项目的 `.env`，应由用户明确授权复用，或将相应变量安全地导出到当前作业环境后再运行标注脚本。

## GPU 训练

服务器当前可见 8 张 NVIDIA A800-SXM4-80GB。基础模型为：

```text
/file_storage01/home/juanliu/25_ymj/model/Qwen3-Coder-30B-A3B-Instruct
```

默认脚本申请 8 卡；配额不足时可以覆盖 Slurm 资源并让脚本使用 4 卡：

```bash
sbatch --gres=gpu:4 \
  --cpus-per-task=32 \
  --export=ALL,NUM_GPUS=4,MAX_STEPS=20,OUTPUT_DIR=/file_storage01/home/juanliu/25_ymj/search_data/models/pnsearch-reranker-smoke \
  scripts/slurm_train_reranker.sh
```

正式训练前先确认冒烟作业能够完成模型加载、首个反向传播和 adapter 保存。
