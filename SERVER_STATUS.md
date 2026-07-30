# 超算部署状态

更新时间：2026-07-30

## 路径

```text
代码：/file_storage01/home/juanliu/25_ymj/search
数据：/file_storage01/home/juanliu/25_ymj/search_data
基础模型：/file_storage01/home/juanliu/25_ymj/model
```

Git 分支为 `main`，远端为 `git@github.com:yangCode-res/search.git`。

## 已完成

- 本地和超算代码通过 Git 同步；
- LitSearch 官方 GitHub 代码在 `search_data/raw/litsearch`；
- LitSearch Query/Corpus Parquet 在 `search_data/raw/litsearch-data`；
- 597 条 Query 和 64,183 篇标题/摘要语料已处理；
- train/validation/test query_id 无交集；
- Reranker Listwise 数据和 Reasoner bootstrap 数据已生成；
- 本地和超算均通过 5 个单元测试；
- Slurm 训练脚本使用 `gre` 分区、单节点 8 GPU、LoRA + DeepSpeed ZeRO-3。

## 数据文件

```text
search_data/processed/litsearch/
├── queries_train.jsonl
├── queries_validation.jsonl
├── queries_test.jsonl
├── candidates_train.jsonl
├── candidates_validation.jsonl
├── candidates_test.jsonl
├── reranker_train.jsonl
├── reranker_validation.jsonl
├── reranker_test.jsonl
├── reasoner_train.jsonl
├── reasoner_validation.jsonl
├── reasoner_test.jsonl
└── litsearch_manifest.json
```

## 数据统计

| Split | Queries | Listwise examples | SELECT positions | REJECT positions |
|---|---:|---:|---:|---:|
| Train | 396 | 400 | 425 | 2,775 |
| Validation | 92 | 92 | 94 | 642 |
| Test | 109 | 111 | 120 | 768 |

## 尚需外部授权

以下 Hugging Face 数据仓库返回 HTTP 403，需账号申请访问后提供 Token：

- `CarlanLark/pasa-dataset`
- `allenai/asta-bench`

不能把没有出现在 open-world gold 中的论文直接当负例。PaSa/AstaBench 获得授权后，应使用其已知正例、relevance criteria 和教师模型证据判定构造难负例。

## 训练前检查

服务器当前已有：

```text
/file_storage01/home/juanliu/25_ymj/model/Qwen3-Coder-30B-A3B-Instruct
/file_storage01/home/juanliu/25_ymj/model/Qwen3.6-27B
```

默认训练配置选择前者，因为它是 `Qwen3MoeForCausalLM`，与当前 SFT 入口兼容。`Qwen3.6-27B` 是 `Qwen3_5ForConditionalGeneration`，不能直接交给当前 `AutoModelForCausalLM` 入口。

在提交训练作业前需要执行：

```bash
cd /file_storage01/home/juanliu/25_ymj/search
bash scripts/setup_training_env.sh
```

然后再提交 Reranker 作业。Reasoner 当前数据是确定性 bootstrap 行为，只适合格式和初始 SFT 验证；正式训练前应通过运行系统收集成功的多轮搜索轨迹并替换或增强 bootstrap 数据。
