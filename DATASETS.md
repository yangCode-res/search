# PN-Search 数据集设计

## 1. 设计依据

赛题三的自动评测以 F1 为核心，占总指标的 70%；运行效率占 20%，结构化输出占 10%。因此数据设计必须同时训练：

1. Reasoner 的高召回搜索行为；
2. Reranker 的精确纳入和明确排除能力；
3. 排序质量和停止决策；
4. API 调用和 Token 成本意识。

## 2. 参考 benchmark 的关键结论

### 2.1 PaSa

PaSa 使用 Crawler 和 Selector 两个模型。Crawler 负责搜索、阅读和引用扩展，Selector 根据查询以及论文标题和摘要判断相关性。

- AutoScholarQuery：约 35,000 条合成细粒度学术查询，用于训练与测试；
- RealScholarQuery：50 条真实研究者查询，仅用于现实场景测试；
- 官方数据包含 AutoScholarQuery 的 train/dev/test、RealScholarQuery test、Crawler SFT 和 Selector SFT 数据；
- 主要指标包括 Precision、Recall、Recall@20/50/100，并对搜索动作成本进行统计。

### 2.2 AstaBench PaperFindingBench

PaperFindingBench 包含三类查询：

- 48 条 navigational 查询：寻找一篇已知论文；
- 43 条 metadata 查询：按作者、年份、引用等元数据条件寻找完整集合；
- 242 条 semantic 查询：根据内容和细粒度语义条件寻找论文集合。

其设计对 PN-Search 有三个重要启示：

1. navigational 和 metadata 查询有完整 gold 集合，适合计算标准 F1；
2. semantic 查询通常只有不完整正例集合，需要使用细粒度 relevance criteria 和论文证据进行判断；
3. semantic 查询使用 estimated recall@estimated 与 nDCG 的调和平均，防止通过返回大量垃圾论文虚增召回。

AstaBench 最多允许返回 250 篇论文，但排序顺序会影响最终得分；输出要求使用 Semantic Scholar CorpusID，并附带能够证明相关性的原文证据。

## 3. 数据目录约束

本地 Git 仓库只保存代码、配置和文档。所有真实数据均存放在超算：

```text
/file_storage01/home/juanliu/25_ymj/search_data/
```

数据、模型权重、缓存和输出均不得提交 Git。

### 当前已准备数据

LitSearch 已在超算完成处理：

```text
原始查询：597
检索语料：64,183 篇论文
训练集：396 queries / 400 listwise examples
验证集：92 queries / 92 listwise examples
测试集：109 queries / 111 listwise examples
Query split 交集：0
```

原始数据约 1.2GB，处理后数据约 60MB。PaSa 和 AstaBench 官方 Hugging Face 数据仓库需要申请访问权限，当前转换和中转代码已准备好，授权后可直接补充。

## 4. 统一 Query 数据格式

```json
{
  "query_id": "semantic_001",
  "query": "visual question answering papers using EMD as an evaluation metric",
  "source": "asta_paper_finder",
  "split": "validation",
  "query_type": "semantic",
  "positive_papers": [
    {
      "paper_id": "173990882",
      "title": ""
    }
  ],
  "known_bad": [],
  "relevance_criteria": [
    {
      "name": "visual_qa",
      "description": "论文必须研究视觉领域的问答任务",
      "weight": 0.5
    },
    {
      "name": "emd_metric",
      "description": "论文必须实际使用 EMD 作为评估指标",
      "weight": 0.5
    }
  ],
  "metadata_constraints": {}
}
```

## 5. 数据划分

### 5.1 训练集

- AutoScholarQuery 官方 train；
- PaSa Crawler/Selector SFT 数据；
- 从训练 Query 出发通过 Semantic Scholar/OpenAlex 挖掘的候选论文；
- 经过 gold 匹配、人工或教师模型确认的正例、边界例和难负例。

### 5.2 验证集

- AutoScholarQuery 官方 dev；
- AstaBench 官方 validation；
- 不与训练集共享 query_id；
- 用于选择阈值、停止条件、Listwise batch size 和 API 预算。

### 5.3 测试集

- AutoScholarQuery 官方 test；
- RealScholarQuery test；
- AstaBench 官方 test；
- 测试标签不得用于 Prompt、候选标注、难负例挖掘或模型训练。

## 6. 防止数据泄漏

采用 Query 级隔离：一个 query_id 的全部搜索动作、论文候选和正负标签只能存在于同一个 split。

额外进行以下检查：

- 规范化查询文本去重；
- 检查训练和测试之间的完全重复 Query；
- 合成 Query 的原始论文若来自测试 Query 的 gold，不用于构造直接记忆式训练样本；
- 教师标注时不向模型暴露 test gold；
- 所有数据源、版本、下载时间和转换参数写入 manifest。

## 7. Reranker 数据构造

每个样本以 Query 为组，包含一组论文：

```text
20%～30% SELECT
20%～30% BORDERLINE
40%～60% REJECT
```

实际比例在训练时随机变化，避免模型通过列表位置或正例数量猜测标签。

### SELECT

- 在完整 gold 集合中；或
- 教师模型依据 relevance criteria 和摘要证据确认满足所有必要条件。

### BORDERLINE

- 主题明显相关但摘要证据不足；
- 只满足部分必要条件；
- 可能需要全文、引用关系或元数据进一步确认。

### REJECT

- benchmark 明确标注的 known bad；或
- 教师模型发现明确违反纳入/排除条件；或
- 经人工确认的高相似难负例。

不在 gold 集合中的论文不能自动视为负例，因为 semantic benchmark 的 gold 通常不完整。

## 8. 难负例类型

- 关键词相同但研究任务不同；
- 研究任务相同但方法不满足；
- 只在背景或相关工作中提到目标概念；
- 通用 Web Search Agent；
- 固定语料上的论文问答；
- 不含 LLM Agent 的传统推荐系统；
- 引用目标论文但自身问题不同；
- 当前 Reranker 高分误选论文。

## 9. Reasoner 数据构造

Reasoner 的单条训练样本为：

```text
Query + Search History + Positive/Negative/Borderline Feedback + Remaining Budget
    -> Next Search Actions or STOP
```

轨迹应记录：

- 每轮查询与搜索来源；
- 返回论文和去重统计；
- 新增正例数量；
- REJECT 比例；
- 正选扩展词与负选错误模式；
- API 调用、Token、延迟；
- 下一步动作与停止原因。

## 10. 评测协议

对于存在完整 gold 集合的查询，计算：

```text
Precision
Recall
F1
Recall@20
Recall@50
Recall@100
nDCG
```

对于 semantic 查询，需要同时报告：

- 已知正例召回；
- relevance criteria 教师判定的 Precision；
- nDCG；
- 平均 API 调用；
- Token 和端到端延迟；
- 返回集合大小。

最终实验必须同时比较固定 Top-K、Pointwise Reranker、普通 Listwise Reranker、双边界 Reranker，以及是否启用负反馈查询演化。
