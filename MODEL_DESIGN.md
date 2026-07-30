# PN-Search：正负反馈驱动的复杂学术论文搜索系统

## 1. 项目背景

复杂学术查询通常同时包含研究主题、方法、数据集、时间、领域和论文类型等多维约束。传统关键词检索难以兼顾召回率与精确率：查询过窄会遗漏相关论文，查询过宽则会产生大量语义相似但实际不符合范围的结果。

本方案提出 **PN-Search（Positive-Negative Feedback Search）**：由搜索推理模型 Reasoner 主动生成多样化查询、扩大候选论文召回范围，再由双边界 Listwise Reranker 同时识别应纳入、待复核和应排除的论文。系统进一步利用正选与负选结果共同优化下一轮搜索，实现高召回、低噪声、可迭代且成本可控的论文检索。

核心分工如下：

```text
Reasoner：扩大搜索范围，优先优化 Recall
Reranker：学习纳入和排除边界，优先优化 Precision
联合目标：在有限搜索成本下最大化最终 F1
```

## 2. 核心创新

### 2.1 Recall-first 搜索策略

Reasoner 不在第一轮过早过滤候选论文，而是通过关键词扩展、同义表达、方法名称、数据集名称、引用关系和相似论文等多种路径构造互补搜索动作，以提高相关论文召回率。

### 2.2 双边界 Listwise Reranker

传统 Reranker 通常独立判断一篇论文与查询的相关性：

```text
(Query, Paper) -> Relevance Score
```

本方案让模型在候选集合上下文中同时学习纳入边界和排除边界：

```text
(Query, Inclusion Criteria, Exclusion Criteria, Paper List)
    -> SELECT / BORDERLINE / REJECT
    -> Listwise Ranking
    -> Criteria-level Judgments
```

- `SELECT`：摘要中有明确证据满足必要条件。
- `BORDERLINE`：可能相关，但只满足部分条件或摘要证据不足。
- `REJECT`：明确不满足必要条件，或触发排除条件。

### 2.3 正负反馈驱动的查询演化

正选论文用于发现新的方法名、数据集、作者、benchmark 和引用线索；负选论文则用于识别当前查询的歧义和高频噪声模式。Reasoner 同时利用两类反馈生成下一轮搜索动作：

```text
正反馈 -> 扩展有效概念、引用链和相似论文
负反馈 -> 增加排除约束、修正查询歧义、降低噪声
边界反馈 -> 获取全文、检查引用或进行定向复核
```

负选结果不再只是被丢弃的数据，而是下一轮查询改写的监督信号。

## 3. 系统总体架构

```text
用户复杂学术 Query
          |
          v
1. Query Analyzer
   提取纳入条件、偏好条件、排除条件
          |
          v
2. Search Reasoner
   生成多个高召回搜索动作
          |
          v
3. Academic Search Executor
   调用 Semantic Scholar / OpenAlex / arXiv 等 API
          |
          v
4. Candidate Pool
   合并、去重、摘要补全、低成本粗排
          |
          v
5. Boundary-aware Listwise Reranker
   +-- SELECT：明确相关
   +-- BORDERLINE：证据不足
   +-- REJECT：明确不相关
          |
          v
6. Feedback Analyzer
   +-- 正选概念扩展
   +-- 负选错误模式总结
   +-- 边界论文待验证信息
          |
          v
7. Search Reasoner
   继续搜索 / 引文扩展 / 相似论文扩展 / 停止
          |
          v
8. Final Reranker & Structured Output
```

模型可以先采用两个相互独立的 Qwen 系列模型：

```text
Search Reasoner：生成式推理模型
Listwise Reranker：生成式排序模型或专门的 Reranker 模型
```

初期分别训练两个模型，系统稳定后再使用搜索轨迹进行联合优化。

## 4. Query Analyzer

Query Analyzer 负责把用户查询转换为明确、可验证的检索规则。该模块可以与 Reasoner 共用基础模型。

### 4.1 输入示例

```text
寻找使用 LLM Agent 进行复杂学术论文搜索的方法，要求支持查询分解
或查询演化，关注 2023 年之后的研究，排除只对固定论文集合进行
问答的系统。
```

### 4.2 输出格式

```json
{
  "research_intent": "LLM Agent 学术论文搜索",
  "inclusion_criteria": [
    {
      "id": "I1",
      "criterion": "研究对象是学术论文发现或检索",
      "required": true
    },
    {
      "id": "I2",
      "criterion": "核心方法包含 LLM Agent",
      "required": true
    },
    {
      "id": "I3",
      "criterion": "支持查询分解、改写或迭代搜索",
      "required": false
    }
  ],
  "exclusion_criteria": [
    {
      "id": "E1",
      "criterion": "只在固定论文集合上进行问答"
    },
    {
      "id": "E2",
      "criterion": "只进行通用网页搜索"
    },
    {
      "id": "E3",
      "criterion": "只在背景中提及学术检索"
    }
  ],
  "metadata_constraints": {
    "year_min": 2023,
    "venues": [],
    "domains": []
  }
}
```

其中：

- `required=true` 表示论文必须满足的纳入条件；
- `required=false` 表示用于排序的偏好条件；
- `exclusion_criteria` 表示触发后应排除或降级为边界项的条件；
- 时间、venue、文献类型等可确定条件优先交给规则引擎处理。

## 5. Search Reasoner

### 5.1 模型职责

Reasoner 负责：

1. 解析复杂查询并制定搜索计划；
2. 第一轮生成语义互补的高召回查询；
3. 根据正选论文扩展有效检索概念；
4. 根据负选模式修正歧义和噪声；
5. 对边界论文选择全文复核或引用扩展；
6. 根据边际收益和剩余预算决定是否停止。

### 5.2 动作空间

```text
KEYWORD_SEARCH       关键词搜索
SEMANTIC_SEARCH      语义搜索
QUERY_REWRITE        查询改写
CITATION_FORWARD     搜索引用目标论文的后续工作
CITATION_BACKWARD    搜索目标论文的参考文献
SIMILAR_PAPER        相似论文扩展
AUTHOR_EXPANSION     作者相关工作扩展
READ_FULLTEXT        获取边界论文的全文证据
STOP                 停止搜索
```

### 5.3 首轮输出示例

```json
{
  "analysis": {
    "current_goal": "扩大候选论文召回",
    "remaining_uncertainty": [
      "查询演化相关方法是否覆盖充分"
    ]
  },
  "actions": [
    {
      "type": "KEYWORD_SEARCH",
      "query": "academic paper search LLM agent",
      "source": "semantic_scholar",
      "purpose": "寻找直接研究学术搜索 Agent 的论文",
      "max_results": 30
    },
    {
      "type": "KEYWORD_SEARCH",
      "query": "scientific literature retrieval query decomposition agent",
      "source": "openalex",
      "purpose": "寻找查询分解和演化方法",
      "max_results": 30
    },
    {
      "type": "KEYWORD_SEARCH",
      "query": "paper finding autonomous search language model",
      "source": "semantic_scholar",
      "purpose": "使用不同术语发现潜在论文",
      "max_results": 30
    }
  ],
  "stop": false
}
```

第一轮应强调查询多样性，后续轮次根据反馈逐渐提高精确度。

## 6. Candidate Pool

学术 API 返回结果后先进行低成本处理：

```text
多 API 结果合并
    -> DOI / Paper ID / 标题去重
    -> 摘要与元数据补全
    -> 年份、语言、文献类型等硬约束过滤
    -> BM25 或 Embedding 粗排
    -> 保留 Top 50～100 进入 Reranker
```

论文结构示例：

```json
{
  "paper_id": "p001",
  "title": "...",
  "abstract": "...",
  "year": 2025,
  "venue": "ACL",
  "authors": [],
  "citation_count": 20,
  "retrieved_by": ["action_1", "action_3"]
}
```

## 7. Boundary-aware Listwise Reranker

### 7.1 模型输入

```text
原始 Query
+ 纳入条件
+ 排除条件
+ 多篇候选论文的标题和摘要
```

标题应与摘要共同输入。标题通常包含任务、方法和数据集名称，能帮助模型更准确地区分直接相关与表面相关论文。

### 7.2 模型输出

```json
{
  "results": [
    {
      "paper_id": "p001",
      "label": "SELECT",
      "relevance_score": 0.94,
      "inclusion_judgments": {
        "I1": "satisfied",
        "I2": "satisfied",
        "I3": "satisfied"
      },
      "exclusion_judgments": {
        "E1": "not_violated",
        "E2": "not_violated",
        "E3": "not_violated"
      },
      "evidence_sufficiency": 0.91,
      "reason_codes": ["DIRECT_TASK_MATCH", "METHOD_MATCH"]
    },
    {
      "paper_id": "p002",
      "label": "REJECT",
      "relevance_score": 0.23,
      "inclusion_judgments": {
        "I1": "unknown",
        "I2": "satisfied",
        "I3": "unknown"
      },
      "exclusion_judgments": {
        "E1": "violated",
        "E2": "not_violated",
        "E3": "not_violated"
      },
      "evidence_sufficiency": 0.84,
      "reason_codes": ["FIXED_CORPUS_QA"]
    }
  ],
  "ranking": ["p001", "p005", "p009", "p002"]
}
```

### 7.3 为什么保留 BORDERLINE

只有正负二分类容易过度过滤，导致 Recall 下降。对于摘要信息不足、只满足部分条件或存在歧义的论文，系统应标记为 `BORDERLINE`，随后执行：

- 获取全文或关键段落；
- 检查引用与被引关系；
- 交给更强模型复核；
- 降权保留，而不是立即删除。

### 7.4 分组 Listwise 推理

不建议一次输入全部候选摘要。推荐：

```text
候选 Top 80
    -> 分成 8 组，每组 10 篇
    -> 组内 Listwise 分类与排序
    -> 保留 SELECT 和高分 BORDERLINE
    -> 跨组最终 Listwise 重排
```

为降低位置偏差：

- 训练时随机打乱论文顺序；
- 推理时可对关键批次使用两种排列；
- 对多次结果进行分数平均或投票；
- 保证不同组中正例、边界例和负例比例具有变化。

## 8. Reranker 训练

### 8.1 训练样本格式

每个训练样本是一组论文：

```json
{
  "query": "...",
  "criteria": {
    "inclusion": [],
    "exclusion": []
  },
  "candidates": [
    {
      "paper_id": "p1",
      "title": "...",
      "abstract": "...",
      "label": "SELECT",
      "criteria_labels": {
        "I1": 1,
        "I2": 1,
        "E1": 0
      }
    },
    {
      "paper_id": "p2",
      "title": "...",
      "abstract": "...",
      "label": "REJECT",
      "criteria_labels": {
        "I1": 0,
        "I2": 1,
        "E1": 1
      }
    }
  ],
  "target_ranking": ["p1", "p3", "p4", "p2"]
}
```

### 8.2 样本比例

每个候选列表建议包含：

```text
20%～30% 正例
20%～30% 边界例
40%～60% 负例
```

比例不应固定不变，训练时需要随机变化，避免模型利用列表中正例数量等伪特征。

### 8.3 难负例设计

重点收集以下负样本：

- 关键词相同但研究任务不同；
- 研究任务相同但方法不满足要求；
- 摘要提到目标概念，但不是论文核心贡献；
- 通用 Web 搜索 Agent；
- 固定语料上的论文问答系统；
- 不含 LLM Agent 的传统学术推荐系统；
- 引用了目标论文，但自身研究问题不同；
- 被当前 Reranker 错误打高分的论文。

不能直接把“不在 benchmark gold 集合中的论文”全部视为负例，因为 gold 标注可能不完整。这类候选应经过人工、强模型或多模型一致性复核，避免假负例污染训练。

### 8.4 多任务损失

Reranker 联合优化三个目标：

\[
L_{reranker}
= \lambda_1 L_{listwise}
+ \lambda_2 L_{boundary}
+ \lambda_3 L_{criteria}
\]

- `L_listwise`：使直接相关论文排在边界和负例之前；
- `L_boundary`：预测 `SELECT / BORDERLINE / REJECT`；
- `L_criteria`：逐项判断纳入和排除条件。

初始权重可以设置为：

```text
lambda_1 = 0.5
lambda_2 = 0.3
lambda_3 = 0.2
```

排序关系为：

```text
直接相关论文 > 部分相关论文 > 难负例 > 随机负例
```

## 9. 正负反馈分析

### 9.1 正反馈

从 `SELECT` 论文中提取：

- 方法名称；
- 数据集和 benchmark；
- 作者和研究团队；
- 任务的同义表达；
- 可扩展的引用和被引论文。

```json
{
  "positive_expansions": [
    "comprehensive academic paper search",
    "scholar paper retrieval agent",
    "RefChain",
    "PaperFindingBench"
  ]
}
```

### 9.2 负反馈

从 `REJECT` 论文中聚合高频错误模式：

```json
{
  "negative_patterns": [
    {
      "type": "FIXED_CORPUS_QA",
      "count": 12,
      "trigger_terms": ["question answering", "given documents"]
    },
    {
      "type": "GENERAL_WEB_SEARCH",
      "count": 8,
      "trigger_terms": ["web browsing", "search engine agent"]
    }
  ]
}
```

### 9.3 边界反馈

```json
{
  "borderline_needs": [
    {
      "paper_id": "p023",
      "missing_evidence": "摘要未说明是否自主生成搜索查询",
      "next_action": "READ_FULLTEXT"
    }
  ]
}
```

## 10. 迭代查询演化

第二轮 Reasoner 读取：

```text
原始 Query
+ 纳入与排除标准
+ 正选论文概要
+ 正例扩展概念
+ 高频负选模式
+ 边界论文的缺失证据
+ 搜索历史和剩余预算
```

输出示例：

```json
{
  "actions": [
    {
      "type": "KEYWORD_SEARCH",
      "query": "RefChain scholar paper retrieval",
      "feedback_source": "positive",
      "purpose": "沿正选论文中的方法名继续扩展"
    },
    {
      "type": "CITATION_FORWARD",
      "seed_paper": "p001",
      "feedback_source": "positive",
      "purpose": "寻找引用核心系统的后续工作"
    },
    {
      "type": "QUERY_REWRITE",
      "query": "autonomous academic paper retrieval agent",
      "exclude_concepts": [
        "fixed corpus question answering",
        "general web search"
      ],
      "feedback_source": "negative",
      "purpose": "降低固定语料问答和通用 Web 搜索噪声"
    }
  ],
  "stop": false
}
```

部分学术 API 不支持负关键词语法，因此 `exclude_concepts` 既可用于搜索表达式，也可作为候选池的规则预过滤条件。

## 11. 停止策略

系统不能因为找到一两篇相关论文就停止，也不能无限扩大结果集。停止条件应综合搜索边际收益、噪声比例、重复率与成本预算。

可以在满足以下任意两项时停止：

1. 连续两轮新增 `SELECT` 论文数不超过 1；
2. 新候选与已有候选的重复率不低于 80%；
3. 新候选中 `REJECT` 比例不低于 85%；
4. 估计的 F1 或 Recall 增益连续两轮低于阈值；
5. 已达到最大搜索轮数或 API 调用预算。

当高分 `BORDERLINE` 数量较多时，不应直接停止，应优先复核其中最可能提升 Recall 的论文。

## 12. Reasoner 训练

### 12.1 监督微调样本

Reasoner 的训练映射为：

```text
Query + Search History + Positive/Negative/Borderline Feedback
    -> Next Search Actions or STOP
```

重点训练：

- 第一轮多样化搜索词生成；
- 从正选论文中提取扩展线索；
- 根据负选模式修正查询；
- 根据边界项选择验证方式；
- 搜索收益下降时及时停止。

### 12.2 奖励函数

后续可使用强化学习优化：

\[
R =
\alpha \Delta Recall
+ \beta \Delta F1
+ \gamma N_{new\_positive}
- \delta N_{new\_negative}
- \eta Cost
\]

其中：

- `Delta Recall`：本轮新发现 gold 论文带来的召回提升；
- `Delta F1`：本轮对最终集合 F1 的提升；
- `N_new_positive`：新增正例数量；
- `N_new_negative`：新增噪声数量；
- `Cost`：API 调用、Token 和延迟成本。

比赛主要指标是 F1，因此奖励应以 `Delta F1` 为主，同时保留召回和成本信号。

## 13. 训练流程

### 阶段一：训练 Reranker

```text
Benchmark Query + Gold Papers
    -> BM25 / Embedding / Reasoner 召回混合候选
    -> 构造正例、边界例和难负例
    -> 训练 Listwise Reranker
```

应优先训练稳定的正负边界。否则 Reasoner 搜索范围越广，系统噪声越严重。

### 阶段二：训练 Reasoner

使用 benchmark、学术 API 和教师模型生成搜索轨迹：

```text
Query
-> 第一轮搜索动作
-> 候选论文
-> Reranker 正负分类
-> 第二轮搜索动作
-> 最终结果
```

对轨迹进行质量过滤后用于监督微调。

### 阶段三：在线难负例挖掘

持续收集：

- Reranker 高分但实际不相关的论文；
- Reasoner 频繁召回的错误类型；
- 被错误标记为 `REJECT` 的 gold 论文；
- 不同 Query 下反复出现的通用噪声论文。

将这些样本加入后续训练迭代。

### 阶段四：联合优化

先固定 Reranker 或使用较低学习率，重点优化 Reasoner 的搜索动作。待搜索策略稳定后，再进行小步联合训练，避免两个模型同时快速变化导致训练目标漂移。

## 14. 推理阶段算法

```text
输入：用户 Query、最大轮数 R、API 调用预算 B

1. Query Analyzer 生成纳入和排除标准。
2. Reasoner 生成第一轮多样化搜索动作。
3. Search Executor 执行检索并建立候选池。
4. 对候选论文去重、硬过滤和粗排。
5. Listwise Reranker 输出 SELECT、BORDERLINE、REJECT。
6. Feedback Analyzer 总结正选概念、负选模式和边界缺失证据。
7. 判断停止条件：
   - 若满足，执行最终重排并返回结果；
   - 若不满足，Reasoner 生成下一轮动作并返回步骤 3。
8. 最终输出高度相关、部分相关论文及搜索统计信息。
```

## 15. 最终结果结构

```json
{
  "highly_relevant": [
    {
      "title": "...",
      "year": 2025,
      "relevance": 0.96,
      "matched_criteria": ["I1", "I2", "I3"],
      "reason": "直接研究自主学术论文检索 Agent"
    }
  ],
  "partially_relevant": [
    {
      "title": "...",
      "relevance": 0.67,
      "missing_criteria": ["I3"],
      "reason": "研究论文检索，但摘要未体现查询迭代"
    }
  ],
  "search_summary": {
    "rounds": 3,
    "api_calls": 8,
    "retrieved_count": 146,
    "selected_count": 18,
    "rejected_count": 112,
    "borderline_count": 16
  }
}
```

## 16. 实验设计

### 16.1 对比基线

1. 单轮关键词搜索；
2. 单轮搜索 + Pointwise Reranker；
3. 多轮 Reasoner + Pointwise Reranker；
4. 多轮 Reasoner + 普通 Listwise Reranker；
5. 多轮 Reasoner + 双边界 Listwise Reranker；
6. 完整 PN-Search：双边界 Reranker + 正负反馈查询演化。

### 16.2 消融实验

| 实验 | 移除的模块 | 目的 |
|---|---|---|
| w/o negative criteria | 不显式建模排除条件 | 验证排除规则对 Precision 和 F1 的影响 |
| w/o negative feedback | 负选结果不反馈给 Reasoner | 验证负反馈能否降低后续搜索噪声 |
| w/o borderline | 使用正负二分类 | 验证边界类别能否减少 gold 论文误杀 |
| w/o listwise | 每篇论文独立评分 | 验证候选集合上下文的价值 |
| w/o hard negatives | 不使用难负例 | 验证难负例对决策边界的影响 |
| w/o iterative search | 仅执行第一轮检索 | 验证多轮查询演化的效果 |

### 16.3 评测指标

主要效果指标：

```text
Precision
Recall
F1
Precision@K
Recall@K
```

错误分析指标：

```text
False Negative Rate
False Positive Rate
BORDERLINE 转化为 SELECT 的比例
难负例误选率
```

效率指标：

```text
平均 API 调用次数
平均 Token 消耗
平均搜索轮数
平均端到端延迟
Recall / API Call
F1 / Token
```

## 17. 最小可行版本

第一版建议限制为三轮搜索：

```text
第一轮：Reasoner 生成 3 个互补查询，每个召回 Top 20
第二步：去重和 Embedding 粗排，保留 Top 40
第三步：每组 8～10 篇执行 Listwise Reranker
第四步：聚合正选和负选反馈
第二轮：最多生成 2 个补充搜索动作
第三轮：只执行引用扩展或边界论文复核
最终：跨批次重排，输出 Top-K 论文
```

第一版先验证：

1. Reasoner 是否确实提高 Recall；
2. 双边界 Reranker 是否减少高相似难负例；
3. 负反馈是否降低下一轮的 `REJECT` 比例；
4. `BORDERLINE` 是否减少 gold 论文被错误删除；
5. 最终 F1 提升是否足以抵消额外推理成本。

## 18. 方案摘要

PN-Search 采用 Recall-first 的搜索策略，由 Reasoner 生成多样化检索动作以扩大候选论文集合；双边界 Listwise Reranker 基于显式的纳入和排除条件，将候选论文划分为正选、边界和负选集合；随后系统使用正选论文扩展检索概念，使用负选论文识别查询歧义并抑制噪声，使用边界论文触发定向证据验证，从而迭代优化学术论文搜索，在高召回、高精度和低成本之间取得平衡。

该方案的主要贡献为：

1. 将复杂学术检索显式解耦为高召回搜索与高精度边界判定；
2. 提出同时建模纳入与排除条件的双边界 Listwise Reranker；
3. 将负选论文转化为查询演化信号，而非简单丢弃；
4. 引入 `BORDERLINE` 机制，降低摘要信息不足造成的 gold 论文误杀；
5. 通过边际收益和成本约束实现自适应停止。
