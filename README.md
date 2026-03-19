# Danbooru 标签工程自动化管理工具

一个面向 Danbooru 图库标签数据的全链路自动化处理管线。从 SQLite 原始数据出发，经过 Wiki 抓取、LLM 翻译增强、共现矩阵构建，最终生成可供下游任务（如标签推荐、语义搜索）直接消费的高质量中文标签库与共现关系图。

---

## 目录

- [项目概览](#项目概览)
- [环境配置](#环境配置)
- [目录结构](#目录结构)
- [配置文件 config.yaml](#配置文件-configyaml)
- [模块详解](#模块详解)
  - [Step 1 — sync_tags.py](#step-1--sync_tagspy)
  - [Step 2 — fetch_wiki.py](#step-2--fetch_wikipy)
  - [Step 3 — llm_processor.py](#step-3--llm_processorpy)
  - [Step 4 — fetch_cooc.py](#step-4--fetch_coocpy)
  - [Step 5 — trim_cooc.py](#step-5--trim_coocpy)
- [CLI 用法](#cli-用法)
- [数据流总览](#数据流总览)
- [断点续传机制](#断点续传机制)
- [常见问题](#常见问题)

---

## 项目概览

```
SQLite 标签库
    │
    ▼ Step 1: sync_tags     ← 同步标签基础信息，新增符合条件的标签
    │
    ▼ Step 2: fetch_wiki    ← 增量抓取 Danbooru Wiki 页面
    │
    ▼ Step 3: llm_processor ← LLM 翻译、中文名修正、NSFW 判定
    │
    ▼ Step 4: fetch_cooc    ← 抓取全量标签共现关系矩阵
    │
    ▼ Step 5: trim_cooc     ← PMI 计算 + Top-K 截断，输出精简共现图
```

每个步骤均可单独执行，也可通过 `pipeline` 命令一键串联。

---

## 环境配置

### 依赖安装

```bash
pip install pandas pyarrow requests openai click python-dotenv python-dateutil
```

### 环境变量

在项目根目录创建 `.env` 文件，填入以下凭证：

```env
# Danbooru API 凭证（用于 Wiki 抓取和共现矩阵抓取）
DANBOORU_USER_NAME=your_username
DANBOORU_API_KEY=your_api_key

# OpenAI 兼容接口（用于 LLM 处理步骤）
OPENAI_API_KEY=your_llm_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # 或其他兼容端点

# Bangumi API Token（可选，用于实体标签的中文名精准查询）
BANGUMI_TOKEN=your_bangumi_token
```

> **注意**：`DANBOORU_USER_NAME` 和 `DANBOORU_API_KEY` 是抓取步骤的硬性要求。LLM 步骤额外需要 `OPENAI_API_KEY`。

---

## 目录结构

```
project_root/
├── main.py                  # CLI 入口
├── config.yaml              # 全局配置
├── .env                     # 凭证（不纳入版本控制）
├── modules/
│   ├── sync_tags.py
│   ├── fetch_wiki.py
│   ├── llm_processor.py
│   ├── fetch_cooc.py
│   └── trim_cooc.py
└── data/
    ├── raw/
    │   ├── tag.sqlite           # 输入：原始标签数据库
    │   └── cooccurrence_matrix.csv  # 输出：原始共现矩阵
    ├── checkpoint/              # 断点续传文件
    │   ├── llm_history.json
    │   ├── llm_temp.jsonl
    │   ├── wiki_progress.txt
    │   ├── cooc_progress.txt
    │   └── cooc_history.json
    └── processed/
        ├── tags_enhanced.csv        # 输出：增强后的标签主表
        ├── wiki_pages.parquet       # 输出：Wiki 数据库
        └── cooccurrence_clean.parquet  # 输出：清洗后的共现图
```

---

## 配置文件 config.yaml

所有模块通过统一的 `config.yaml` 管理路径和超参数，无需修改代码即可调整行为。

```yaml
paths:
  raw:
    sqlite_db: "data/raw/tag.sqlite"          # 上游 SQLite 数据库
    cooc_raw_csv: "data/raw/cooccurrence_matrix.csv"  # 原始共现矩阵

  checkpoint:
    llm_history: "data/checkpoint/llm_history.json"   # LLM 处理历史（永久豁免表）
    llm_temp: "data/checkpoint/llm_temp.jsonl"         # LLM 崩溃恢复临时文件
    wiki_progress: "data/checkpoint/wiki_progress.txt" # Wiki 抓取断点
    cooc_progress: "data/checkpoint/cooc_progress.txt" # 共现抓取断点（会话内）
    cooc_history: "data/checkpoint/cooc_history.json"  # 共现抓取历史（永久豁免表）

  processed:
    tags_enhanced: "data/processed/tags_enhanced.csv"
    wiki_parquet: "data/processed/wiki_pages.parquet"
    cooc_clean: "data/processed/cooccurrence_clean.parquet"

settings:
  llm:
    model_name: "x-ai/grok-4.1-fast"   # 任何 OpenAI 兼容模型均可
    batch_size: 10                       # 每批提交给 LLM 的标签数量

  cooc_trim:
    top_k: 50       # 每个标签保留的最强关联标签数
    min_pmi: 1.0    # PMI 过滤阈值
```

---

## 模块详解

### Step 1 — `sync_tags.py`

**功能**：将 SQLite 数据库中的标签变更同步到主标签表 `tags_enhanced.csv`，包括新增标签和更新已有标签的元数据。

**核心行为**：

- **读取源数据**：从 `tag.sqlite` 的 `tags` 表中查询 `name`、`category`、`cn_name`、`post_count` 四个字段。
- **筛选新增标签**：仅将满足以下条件的标签加入主表：
  - `post_count >= 100`（帖子量足够大，过滤噪声标签）
  - `category` 属于 `0`（通用）、`3`（版权/作品）、`4`（角色）
  - 在现有主表中尚不存在
- **保护人工成果**：对已存在的标签，仅更新 `post_count` 和 `category` 两个字段，**绝不覆盖** `cn_name`（此字段可能已由 LLM 精心修正）。
- **新标签的中文名**：直接继承 SQLite 中的 `cn_name`（如有），而非留空，为后续 LLM 步骤提供初始参照。

**输入**：`data/raw/tag.sqlite`、`data/processed/tags_enhanced.csv`（若存在）

**输出**：更新后的 `data/processed/tags_enhanced.csv`

---

### Step 2 — `fetch_wiki.py`

**功能**：从 Danbooru API 增量抓取 Wiki 页面，构建并维护本地 Wiki 数据库。

**核心行为**：

- **增量检测**：启动时读取本地 `wiki_pages.parquet`，找出最新的 `updated_at` 时间戳。抓取时遇到早于此时间的条目即停止，实现精准增量更新。
- **首次运行**：若本地无数据库，以 `2000-01-01` 为起点进行全量抓取。
- **翻页限制突破**：Danbooru API 单次分页上限为约 1000 页（每页 100 条）。当页码逼近上限（`> 900`）时，自动将时间轴重置为当前页最后一条记录的 `updated_at`，再从第 1 页重新翻取，从而突破分页限制，不丢失任何历史数据。
- **断点续传**：每 20 页保存一次检查点（页码 + 当前时间轴上限）到 `wiki_progress.txt`。重启后自动回退 2 页（防止边界数据丢失）并恢复抓取。
- **数据合并**：以 `id` 为主键，新数据覆盖旧数据（`keep='last'`），最终以 Parquet 格式持久化，兼顾读取速度与存储效率。
- **频率控制**：请求间隔 1.5~2.5 秒（含随机抖动）；遇到 429 自动休眠 3 分钟；每 20 页强制休息 60 秒。

**输入**：Danbooru API `/wiki_pages.json`

**输出**：`data/processed/wiki_pages.parquet`

---

### Step 3 — `llm_processor.py`

**功能**：调用大语言模型对标签进行中文化增强，包括翻译、中文名修正、中文 Wiki 生成和 NSFW 判定。同时通过 Bangumi API 为角色/作品类标签补充权威参考信息。

**核心行为**：

#### 处理目标筛选

每次运行时，将主表中所有标签分为三类：

1. **已完成**（在 `llm_history.json` 中）：跳过，永久豁免。
2. **实体标签**（`category` 为 3 或 4，且本地 Wiki 数据库中无记录）：进入「防幻觉重写」流程。
3. **普通标签**（`wiki` 字段为空或过短）：进入「常规翻译 / 无 Wiki 兜底」流程。

#### 三种 LLM 处理模式

| 模式 | 触发条件 | 特点 |
|------|----------|------|
| `general`（常规翻译）| 普通标签 + 有可用 Wiki 描述 | 基于英文 Wiki 翻译，temperature=0.4 |
| `fallback`（无 Wiki 兜底）| 普通标签 + 无任何 Wiki 描述 | 纯粹依赖模型知识库，temperature=0.5 |
| `entity`（防幻觉重写）| 角色/作品类标签 | 结合 Bangumi API 权威数据，防止模型生成错误中文名 |

#### 每个标签的输出字段

- **`cn_name`**：基础中文名 + 扩展同义词/别名，以逗号拼接（如：`蝶祈祷,罪恶王冠,祈妹`）
- **`wiki`**：50 字左右的中文视觉描述
- **`nsfw`**：`0` 或 `1`

#### Wiki 来源优先级

1. 本地 `wiki_pages.parquet`（批量预取）
2. 向 Danbooru API 发起单点精准查询（仅在本地未命中时触发）
3. 纯依赖 LLM 知识库（兜底）

#### 容灾机制

- **临时文件**：每批次处理结果实时追加到 `llm_temp.jsonl`，进程意外终止后下次启动自动合并恢复。
- **历史豁免表**：成功处理的标签名写入 `llm_history.json`，重启后不会重复处理，节省 API 费用。
- **预览模式**（`--preview`）：仅统计本次待处理数量，不发起任何 API 请求。

**输入**：`tags_enhanced.csv`、`wiki_pages.parquet`、Bangumi API、LLM API

**输出**：就地更新 `tags_enhanced.csv`，更新 `llm_history.json`

---

### Step 4 — `fetch_cooc.py`

**功能**：遍历目标标签，通过 Danbooru 的 `related_tag` API 抓取两两共现关系，构建原始共现矩阵。支持**增量模式**（默认）和**全量更新模式**（`--full`）。

#### 两种运行模式

| 模式 | 触发方式 | 行为 |
|------|----------|------|
| **增量模式**（默认）| `python main.py fetch-cooc` | 跳过历史记录中已抓取的标签，仅处理新增标签；新抓数据与旧共现矩阵合并后去重 |
| **全量更新模式** | `python main.py fetch-cooc --full` | 无视历史记录，重新抓取所有标签；完成后直接覆写旧共现矩阵 |

#### 核心行为

- **历史豁免（增量核心）**：每次成功抓取的标签名实时写入 `cooc_history.json`。增量模式下，启动时从历史文件读取已处理标签集合，将其从本次任务列表中剔除，避免重复请求。全量模式下，历史文件被忽略（但不会删除），断点文件会被强制清除。
- **共现次数还原**：Danbooru API 返回的是共现**频率**（`frequency`，介于 0~1 的比例），而非绝对次数。模块通过 `raw_count = round(frequency × post_count_of_query_tag)` 将其还原为近似的绝对共现次数。
- **兼容多种响应格式**：`parse_related_tags` 函数同时处理字典格式（含 `related_tags` 键）和列表格式两种 API 响应结构。
- **自动过滤**：仅保留双方均在目标标签集合内的配对，过滤掉非目标标签的噪声关系。
- **标准化存储**：每对 `(tag_a, tag_b)` 按字典序排列，避免 `(A, B)` 和 `(B, A)` 重复存储。
- **断点续传**：每处理 100 个标签，将本批结果追加到临时 CSV，同时更新 `cooc_progress.txt`（会话内进度）和 `cooc_history.json`（跨会话历史），并强制休息 30 秒。
- **收尾合并**：任务结束后读取临时 CSV；增量模式下与已有的 `cooccurrence_matrix.csv` 合并，全量模式下直接覆写。合并完成后按 `(raw_count, cosine_similarity)` 降序全局去重。
- **频率控制**：正常请求间隔 1.5 秒；429 休眠 3 分钟；网络异常休眠 60 秒后重试。

**输入**：`tags_enhanced.csv`、Danbooru API `/related_tag.json`

**输出**：`data/raw/cooccurrence_matrix.csv`（列：`tag_a`, `tag_b`, `raw_count`, `cosine_similarity`）

---

### Step 5 — `trim_cooc.py`

**功能**：对原始共现矩阵进行 PMI（逐点互信息）计算和 Top-K 截断，将庞大的原始矩阵精简为高质量的稀疏共现图。

**核心行为**：

#### PMI 计算

使用经典 PMI 公式：

```
PMI(a, b) = log₂( P(a,b) / (P(a) × P(b)) )
           ≈ log₂( cooc(a,b) × D / (count(a) × count(b)) )
```

其中 `D` 取标签表中 `post_count` 的最大值作为总文档数的近似估计。

#### Top-K 截断

1. 按 `min_pmi` 阈值过滤低质量共现对。
2. 将每条边**双向展开**为有向边，对每个源节点按 `(PMI 降序, 共现次数降序)` 排列。
3. 每个节点仅保留前 `top_k` 条出边。
4. 将保留的有向边重新折叠回无向边（字典序规范化），去重后输出。

最终结果仅保留 `tag_a`、`tag_b`、`count` 三列，以 Snappy 压缩的 Parquet 格式存储。

#### Dry-Run 测试模式

使用 `--dry-run` 标志可进入测试模式，该模式会：

- 遍历 PMI 阈值 1 ~ 5，依次模拟过滤和截断操作
- 以表格形式输出每个阈值下「PMI 过滤后剩余行数」和「Top-K 截断后最终行数」
- **不向磁盘写入任何文件**，用于在正式执行前确定合理的参数组合

示例输出：
```
PMI 阈值   | PMI过滤后剩余行数    | Top-K截断后最终行数
>= 1.0     | 1,234,567            | 456,789
>= 2.0     | 789,012              | 321,456
...
```

**输入**：`cooccurrence_matrix.csv`、`tags_enhanced.csv`

**输出**：`data/processed/cooccurrence_clean.parquet`（列：`tag_a`, `tag_b`, `count`）

---

## CLI 用法

所有命令均通过 `main.py` 入口调用：

```bash
# 查看帮助
python main.py --help

# 逐步执行
python main.py sync-tags         # 步骤 1
python main.py fetch-wiki        # 步骤 2
python main.py llm-process       # 步骤 3
python main.py llm-process --preview  # 步骤 3（预览模式，不消耗 API）
python main.py fetch-cooc        # 步骤 4（增量模式，跳过已抓取标签）
python main.py fetch-cooc --full # 步骤 4（全量模式，重新抓取所有标签）
python main.py trim-cooc         # 步骤 5

# 步骤 5 的高级选项
python main.py trim-cooc --top-k 30 --min-pmi 2.0      # 覆盖 config.yaml 参数
python main.py trim-cooc --dry-run                       # 参数测试，不写盘

# 一键全流程
python main.py pipeline
```

> CLI 命令名中的下划线（`_`）和连字符（`-`）均可识别（Click 标准行为）。

---

## 数据流总览

```
data/raw/tag.sqlite
        │
        │ sync_tags
        ▼
data/processed/tags_enhanced.csv  ◄──────────────────┐
        │                                              │
        │ fetch_wiki                                   │
        ▼                                              │
data/processed/wiki_pages.parquet                      │
        │                                              │
        │ llm_processor (读取 wiki_pages + 标签表)     │
        └──────────────────────────────────────────────┘
                   (就地更新 tags_enhanced.csv)

data/processed/tags_enhanced.csv
        │
        │ fetch_cooc
        ▼
data/raw/cooccurrence_matrix.csv
        │
        │ trim_cooc
        ▼
data/processed/cooccurrence_clean.parquet
```

---

## 断点续传机制

本项目所有涉及网络请求的长耗时任务均内置断点续传，可安全应对程序崩溃、网络中断或手动中止（Ctrl+C）。

| 模块 | 检查点文件 | 恢复粒度 |
|------|-----------|---------|
| `fetch_wiki` | `wiki_progress.txt` | 每 20 页（含时间轴上限） |
| `fetch_cooc` | `cooc_progress.txt` | 每 100 个标签（会话内断点） |
| `fetch_cooc` | `cooc_history.json` | 永久历史豁免（跨会话，增量模式使用） |
| `llm_processor` | `llm_temp.jsonl` | 每个 Batch 结束后 |
| `llm_processor` | `llm_history.json` | 永久历史豁免（跨会话） |

所有检查点文件在对应任务正常完成后会被**自动清理**。若希望强制从头重新运行某个步骤，手动删除对应检查点文件即可。

---

## 常见问题

**Q：为什么 `sync_tags` 没有更新某些标签的 `cn_name`？**

这是设计行为。`cn_name` 字段由 LLM 步骤精心生成和修正，`sync_tags` 刻意不覆盖此字段，以保护人工/模型成果。如需强制重置某标签的中文名，请直接编辑 `tags_enhanced.csv`，并从 `llm_history.json` 中删除对应条目。

**Q：LLM 步骤消耗了多少 API 调用？**

使用 `python main.py llm-process --preview` 可在不消耗任何 Token 的情况下预览本次待处理的标签数量，据此估算费用后再决定是否执行。

**Q：`trim_cooc` 的参数应该如何选取？**

先运行 `python main.py trim-cooc --dry-run`，查看不同 PMI 阈值下的数据保留量，选择在数据量和质量之间取得平衡的阈值，再正式执行。通常 `min_pmi=1.0` 搭配 `top_k=50` 是较为合理的起点。

**Q：`fetch-cooc` 的增量模式和全量模式应该如何选择？**

日常运行（如在 `sync_tags` 新增了若干标签后）使用默认的**增量模式**即可，程序会自动识别 `cooc_history.json` 中尚未覆盖的新标签，只为它们发请求，节省大量时间和 API 配额。当标签库经历了大规模重构、或怀疑历史共现数据已严重过时时，才需要使用 `--full` 强制全量重建。注意全量模式会覆写整张共现矩阵，建议事先备份 `cooccurrence_matrix.csv`。

**Q：遭遇 Danbooru 429 限流怎么办？**

程序会自动检测 429 响应并休眠 3 分钟后重试，无需手动干预。若频繁触发，可考虑在 `fetch_cooc.py` 中增大请求间隔（当前为 1.5 秒）。

**Q：如何只重新处理某些特定标签（跳过 LLM 历史豁免）？**

从 `data/checkpoint/llm_history.json` 中删除对应标签名，并清空该标签在 `tags_enhanced.csv` 中的 `wiki` 字段，再重新运行 `llm-process` 即可。