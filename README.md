# Danbooru 标签工程自动化管理工具

一个面向 Danbooru 图库标签数据的全链路自动化处理管线。从 SQLite 原始数据出发，经过 Wiki 抓取、标签组获取、LLM 翻译增强、标签共现矩阵构建、画师共现矩阵构建，最终生成可供下游任务（如标签推荐、语义搜索）直接消费的高质量中文标签库与共现关系图。

---

## 目录

- [项目概览](#项目概览)
- [环境配置](#环境配置)
- [目录结构](#目录结构)
- [配置文件 config.yaml](#配置文件-configyaml)
- [数据文件结构](#数据文件结构)
- [模块详解](#模块详解)
- [独立工具](#独立工具)
- [CLI 用法](#cli-用法)
- [数据流总览](#数据流总览)
- [断点续传机制](#断点续传机制)
- [常见问题](#常见问题)

---

## 项目概览

```
SQLite 标签库
    │
    ▼ Step 1: sync_tags          ← 同步标签基础信息，新增符合条件的标签
    │
    ▼ Step 2: fetch_tag_groups   ← 从 Danbooru Wiki 获取标签组（分组）信息
    │
    ▼ Step 3: fetch_wiki         ← 增量抓取 Danbooru Wiki 页面
    │
    ▼ Step 4: llm_processor      ← LLM 翻译、中文名修正、NSFW 判定
    │
    ▼ Step 5: fetch_cooc         ← 抓取全量标签共现关系矩阵
    │
    ▼ Step 6: fetch_artist_cooc  ← 抓取标签-画师共现关系
    │
    ▼ Step 7: trim_cooc          ← PMI 计算 + Top-K 截断，输出精简共现图
    │
    ▼ Step 8: trim_artist_cooc   ← NPMI 计算 + Top-K 截断，输出画师共现表
```

每个步骤均可单独执行，也可通过 `pipeline` 命令一键串联全部 8 个步骤。`pipeline` 命令在执行前会自动备份 `data/` 文件夹。

---

## 环境配置

### 依赖安装

```bash
pip install pandas pyarrow requests openai click python-dotenv python-dateutil
```

部分辅助工具还需要额外依赖：

```bash
pip install lxml                                    # Bangumi API 解析
```

### 环境变量

在项目根目录创建 `.env` 文件，填入以下凭证：

```env
# Danbooru API 凭证（用于 Wiki 抓取、标签组获取和共现矩阵抓取）
DANBOORU_USER_NAME=your_username
DANBOORU_API_KEY=your_api_key

# OpenAI 兼容接口（用于 LLM 处理步骤）
OPENAI_API_KEY=your_llm_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # 或其他兼容端点

# Bangumi API Token（可选，用于实体标签的中文名精准查询）
BANGUMI_ACCESS_TOKEN=your_bangumi_token
```

> **注意**：`DANBOORU_USER_NAME` 和 `DANBOORU_API_KEY` 是抓取步骤的硬性要求。LLM 步骤额外需要 `OPENAI_API_KEY`。

---

## 目录结构

可以在[此链接](https://pan.quark.cn/s/1e7371a136a7)下载我目前使用的 `data/` 数据文件夹。内含目前我正在使用的最新全套数据文件。

```
project_root/
├── main.py                         # CLI 入口
├── config.yaml                     # 全局配置
├── .env                            # 凭证（不纳入版本控制）
├── .gitignore
├── modules/                        # Pipeline 核心模块
│   ├── sync_tags.py                # Step 1: 同步标签基础信息
│   ├── fetch_tag_groups.py         # Step 2: 获取标签组
│   ├── fetch_wiki.py               # Step 3: 增量抓取 Wiki
│   ├── llm_processor.py            # Step 4: LLM 处理
│   ├── fetch_cooc.py               # Step 5: 抓取标签共现关系
│   ├── fetch_artist_cooc.py        # Step 6: 抓取标签-画师共现
│   ├── trim_cooc.py                # Step 7: 标签共现 PMI 降维截断
│   ├── trim_artist_cooc.py         # Step 8: 画师共现 NPMI 降维截断
│   └── parquet2csv.py              # 格式转换工具 (Parquet ↔ CSV)
├── data/                           # 管线数据（不纳入版本控制）
│   ├── raw/
│   │   ├── tag.sqlite              # 输入：原始标签数据库
│   │   └── cooccurrence_matrix.csv # 原始共现矩阵
│   ├── checkpoint/                 # 断点续传文件
│   │   ├── tags_sync.json
│   │   ├── llm_history.json
│   │   ├── llm_temp.jsonl
│   │   ├── wiki_progress.txt
│   │   ├── wiki_updated_tags.json
│   │   ├── cooc_progress.txt
│   │   ├── cooc_history.json
│   │   ├── artist_cooc_progress.txt
│   │   ├── artist_cooc_history.json
│   │   ├── artist_download.json
│   │   └── artist_quality_rank.json
│   └── processed/
│       ├── tags_enhanced.csv       # 输出：增强后的标签主表
│       ├── wiki_pages.parquet      # 输出：Wiki 数据库
│       ├── tag_groups.json         # 输出：标签组信息
│       ├── cooccurrence_clean.parquet  # 输出：清洗后的共现图
│       ├── tag_artist_cooc.parquet # 输出：标签-画师共现表
│       └── artist_quality_top1000.json # 输出：画师评分排名
└── image/                          # 画师图片下载目录
```

---

## 配置文件 config.yaml

所有模块通过统一的 `config.yaml` 管理路径和超参数，无需修改代码即可调整行为。

```yaml
paths:
  raw:
    sqlite_db: "data/raw/tag.sqlite"
    cooc_raw_csv: "data/raw/cooccurrence_matrix.csv"

  checkpoint:
    llm_history: "data/checkpoint/llm_history.json"
    llm_temp: "data/checkpoint/llm_temp.jsonl"
    wiki_progress: "data/checkpoint/wiki_progress.txt"
    wiki_updated_tags: "data/checkpoint/wiki_updated_tags.json"
    cooc_progress: "data/checkpoint/cooc_progress.txt"
    cooc_history: "data/checkpoint/cooc_history.json"
    artist_cooc_progress: "data/checkpoint/artist_cooc_progress.txt"
    artist_cooc_history: "data/checkpoint/artist_cooc_history.json"

  processed:
    tags_enhanced: "data/processed/tags_enhanced.csv"
    wiki_parquet: "data/processed/wiki_pages.parquet"
    tag_groups: "data/processed/tag_groups.json"
    cooc_clean: "data/processed/cooccurrence_clean.parquet"
    tag_artist_cooc: "data/processed/tag_artist_cooc.parquet"

settings:
  llm:
    model_name: "deepseek/deepseek-v4-flash"
    batch_size: 20

  cooc_trim:
    top_k: 50
    min_pmi: 1.0

  artist_cooc_trim:
    top_k: 50
    min_npmi: 0.15
```

---

## 数据文件结构

### 输入：`tag.sqlite`

上游数据库，由外部工具维护，本项目只读不写。本项目中此文件的来源是 [ffdkj/ffdkj-Danbooru_Tag-Chinese-English-Translation-Table](https://github.com/ffdkj/ffdkj-Danbooru_Tag-Chinese-English-Translation-Table?tab=readme-ov-file)。管线依赖其中的 `tags` 表：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | TEXT | 标签英文名（主键，唯一） |
| `category` | INTEGER | 标签分类：`0` 通用、`3` 版权/作品、`4` 角色（其余类型不纳入处理） |
| `post_count` | INTEGER | 该标签在 Danbooru 上的帖子数量 |
| `cn_name` | TEXT | 预置中文名（可为空，LLM 步骤会进一步修正和扩展） |

---

### 中间/输出：`tags_enhanced.csv`

整个管线最核心的数据文件，贯穿全部步骤，由 `sync_tags` 创建，由 `llm_processor` 持续丰富。

| 字段 | 来源步骤 | 说明 |
|------|----------|------|
| `name` | Step 1 | 标签英文名，全表唯一主键 |
| `category` | Step 1 | 标签分类（同 SQLite，随每次同步刷新） |
| `post_count` | Step 1 | 帖子数量（随每次同步刷新） |
| `cn_name` | Step 1 / Step 4 | 中文名。初始值继承自 SQLite；Step 4 执行后格式为「基础中文名,同义词1,别名2」的逗号拼接串 |
| `wiki` | Step 4 | LLM 生成的中文视觉描述，约 50 字。初始为空 |
| `nsfw` | Step 4 | NSFW 标记：`0` 安全、`1` 不安全。初始为 `0` |

**注意**：`cn_name` 字段在 Step 4 后会包含扩展词，格式示例：

```
original_name  →  蝶祈,罪恶王冠,祈妹
blue_eyes      →  蓝眼睛,碧眸,蓝瞳
```

---

### 中间/输出：`wiki_pages.parquet`

由 `fetch_wiki` 构建的本地 Wiki 镜像库，供 `llm_processor` 批量读取英文原文。以 `id` 为主键，字段直接来自 Danbooru API 响应：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | Wiki 页面 ID（主键） |
| `title` | TEXT | 页面标题，通常与标签英文名一致 |
| `body` | TEXT | Wiki 正文（Danbooru 自有标记语法，`llm_processor` 读取时会自动清洗） |
| `updated_at` | TEXT | 最后更新时间（ISO 8601），用于增量检测的时间基准 |
| `other_names` | TEXT | 别名列表（原为 JSON 数组，存储时序列化为字符串） |

> `body` 字段在传入 LLM 前会经过 `clean_wiki_text()` 清洗：去除 `[[链接]]` 标记语法、加粗符号、标题前缀，并截断至 400 字。

---

### 中间/输出：`tag_groups.json`

由 `fetch_tag_groups` 从 Danbooru Wiki 的 `tag_groups` 目录页爬取并解析的标签组信息，供 `llm_processor` 在翻译标签组中文名时使用。

| 字段 | 类型 | 说明 |
|------|------|------|
| `tag_to_groups` | `dict[str, list[str]]` | 标签名到所属组的映射。键为标签英文名，值为该标签所属的 tag group ID 列表 |
| `group_to_tags` | `dict[str, list[str]]` | 组到成员标签的映射。键为 tag group ID（如 `tag_group:eye_style`），值为组内所有成员标签名列表 |
| `group_cn_names` | `dict[str, str]` | 组 ID 到中文名的映射。初始为空字符串，由 `llm_processor` 的 `_fill_group_cn_names()` 翻译填充 |

> `tag_group` 是 Danbooru 提供的一种标签分类体系，将语义相近的标签归入同一组（如 `tag_group:eye_style` 下包含 `blue_eyes`、`red_eyes` 等）。此文件帮助下游任务理解标签之间的层级和分组关系。

---

### 中间：`cooccurrence_matrix.csv`

由 `fetch_cooc` 生成的原始标签共现矩阵，作为 `trim_cooc` 的输入。每行代表一对共现标签：

| 字段 | 类型 | 说明 |
|------|------|------|
| `tag_a` | TEXT | 共现对中字典序较小的标签名 |
| `tag_b` | TEXT | 共现对中字典序较大的标签名 |
| `raw_count` | INTEGER | 估算的绝对共现次数（由 `frequency × post_count` 还原） |
| `cosine_similarity` | FLOAT | Danbooru API 返回的余弦相似度（仅在字典格式响应中存在，列表格式下为 `0.0`） |

每对 `(tag_a, tag_b)` 保证 `tag_a < tag_b`（字典序），全表无重复配对。

---

### 输出：`cooccurrence_clean.parquet`

Step 7 的最终产物，经 PMI 过滤和 Top-K 截断后的高质量稀疏共现图，以 Snappy 压缩存储：

| 字段 | 类型 | 说明 |
|------|------|------|
| `tag_a` | TEXT | 共现对中字典序较小的标签名 |
| `tag_b` | TEXT | 共现对中字典序较大的标签名 |
| `count` | INTEGER | 共现次数（与原始矩阵中的 `raw_count` 含义相同） |

---

### 输出：`tag_artist_cooc.parquet`

Step 8 的最终产物，经 NPMI 过滤和 Top-K 截断后的标签-画师共现表：

| 字段 | 类型 | 说明 |
|------|------|------|
| `artist` | TEXT | 画师名 |
| `tag` | TEXT | 关联标签名 |
| `npmi` | FLOAT | 归一化逐点互信息分数 |
| `count` | INTEGER | 共现次数 |

---

### 检查点文件

以下文件均位于 `data/checkpoint/`，由程序自动管理，正常情况下无需手动修改：

| 文件 | 格式 | 内容 |
|------|------|------|
| `wiki_progress.txt` | 纯文本 | 第一行：当前抓取页码；第二行（可选）：当前时间轴上限（ISO 8601） |
| `wiki_updated_tags.json` | JSON | 追踪 Wiki 更新过的标签 ID 集合，用于增量更新判断 |
| `cooc_progress.txt` | 纯文本 | 单个整数，表示本次任务已完成的标签索引位置 |
| `cooc_history.json` | JSON 数组 | 历史上所有已成功抓取过共现数据的标签名列表 |
| `artist_cooc_progress.txt` | 纯文本 | 画师共现抓取进度 |
| `artist_cooc_history.json` | JSON 数组 | 历史上已成功抓取过共现数据的画师名列表 |
| `llm_temp.jsonl` | JSON Lines | 每行一条 LLM 返回的标签处理结果，崩溃恢复时读取并合并入主表 |
| `llm_history.json` | JSON 数组 | 历史上所有已由 LLM 处理完毕的标签名列表（永久豁免） |
| `tags_sync.json` | JSON | `fetch_tags` 脚本的最后同步时间戳，用于增量抓取 |

---

## 模块详解

### Step 1 — `sync_tags.py`

**功能**：将 SQLite 数据库中的标签变更同步到主标签表 `tags_enhanced.csv`，包括新增标签和更新已有标签的元数据。

**核心行为**：

- **下载 SQLite**：从 GitHub 自动下载最新的 `tag.sqlite` 数据库文件。
- **筛选新增标签**：仅将满足以下条件的标签加入主表：
  - `post_count >= 100`（帖子量足够大，过滤噪声标签）
  - `category` 属于 `0`（通用）、`3`（版权/作品）、`4`（角色）
  - 在现有主表中尚不存在
- **保护人工成果**：对已存在的标签，仅更新 `post_count` 和 `category` 两个字段，**绝不覆盖** `cn_name`（此字段可能已由 LLM 精心修正）。
- **新标签的中文名**：直接继承 SQLite 中的 `cn_name`（如有），而非留空，为后续 LLM 步骤提供初始参照。

**输入**：`data/raw/tag.sqlite`、`data/processed/tags_enhanced.csv`（若存在）

**输出**：更新后的 `data/processed/tags_enhanced.csv`

---

### Step 2 — `fetch_tag_groups.py`

**功能**：从 Danbooru Wiki 的 `tag_groups` 目录页爬取所有标签组信息，解析每个组的成员标签，输出标签到组、组到标签的双向映射，并预留组中文名字段供后续 LLM 翻译填充。

**核心行为**：

- **抓取主目录页**：向 Danbooru Wiki API 查询 `tag_groups` 页面，从正文中解析所有 `[[Tag group:xxx]]` 格式的组标题链接。
- **逐个解析成员**：对每个 tag group 页面发起精确查询，从正文的 `[[tag_name]]` 链接中提取成员标签名，并自动过滤掉导航性 Wiki 页面引用。
- **保留已有翻译**：若输出文件已存在且包含 `group_cn_names`，增量运行时保留已有中文名，新 group 留空待后续 LLM 步骤翻译填充。
- **输出结构**：包含三个字段 —— `tag_to_groups`（标签→组映射）、`group_to_tags`（组→标签映射）、`group_cn_names`（组中文名，由 Step 4 的 `_fill_group_cn_names()` 填充）。

**输入**：Danbooru Wiki API

**输出**：`data/processed/tag_groups.json`

---

### Step 3 — `fetch_wiki.py`

**功能**：从 Danbooru API 增量抓取 Wiki 页面，构建并维护本地 Wiki 数据库。

**核心行为**：

- **增量检测**：启动时读取本地 `wiki_pages.parquet`，找出最新的 `updated_at` 时间戳。抓取时遇到早于此时间的条目即停止，实现精准增量更新。
- **首次运行**：若本地无数据库，以 `2000-01-01` 为起点进行全量抓取。
- **翻页限制突破**：Danbooru API 单次分页上限为约 1000 页（每页 100 条）。当页码逼近上限（`> 900`）时，自动将时间轴重置为当前页最后一条记录的 `updated_at`，再从第 1 页重新翻取，从而突破分页限制，不丢失任何历史数据。
- **断点续传**：每 20 页保存一次检查点（页码 + 当前时间轴上限）到 `wiki_progress.txt`。重启后自动回退 2 页（防止边界数据丢失）并恢复抓取。
- **数据合并**：以 `id` 为主键，新数据覆盖旧数据（`keep='last'`），最终以 Parquet 格式持久化。
- **频率控制**：请求间隔 1.5~2.5 秒（含随机抖动）；遇到 429 自动休眠 3 分钟；每 20 页强制休息 60 秒。

**输入**：Danbooru API `/wiki_pages.json`

**输出**：`data/processed/wiki_pages.parquet`

---

### Step 4 — `llm_processor.py`

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

- **`cn_name`**：基础中文名 + 扩展同义词/别名，以逗号拼接
- **`wiki`**：50 字左右的中文视觉描述
- **`nsfw`**：`0` 或 `1`

#### Wiki 来源优先级

1. 本地 `wiki_pages.parquet`（批量预取）
2. 向 Danbooru API 发起单点精准查询（仅在本地未命中时触发）
3. 纯依赖 LLM 知识库（兜底）

#### 辅助功能

- **Qualifier 匹配**（`_resolve_qualifier()`）：从 `character(franchise)` 格式的标签名中提取作品限定词
- **作品名补全**（`_patch_qualifier_cn_names()`）：LLM 处理后的兜底精确匹配
- **标签组中文名翻译**（`_fill_group_cn_names()`）：读取 `tag_groups.json`，调用 LLM 为每个 group ID 翻译中文名称
- **中文名提取**（`extract_chinese_from_other_names()`）：从 Wiki 的 `other_names` 字段提取纯中文别名

#### 容灾机制

- **临时文件**：每批次处理结果实时追加到 `llm_temp.jsonl`，进程意外终止后下次启动自动合并恢复。
- **历史豁免表**：成功处理的标签名写入 `llm_history.json`，重启后不会重复处理，节省 API 费用。
- **预览模式**（`--preview`）：仅统计本次待处理数量，不发起任何 API 请求。
- **调试模式**（`--debug`）：输出 Wiki 清洗、other_names 匹配、LLM 收发的详细过程。

**输入**：`tags_enhanced.csv`、`wiki_pages.parquet`、Bangumi API、LLM API

**输出**：就地更新 `tags_enhanced.csv`，更新 `llm_history.json`

---

### Step 5 — `fetch_cooc.py`

**功能**：遍历目标标签，通过 Danbooru 的 `related_tag` API 抓取两两共现关系，构建原始共现矩阵。支持**增量模式**（默认）和**全量更新模式**（`--full`）。

**警告**：全量更新模式耗时极长，约需 24 小时。

#### 两种运行模式

| 模式 | 触发方式 | 行为 |
|------|----------|------|
| **增量模式**（默认）| `python main.py fetch-cooc` | 跳过历史记录中已抓取的标签，仅处理新增标签；新抓数据与旧共现矩阵合并后去重 |
| **全量更新模式** | `python main.py fetch-cooc --full` | 无视历史记录，重新抓取所有标签；完成后直接覆写旧共现矩阵 |

#### 核心行为

- **历史豁免**：每次成功抓取的标签名实时写入 `cooc_history.json`。增量模式下跳过已处理标签。
- **共现次数还原**：通过 `raw_count = round(frequency × post_count_of_query_tag)` 还原为近似的绝对共现次数。
- **兼容多种响应格式**：同时处理字典格式和列表格式两种 API 响应结构。
- **自动过滤**：仅保留双方均在目标标签集合内的配对。
- **标准化存储**：每对 `(tag_a, tag_b)` 按字典序排列。
- **断点续传**：每处理 100 个标签保存进度，强制休息 30 秒。
- **频率控制**：正常请求间隔 1.5 秒；429 休眠 3 分钟；网络异常休眠 60 秒后重试。

**输入**：`tags_enhanced.csv`、Danbooru API `/related_tag.json`

**输出**：`data/raw/cooccurrence_matrix.csv`

---

### Step 6 — `fetch_artist_cooc.py`

**功能**：遍历高热度画师，抓取每位画师与其作品标签的共现关系，构建标签-画师共现矩阵。支持增量模式和全量模式。

**核心行为**：

- **画师筛选**：从 `tags_enhanced.csv` 中筛选 category=1（画师）的标签，按 post_count 降序排列。
- **共现抓取**：对每位画师，通过 `related_tag` API 获取其最常关联的标签及频率。
- **断点续传**：与标签共现抓取（Step 5）采用相同的检查点机制。

**输入**：`tags_enhanced.csv`、Danbooru API

**输出**：`data/processed/tag_artist_cooc.parquet`（原始，未经裁剪）

---

### Step 7 — `trim_cooc.py`

**功能**：对原始标签共现矩阵进行 PMI（逐点互信息）计算和 Top-K 截断，将庞大的原始矩阵精简为高质量的稀疏共现图。

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
4. 将保留的有向边重新折叠回无向边，去重后输出。

最终结果仅保留 `tag_a`、`tag_b`、`count` 三列，以 Snappy 压缩的 Parquet 格式存储。

#### Dry-Run 测试模式

使用 `--dry-run` 标志可进入测试模式，该模式会输出不同 PMI 阈值下的数据保留量报告，不向磁盘写入任何文件，用于在正式执行前确定合理的参数组合。

**输入**：`cooccurrence_matrix.csv`、`tags_enhanced.csv`

**输出**：`data/processed/cooccurrence_clean.parquet`

---

### Step 8 — `trim_artist_cooc.py`

**功能**：对 Step 6 生成的原始标签-画师共现表进行 NPMI（归一化逐点互信息）计算和 Top-K 截断。

**核心行为**：

- **NPMI 计算**：使用归一化 PMI 公式评估标签与画师的关联强度，分数归一化到 `[-1, 1]` 区间。
- **Top-K 截断**：每位画师按 `(NPMI 降序, 共现次数降序)` 仅保留前 `top_k` 条标签关联。
- **Dry-Run 测试**：支持 `--dry-run` 参数预览不同阈值下的保留量。

**输入**：`tag_artist_cooc.parquet`（原始）、`tags_enhanced.csv`

**输出**：`data/processed/tag_artist_cooc.parquet`（覆盖为裁剪后版本）

---

## CLI 用法

所有命令均通过 `main.py` 入口调用：

```bash
# 查看帮助
python main.py --help

# 逐步执行
python main.py sync-tags             # Step 1: 同步标签基础信息
python main.py fetch-tag-groups      # Step 2: 获取标签组信息
python main.py fetch-wiki            # Step 3: 增量抓取 Wiki
python main.py llm-process           # Step 4: LLM 处理
python main.py llm-process --preview # Step 4（预览模式，不消耗 API）
python main.py llm-process --debug   # Step 4（调试模式，输出详细过程）
python main.py fetch-cooc            # Step 5（增量模式，跳过已抓取标签）
python main.py fetch-cooc --full     # Step 5（全量模式，重新抓取所有标签）
python main.py fetch-artist-cooc     # Step 6（增量模式）
python main.py fetch-artist-cooc --full  # Step 6（全量模式）
python main.py trim-cooc             # Step 7: PMI 降维截断
python main.py trim-cooc --top-k 30 --min-pmi 2.0  # 覆盖参数
python main.py trim-cooc --dry-run   # 参数测试，不写盘
python main.py trim-artist-cooc      # Step 8: NPMI 降维截断
python main.py trim-artist-cooc --dry-run

# 一键全流程（自动备份 data/ 后顺序执行 Step 1-8）
python main.py pipeline
```

> **自动备份**：`pipeline` 命令启动时会将 `data/` 打包为 `data_backup_YYYYMMDD_HHMMSS.zip`，防止意外数据损坏。备份文件已在 `.gitignore` 中排除。

---

## 数据流总览

```
data/raw/tag.sqlite
        │
        │ Step 1: sync_tags
        ▼
data/processed/tags_enhanced.csv  ◄──────────────────────────┐
        │                                                      │
        │ Step 2: fetch_tag_groups → tag_groups.json           │
        │                                                      │
        │ Step 3: fetch_wiki                                    │
        ▼                                                      │
data/processed/wiki_pages.parquet                              │
        │                                                      │
        │ Step 4: llm_processor (读取 wiki + tag_groups)       │
        └──────────────────────────────────────────────────────┘
                   (就地更新 tags_enhanced.csv)

data/processed/tags_enhanced.csv
        │
        ├── Step 5: fetch_cooc
        │       ▼
        │   data/raw/cooccurrence_matrix.csv
        │       │
        │       │ Step 7: trim_cooc
        │       ▼
        │   data/processed/cooccurrence_clean.parquet
        │
        └── Step 6: fetch_artist_cooc
                │
                │ Step 8: trim_artist_cooc
                ▼
            data/processed/tag_artist_cooc.parquet
```

---

## 断点续传机制

本项目所有涉及网络请求的长耗时任务均内置断点续传，可安全应对程序崩溃、网络中断或手动中止（Ctrl+C）。

| 模块 | 检查点文件 | 恢复粒度 |
|------|-----------|---------|
| `fetch_wiki` | `wiki_progress.txt` | 每 20 页（含时间轴上限） |
| `fetch_cooc` | `cooc_progress.txt` | 每 100 个标签（会话内断点） |
| `fetch_cooc` | `cooc_history.json` | 永久历史豁免（跨会话，增量模式使用） |
| `fetch_artist_cooc` | `artist_cooc_progress.txt` | 每 N 个画师（会话内断点） |
| `fetch_artist_cooc` | `artist_cooc_history.json` | 永久历史豁免（跨会话） |
| `llm_processor` | `llm_temp.jsonl` | 每个 Batch 结束后 |
| `llm_processor` | `llm_history.json` | 永久历史豁免（跨会话） |

所有检查点文件在对应任务正常完成后会被**自动清理**。若希望强制从头重新运行某个步骤，手动删除对应检查点文件即可。

---

## 常见问题

**Q：为什么 `sync_tags` 没有更新某些标签的 `cn_name`？**

这是设计行为。`cn_name` 字段由 LLM 步骤精心生成和修正，`sync_tags` 刻意不覆盖此字段，以保护人工/模型成果。如需强制重置某标签的中文名，请直接编辑 `tags_enhanced.csv`，并从 `llm_history.json` 中删除对应条目。

**Q：LLM 步骤消耗了多少 API 调用？**

使用 `python main.py llm-process --preview` 可在不消耗任何 Token 的情况下预览本次待处理的标签数量，据此估算费用后再决定是否执行。

**Q：`pipeline` 命令会备份数据吗？**

会。`pipeline` 启动时自动将 `data/` 打包为 `data_backup_YYYYMMDD_HHMMSS.zip`。单步执行不会自动备份，如需手动备份请自行复制 `data/` 文件夹。

**Q：`trim_cooc` 的参数应该如何选取？**

先运行 `python main.py trim-cooc --dry-run`，查看不同 PMI 阈值下的数据保留量，选择在数据量和质量之间取得平衡的阈值，再正式执行。通常 `min_pmi=1.0` 搭配 `top_k=50` 是较为合理的起点。

**Q：`fetch-cooc` 的增量模式和全量模式应该如何选择？**

日常运行使用默认的**增量模式**即可。当标签库经历了大规模重构、或怀疑历史共现数据已严重过时时，才需要使用 `--full` 强制全量重建。注意全量模式会覆写整张共现矩阵，建议事先备份 `cooccurrence_matrix.csv`。

**Q：遭遇 Danbooru 429 限流怎么办？**

程序会自动检测 429 响应并休眠 3 分钟后重试，无需手动干预。若频繁触发，可考虑在对应模块中增大请求间隔。

**Q：如何只重新处理某些特定标签（跳过 LLM 历史豁免）？**

从 `data/checkpoint/llm_history.json` 中删除对应标签名，并清空该标签在 `tags_enhanced.csv` 中的 `wiki` 字段，再重新运行 `llm-process` 即可。

**Q：如何批量重置某个分类的所有标签 LLM 处理记录？**

使用独立工具 `tools/reset.py`，例如 `python tools/reset.py --category 4` 可一键清空所有角色标签的 `wiki` 字段并从 `llm_history.json` 中移除，使其在下次 `llm-process` 中被重新处理。详见[独立工具](#独立工具)一节。
