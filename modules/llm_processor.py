import json
import os
import re
import sys
import time
import random
import requests
import urllib.parse
from pathlib import Path

import click
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
_DEBUG: bool = False


def dbg(label: str, content=None, *, color: str = "magenta") -> None:
    if not _DEBUG:
        return
    bar = "─" * 60
    click.secho(f"\n{bar}", fg=color)
    click.secho(f"  ▶ {label}", fg=color, bold=True)
    if content is not None:
        if isinstance(content, (dict, list)):
            click.echo(json.dumps(content, ensure_ascii=False, indent=2))
        else:
            click.echo(str(content))
    click.secho(bar, fg=color)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["name", "cn_name", "wiki", "post_count", "category", "nsfw"])
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件（编码未知或文件损坏）：{path}")


def _load_state(config: dict, base_dir: Path) -> tuple:
    """
    加载主表、历史豁免集合、temp 中断记录，将 temp 记录合并入 df。

    返回 (df, history_names, temp_records, csv_path, history_path, temp_path)
    """
    csv_path     = base_dir / config["paths"]["processed"]["tags_enhanced"]
    history_path = base_dir / config["paths"]["checkpoint"]["llm_history"]
    temp_path    = base_dir / config["paths"]["checkpoint"]["llm_temp"]

    history_names: set = set()
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history_names = set(json.load(f))
        click.secho(f"[LLM Processor] 读取长期历史记录：豁免 {len(history_names)} 个已完工标签。", fg="blue")

    temp_records: dict = {}
    if temp_path.exists():
        with open(temp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    temp_records[record["name"]] = record
        click.secho(f"[LLM Processor] 检测到异常中断，恢复 {len(temp_records)} 条未合并的临时进度。", fg="yellow")

    df = read_csv_robust(csv_path)
    if temp_records:
        df.set_index("name", inplace=True)
        for name, item in temp_records.items():
            if name in df.index:
                _write_item_to_df(df, name, item)
        df.reset_index(inplace=True)

    return df, history_names, temp_records, csv_path, history_path, temp_path


def _save_state(df: pd.DataFrame, csv_path: Path, history_path: Path, temp_path: Path,
                history_names: set, current_run_processed: set, temp_records: dict) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8")
    click.secho(f"[LLM Processor] 主结果文件已覆写 (总计 {len(df)} 条)。", fg="green")

    history_names.update(temp_records.keys())
    history_names.update(current_run_processed)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(list(history_names), f, ensure_ascii=False)
    click.secho(f"[LLM Processor] 长期历史记录已更新 (共豁免 {len(history_names)} 词条)。", fg="green")

    if temp_path.exists():
        os.remove(temp_path)
        click.secho("[LLM Processor] 临时过程文件已清理。", fg="green")


# ---------------------------------------------------------------------------
# Wiki cache
# ---------------------------------------------------------------------------

def _parse_other_names(raw) -> list:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw or raw in ("[]", "nan", "None"):
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _load_wiki_cache(wiki_path: Path) -> tuple[dict, dict]:
    """返回 (real_wiki_map, other_names_map)"""
    real_wiki_map: dict = {}
    other_names_map: dict = {}

    if not wiki_path.exists():
        return real_wiki_map, other_names_map

    try:
        df_wiki = pd.read_parquet(wiki_path, columns=["title", "body", "other_names"])
    except Exception:
        try:
            df_wiki = pd.read_parquet(wiki_path, columns=["title", "body"])
            df_wiki["other_names"] = ""
            click.secho("[LLM Processor] 警告：Wiki DB 不含 other_names 列，别名提取将跳过。", fg="yellow")
        except Exception as e:
            click.secho(f"[LLM Processor] 警告：无法读取 Wiki DB: {e}", fg="yellow")
            return real_wiki_map, other_names_map

    for _, row in df_wiki.dropna(subset=["title"]).iterrows():
        title = row["title"]
        if isinstance(row["body"], str) and row["body"].strip():
            real_wiki_map[title] = row["body"]
        other_names_map[title] = _parse_other_names(row["other_names"])

    click.secho(
        f"[LLM Processor] Wiki DB 已加载：{len(real_wiki_map)} 条 body，"
        f"{sum(1 for v in other_names_map.values() if v)} 条含 other_names。",
        fg="blue",
    )
    return real_wiki_map, other_names_map


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_tag_name(tag_name: str) -> str:
    return re.sub(r"_\(.*\)$", "", str(tag_name)).replace("_", " ").strip().lower()


def clean_wiki_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"(?<!\[)\[/?[a-z]+\]", "", text)
    text = re.sub(
        r"h\d\.\s*(Examples?|See also|Colors?|External links?|Tags?|Related tags?|Aliases?)[^\n]*.*",
        "", text, flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"h\d\.\s*\S+[^\n]*\n?", "", text)
    text = re.sub(r"!\w+\s*#\d+[^\n]*", "", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]|]+)\|\]\]",
                  lambda m: m.group(1).split("(")[0].replace("_", " ").strip(), text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r'"[^"]+":https?://\S+', "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^\*[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", " ", text)
    text = text.replace("\n", " ").replace("\r", "")
    text = re.sub(r" {2,}", " ", text)
    result = text[:8000].strip()
    dbg("clean_wiki_text", f"清洗后 ({len(result)} 字符):\n{result}")
    return result


# ---------------------------------------------------------------------------
# Chinese name extraction
# ---------------------------------------------------------------------------

_HANZI_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df"
    r"\U0002a700-\U0002ceaf\uf900-\ufaff]"
)
_ALLOWED_NON_HANZI_RE = re.compile(
    r"[\s·・•\-—–,，.。、：:；;！!？?「」『』【】《》〈〉\u201c\u201d\u2018\u2019（）()\u3000]"
)


def _is_pure_chinese(text: str) -> bool:
    if len(_HANZI_RE.findall(text)) < 2:
        return False
    residual = _ALLOWED_NON_HANZI_RE.sub("", _HANZI_RE.sub("", text))
    return len(residual) == 0


def extract_chinese_from_other_names(other_names_raw) -> str:
    if not other_names_raw:
        return ""
    names: list = []
    if isinstance(other_names_raw, list):
        names = other_names_raw
    elif isinstance(other_names_raw, str):
        raw = other_names_raw.strip()
        if not raw or raw in ("[]", "nan", "None"):
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                names = parsed
        except Exception:
            pass
        if not names:
            names = [raw]

    dbg("extract_chinese_from_other_names 候选列表", names)
    for name in names:
        if isinstance(name, str) and _is_pure_chinese(name):
            dbg("extract_chinese_from_other_names 命中", name)
            return name.strip()
    return ""


# ---------------------------------------------------------------------------
# Work index & qualifier matching
# ---------------------------------------------------------------------------

def _build_work_cn_index(df: pd.DataFrame) -> tuple[dict, list]:
    """
    返回:
    - exact_index: name -> cn_name 第一字段
    - works_list:  [{"name": str, "cn": str}, ...] 按 post_count 降序
    """
    works = df[df["category"].isin(["3", 3]) & (df["cn_name"].str.strip() != "")].copy()
    works["post_count_int"] = pd.to_numeric(works["post_count"], errors="coerce").fillna(0)
    works["cn_first"] = works["cn_name"].str.split(",").str[0].str.strip()
    works = works.sort_values("post_count_int", ascending=False)

    exact_index: dict = {}
    works_list: list = []
    for _, row in works.iterrows():
        name = row["name"].strip()
        cn_first = row["cn_first"]
        if name not in exact_index:
            exact_index[name] = cn_first
        works_list.append({"name": name, "cn": cn_first})
    return exact_index, works_list


def _get_work_candidates(qualifier: str, exact_index: dict, works_list: list) -> tuple:
    """返回 (exact_cn, candidates)"""
    if qualifier in exact_index:
        return exact_index[qualifier], []
    q = qualifier.lower()
    candidates = [w for w in works_list if q in w["name"].lower() or w["name"].lower() in q]
    return None, candidates


_QUALIFIER_RE = re.compile(r"_\(([^)]+)\)$")


def _resolve_qualifier(tag_name: str, exact_index: dict, works_list: list) -> tuple:
    """返回 (exact_cn, candidates, qualifier_str)"""
    m = _QUALIFIER_RE.search(tag_name)
    if not m:
        return None, [], ""
    qualifier = m.group(1)
    exact_cn, candidates = _get_work_candidates(qualifier, exact_index, works_list)
    return exact_cn, candidates, qualifier


def _log_work_match(tag_name: str, exact_cn, candidates: list, qualifier: str, mode: str) -> None:
    if exact_cn:
        click.echo(f"  [WorkExact/{mode}] {tag_name} → {exact_cn}")
    elif candidates:
        click.echo(f"  [WorkCandidates/{mode}] {tag_name} qualifier={qualifier} "
                   f"candidates={[c['cn'] for c in candidates]}")


# ---------------------------------------------------------------------------
# Qualifier patch (post-LLM fallback)
# ---------------------------------------------------------------------------

def _patch_qualifier_cn_names(df: pd.DataFrame, target_names: set,
                               exact_index: dict, works_list: list) -> int:
    """
    对 LLM 未能成功补充作品名的 category=4 角色标签做精确匹配兜底。
    LLM 成功判定：cn_name 第一字段的全角括号内容能反查到 exact_index 的某个 cn 值。
    精确匹配不到则跳过，保持原样。返回修改条数。
    """
    all_work_cns = set(exact_index.values())
    modified = 0

    mask = df["category"].isin(["4", 4]) & df["name"].isin(target_names)
    for idx in df[mask].index:
        tag_name = df.at[idx, "name"]
        m = _QUALIFIER_RE.search(tag_name)
        if not m:
            continue

        work_cn = exact_index.get(m.group(1))
        if work_cn is None:
            continue

        cn_name_raw = df.at[idx, "cn_name"]
        parts = cn_name_raw.split(",")
        cn_first = parts[0].strip()
        if not cn_first:
            continue

        bracket_m = re.search(r"（([^）]+)）", cn_first)
        if bracket_m and bracket_m.group(1).strip() in all_work_cns:
            continue
        if "（" in cn_first:
            continue
        if any(hm.group(1).strip() == work_cn.strip()
               for hm in re.finditer(r"\(([^)]+)\)", cn_first)):
            continue

        parts[0] = f"{cn_first}（{work_cn}）"
        df.at[idx, "cn_name"] = ",".join(parts)
        modified += 1
        click.echo(f"  [Patch] {tag_name}: {cn_name_raw} -> {df.at[idx, 'cn_name']}")

    return modified


# ---------------------------------------------------------------------------
# Bangumi entity lookup
# ---------------------------------------------------------------------------

def validate_and_extract_cn_name(char_data: dict, clean_name: str, qualifier: str = "") -> str | None:
    aliases: set = set()
    if not isinstance(char_data, dict):
        return None

    raw_name = char_data.get("name")
    default_name = str(raw_name) if raw_name is not None else ""
    if default_name:
        aliases.add(default_name.lower())

    cn_name = default_name
    for info in (char_data.get("infobox") or []):
        if not isinstance(info, dict):
            continue
        key = str(info.get("key", ""))
        val = info.get("value")
        val_list = []
        if isinstance(val, str):
            val_list.append(val)
        elif isinstance(val, list):
            for v_item in val:
                if isinstance(v_item, dict) and "v" in v_item and v_item["v"] is not None:
                    val_list.append(str(v_item["v"]))
                elif isinstance(v_item, str):
                    val_list.append(v_item)
                elif v_item is not None:
                    val_list.append(str(v_item))
        elif val is not None:
            val_list.append(str(val))
        for v in val_list:
            if v:
                aliases.add(v.lower())
        if key in ("简体中文名", "中文名") and val_list:
            cn_name = val_list[0]

    clean_parts = set(clean_name.split())
    short_name = len(clean_parts) == 1 and len(clean_name) <= 4
    is_valid = False
    for alias in aliases:
        alias_lower = str(alias).lower()
        if clean_name == alias_lower:
            is_valid = True
            break
        if not short_name:
            alias_parts = set(alias_lower.replace(",", " ").split())
            if clean_parts and clean_parts.issubset(alias_parts):
                is_valid = True
                break
    if not is_valid:
        return None

    if qualifier:
        qualifier_lower = qualifier.lower().replace("_", " ")
        qualifier_words = set(qualifier_lower.split())
        found = False
        for alias in aliases:
            alias_norm = alias.lower().replace("_", " ").replace("-", " ")
            if qualifier_words.issubset(set(alias_norm.split())):
                found = True
                break
            if qualifier_lower.replace(" ", "") in alias_norm.replace(" ", "").replace("-", "").replace(":", ""):
                found = True
                break
        if not found:
            return None

    return cn_name


def fetch_entity_info(tag_name: str, category: int, token: str) -> dict:
    qualifier_match = re.search(r"_\(([^)]+)\)$", str(tag_name))
    qualifier = qualifier_match.group(1) if qualifier_match else ""
    clean_name = clean_tag_name(tag_name)
    headers = {"User-Agent": "YQH/DanbooruTagManager/1.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    result = {"cn_name": "", "summary": ""}

    for attempt in range(3):
        try:
            if category == 3:
                url = f"https://api.bgm.tv/search/subject/{urllib.parse.quote(clean_name)}?responseGroup=large"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("list"):
                        item = data["list"][0]
                        name_lower = str(item.get("name", "")).lower()
                        name_cn_lower = str(item.get("name_cn", "")).lower()
                        is_valid = clean_name in name_lower or clean_name in name_cn_lower
                        if not is_valid:
                            cp = set(clean_name.split())
                            np_ = set(name_lower.replace(":", " ").replace("-", " ").split())
                            if cp and cp.issubset(np_):
                                is_valid = True
                        if is_valid:
                            result["cn_name"] = item.get("name_cn") or item.get("name")
                            if item.get("summary"):
                                result["summary"] = item["summary"].replace("\r", "").replace("\n", "")[:200]
                break
            elif category == 4:
                url = "https://api.bgm.tv/v0/search/characters"
                payload = {"keyword": clean_name}
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200 and not resp.json().get("data"):
                    payload["keyword"] = clean_name.replace(" ", "")
                    resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    for char_data in (resp.json().get("data") or [])[:3]:
                        validated_cn = validate_and_extract_cn_name(char_data, clean_name, qualifier)
                        if validated_cn:
                            result["cn_name"] = validated_cn
                            if char_data.get("summary"):
                                result["summary"] = char_data["summary"].replace("\r", "").replace("\n", "")[:200]
                            break
                break
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2)
            else:
                click.secho(f"[LLM Processor] Bangumi API 网络失败 ({tag_name}): {e}", fg="yellow")
        except Exception as e:
            click.secho(f"[LLM Processor] Bangumi 数据解析异常 ({tag_name}): {e}", fg="yellow")
            break
    return result


# ---------------------------------------------------------------------------
# LLM prompts & calling layer
# ---------------------------------------------------------------------------

_TAG_GROUPS_RULE = """
        - `tag_groups`：该标签所属的 Danbooru 分类组列表（英文，可能为空）。
          若非空，**必须**将每个分类组名称翻译为准确的中文，并全部纳入扩展词。
          翻译规则：保留语义，不要直译生硬词——例如 hair_color→发色、hair_styles→发型、
          attire→服装、accessories→配饰、body_parts→身体部位、expressions→表情、
          actions→动作姿势、settings→场景背景、以此类推。
          这些分类词是搜索锚点，用户会通过它们检索到本标签，务必准确。
          若 `tag_groups` 为空，则根据标签语义自由生成上位概念或同义词。"""

_WORK_CANDIDATES_RULE = """
        - `work_candidates`：当标签名含有作品限定词且精确匹配失败时，由系统提供的候选作品列表，
          格式为 [{"name": "英文名", "cn": "中文名"}, ...]，可能为空列表。
          若非空，请从中选出最符合该角色所属作品的一项，并将其 `cn` 值**原文**追加到 `cn_name` 末尾，
          格式为「角色中文名（作品中文名）」。`cn` 值必须与列表中完全一致，不得改写。
          若列表为空或均不匹配，则不追加任何作品名。"""

_SYSTEM_PROMPT_GENERAL = f"""
# Role
你是一个 Danbooru 标签数据库的专家。

# Task
用户会提供一批标签数据，每条包含以下字段：
- `wiki_data`：官方英文描述，可能缺失
- `cn_name`：数据库中已有的中文名，可能为空或机翻错误
- `other_names`：Danbooru Wiki 记录的别名列表（含各语言），可辅助判断含义
- `cn_hint`：从 other_names 中自动提取的中文别名（若非空，可优先作为中文名参考）
{_TAG_GROUPS_RULE}
{_WORK_CANDIDATES_RULE}

请完成以下四个动作：

1. **生成中文描述 (chinese_wiki)**:
   - 将 `wiki_data` 里的核心信息完整翻译为中文。
   - 如果 `wiki_data` 为空，请根据知识库写一句该标签的中文视觉描述。
   - 如果知识库中没有相关信息且 `wiki_data` 也无效，返回空字符串。
   - 不要在输出里包含任何字数统计或备注信息，只输出描述本身。

2. **修正中文名 (cn_name)**:
   - 若 `cn_hint` 非空，优先将其作为基础中文名（它来自官方别名列表，可信度较高）。
   - 否则结合 `wiki_data` 和 `other_names` 的真实含义，检查 `cn_name` 是否准确；若存在机翻错误或词不达意，请修正为二次元语境下最准确的基础中文标签名。

3. **扩展中文名 (extended_cn_name)**:
   - 按照上方 `tag_groups` 处理规则生成分类锚点词，再补充 1~2 个同义词或近义词。
   - 只写扩展词，不要包含基础中文名，用半角逗号分隔。
   - 扩展中文名的总数为 2~4 个。

4. **NSFW 判定 (nsfw)**:
   - 包含裸露、性行为、生殖器、恋物癖(Fetish)、血腥暴力则为 1，否则为 0。

必须以合法 JSON 格式输出，结构如下，不要输出任何其他内容：
{{"items": [{{"name": "原始英文名", "cn_name": "修正后的准确基础中文名", "extended_cn_name": "扩展词（逗号分隔）", "chinese_wiki": "中文视觉描述", "nsfw": 0}}]}}
"""

_SYSTEM_PROMPT_FALLBACK = f"""
# Role
你是一个 Danbooru 标签数据库的资深专家。

# Task
用户会提供一批缺失 Wiki 描述的普通标签，每条包含以下字段：
- `cn_name`：数据库中已有的中文名，可能为空或机翻错误
- `other_names`：Danbooru Wiki 记录的别名列表（含各语言），可辅助判断含义
- `cn_hint`：从 other_names 中自动提取的中文别名（若非空，可优先作为中文名参考）
{_TAG_GROUPS_RULE}
{_WORK_CANDIDATES_RULE}

请根据标签英文名、`cn_name`、`other_names` 及你的内部知识库完成以下任务：

1. **生成中文描述 (chinese_wiki)**: 解释这个标签的视觉定义或含义，生成一句简练的中文描述（30字左右）。如果完全无法识别该标签，返回空字符串。
2. **修正中文名 (cn_name)**:
   - 若 `cn_hint` 非空，优先将其作为基础中文名。
   - 否则检查 `cn_name` 是否准确，结合 `other_names` 辅助判断，若存在机翻错误则修正为二次元语境下最准确的基础中文标签名。
3. **扩展中文名 (extended_cn_name)**: 按照上方 `tag_groups` 处理规则生成分类锚点词，再补充 1~2 个同义词或近义词，用半角逗号分隔。只写扩展词，不要包含基础中文名。扩展中文名的总数为 2~4 个。
4. **NSFW 判定 (nsfw)**: 包含裸露、性行为、生殖器、恋物癖(Fetish)、血腥暴力则为 1，否则为 0。

必须以合法 JSON 格式输出，结构如下，不要输出任何其他内容：
{{"items": [{{"name": "...", "cn_name": "...", "extended_cn_name": "...", "chinese_wiki": "...", "nsfw": 0}}]}}
"""

_SYSTEM_PROMPT_ENTITY = f"""
# Role
你是一个严谨的 ACG 领域防幻觉整理专家。

# Task
用户会提供角色名/作品名的标签数据，每条数据包含以下字段：
- `ref_cn`：外部数据源（Bangumi）给出的官方中文名，可能为空
- `ref_wiki`：外部数据源的简介，可能为空
- `other_names`：Danbooru Wiki 记录的别名列表（含各语言），可能包含中文名，可作为辅助参考
- `raw_cn_name`：数据库中已有的中文名（可能为空或机翻错误）
{_TAG_GROUPS_RULE}
{_WORK_CANDIDATES_RULE}

请完成以下四个动作：

1. **生成中文描述 (chinese_wiki)**:
   - 将 `ref_wiki` 完整翻译为中文简述。若 `ref_wiki` 为空，请根据你的知识库写约 50 字简介。
   - 不得在输出里包含字数统计或任何备注信息，只输出描述本身。

2. **确定中文名 (cn_name)**:
   - 优先级从高到低：`ref_cn`（外部权威数据）> `other_names` 中的中文名 > 你的知识库。
   - 如果 `ref_cn` 存在，直接采纳。
   - 如果 `ref_cn` 为空，检查 `other_names` 中是否有可信的中文名，若有则采纳。
   - 如果以上均为空，且你对该角色/作品的官方汉字名有把握，则填写；否则保留原英文名，绝不瞎猜音译。

3. **扩展中文名 (extended_cn_name)**:
   - 按照上方 `tag_groups` 处理规则生成分类锚点词。
   - 再补充该角色/作品的所属作品名、阵营或常见别名（1~2 个），用半角逗号分隔。
   - 只写扩展词，不要包含基础中文名。
   - 可参考 `other_names` 中出现的其他语言名称来辅助判断归属。

4. **NSFW 判定 (nsfw)**:
   - 包含裸露、性暗示、血腥暴力等则为 1，否则为 0。

必须以合法 JSON 格式输出，结构如下，不要输出任何其他内容：
{{"items": [{{"name": "原始英文名", "cn_name": "确定的基础中文名", "extended_cn_name": "扩展词（逗号分隔）", "chinese_wiki": "中文简介", "nsfw": 0}}]}}
"""

_PROMPTS: dict[str, tuple[str, float]] = {
    "general":  (_SYSTEM_PROMPT_GENERAL,  0.4),
    "fallback": (_SYSTEM_PROMPT_FALLBACK, 0.5),
    "entity":   (_SYSTEM_PROMPT_ENTITY,   0.1),
}

_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tag_batch_result",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":             {"type": "string"},
                            "cn_name":          {"type": "string"},
                            "extended_cn_name": {"type": "string"},
                            "chinese_wiki":     {"type": "string"},
                            "nsfw":             {"type": "integer"},
                        },
                        "required": ["name", "cn_name", "extended_cn_name", "chinese_wiki", "nsfw"],
                    },
                }
            },
            "required": ["items"],
        },
        "strict": False,
    },
}


def _llm_request(client, model_name: str, system_prompt: str,
                 batch_data: list, temperature: float) -> list:
    extra_body = {"reasoning": {"enabled": False}}
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(batch_data, ensure_ascii=False)},
                ],
                temperature=temperature,
                response_format=_JSON_SCHEMA,
                extra_body=extra_body,
            )
            raw_content = response.choices[0].message.content
            if _DEBUG:
                dbg("LLM 响应", raw_content, color="green")
            return json.loads(raw_content).get("items", [])
        except Exception as e:
            if attempt == max_attempts - 1:
                click.secho(f"[LLM Processor] LLM 请求失败，已重试 {max_attempts} 次，放弃本批次: {e}", fg="red")
                break
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            click.secho(
                f"[LLM Processor] LLM 请求出错 (尝试 {attempt + 1}/{max_attempts})，{wait:.1f}s 后重试: {e}",
                fg="yellow",
            )
            time.sleep(wait)
    return []


def _call_llm(client, model_name: str, mode: str, batch_data: list) -> list:
    system_prompt, temperature = _PROMPTS[mode]
    if _DEBUG:
        dbg(f"LLM 请求 [mode={mode}] — system prompt", system_prompt.strip(), color="cyan")
        dbg(f"LLM 请求 [mode={mode}] — user payload ({len(batch_data)} 条)", batch_data, color="cyan")
    return _llm_request(client, model_name, system_prompt, batch_data, temperature)


# ---------------------------------------------------------------------------
# Result application
# ---------------------------------------------------------------------------

def _combine_cn(base_cn: str, ext_cn: str) -> str:
    return re.sub(r",+", ",", ",".join(filter(None, [base_cn, ext_cn])).strip(","))


def _write_item_to_df(df: pd.DataFrame, key, item: dict) -> None:
    base_cn  = str(item.get("cn_name", "")).strip()
    ext_cn   = str(item.get("extended_cn_name", "")).strip()
    combined = _combine_cn(base_cn, ext_cn)
    if combined:
        df.at[key, "cn_name"] = combined
    if item.get("chinese_wiki"):
        df.at[key, "wiki"] = item["chinese_wiki"]
    if "nsfw" in item:
        df.at[key, "nsfw"] = str(item.get("nsfw", 0))


def _apply_results(df: pd.DataFrame, results: list, temp_path: Path,
                   current_run_processed: set) -> None:
    if not results:
        return
    with open(temp_path, "a", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            current_run_processed.add(item["name"])
    for item in results:
        matches = df[df["name"] == item["name"]].index
        if not matches.empty:
            _write_item_to_df(df, matches[0], item)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _build_entity_payload(row: pd.Series, other_names_map: dict, real_wiki_map: dict,
                           tag_to_groups_map: dict, exact_index: dict, works_list: list,
                           bangumi_token: str) -> dict:
    tag_name = row["name"]
    category = int(row["category"]) if str(row["category"]).lstrip("-").isdigit() else -1
    other_names_list = other_names_map.get(tag_name, [])

    ref_wiki = clean_wiki_text(real_wiki_map.get(tag_name, ""))
    dbg(f"[{tag_name}] other_names", other_names_list)
    dbg(f"[{tag_name}] ref_wiki (前200字)", ref_wiki[:200] if ref_wiki else "(空)")

    ref_cn = extract_chinese_from_other_names(other_names_list)
    if ref_cn:
        click.secho(f"  [Step1/Wiki] {tag_name} → {ref_cn}", fg="blue")

    bangumi_summary = ""
    if not ref_cn:
        ext_info = fetch_entity_info(tag_name, category, bangumi_token)
        if ext_info["cn_name"]:
            ref_cn = ext_info["cn_name"]
            bangumi_summary = ext_info["summary"]
            click.secho(f"  [Step2/Bangumi] {tag_name} → {ref_cn}", fg="blue")

    if not ref_wiki and bangumi_summary:
        ref_wiki = bangumi_summary

    if not ref_cn:
        click.secho(f"  [Step3/LLM] {tag_name} → 交由 LLM 知识库决定", fg="yellow")

    exact_cn, candidates, qualifier = _resolve_qualifier(tag_name, exact_index, works_list)
    _log_work_match(tag_name, exact_cn, candidates, qualifier, "entity")

    tag_groups = [g.replace("tag_group:", "") for g in tag_to_groups_map.get(tag_name, [])]
    if tag_groups:
        click.echo(f"  [TagGroups/entity] {tag_name} → {', '.join(tag_groups)}")

    return {
        "name":            tag_name,
        "raw_cn_name":     str(row["cn_name"]),
        "ref_cn":          ref_cn,
        "ref_wiki":        ref_wiki,
        "other_names":     [n for n in other_names_list if isinstance(n, str) and n.strip()],
        "tag_groups":      tag_groups,
        "work_candidates": [] if exact_cn else candidates,
    }


def _build_general_payload(row: pd.Series, other_names_map: dict, real_wiki_map: dict,
                            tag_to_groups_map: dict, exact_index: dict, works_list: list) -> dict:
    tag_name = row["name"]
    other_names_list = other_names_map.get(tag_name, [])
    clean_wiki = clean_wiki_text(real_wiki_map.get(tag_name, ""))
    dbg(f"[{tag_name}] other_names", other_names_list)
    dbg(f"[{tag_name}] wiki (前200字)", clean_wiki[:200] if clean_wiki else "(空)")

    clean_other_names = [n for n in other_names_list if isinstance(n, str) and n.strip()]
    cn_hint = extract_chinese_from_other_names(other_names_list)

    mode_label = "general" if clean_wiki else "fallback"
    cn_hint_label = f" cn_hint={cn_hint}" if cn_hint else ""
    click.echo(f"  [{mode_label}] {tag_name}{cn_hint_label}")

    exact_cn, candidates, qualifier = _resolve_qualifier(tag_name, exact_index, works_list)
    _log_work_match(tag_name, exact_cn, candidates, qualifier, mode_label)

    tag_groups = [g.replace("tag_group:", "") for g in tag_to_groups_map.get(tag_name, [])]
    if tag_groups:
        click.echo(f"  [TagGroups/{mode_label}] {tag_name} → {', '.join(tag_groups)}")

    payload = {
        "name":            tag_name,
        "cn_name":         str(row["cn_name"]),
        "other_names":     clean_other_names,
        "cn_hint":         cn_hint,
        "tag_groups":      tag_groups,
        "work_candidates": [] if exact_cn else candidates,
    }
    if clean_wiki:
        payload["wiki_data"] = clean_wiki
    return payload


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: dict, preview: bool = False, debug: bool = False) -> None:
    global _DEBUG
    _DEBUG = debug

    base_dir      = Path(__file__).resolve().parent.parent
    API_KEY       = os.getenv("OPENAI_API_KEY")
    BASE_URL      = os.getenv("OPENAI_BASE_URL")
    BANGUMI_TOKEN = os.getenv("BANGUMI_ACCESS_TOKEN", "")

    if not preview and not API_KEY:
        click.secho("[LLM Processor] 错误：未配置 OPENAI_API_KEY", fg="red")
        sys.exit(1)

    client     = OpenAI(base_url=BASE_URL, api_key=API_KEY) if not preview else None
    model_name = config["settings"]["llm"]["model_name"]
    batch_size = config["settings"]["llm"]["batch_size"]

    df, history_names, temp_records, csv_path, history_path, temp_path = _load_state(config, base_dir)

    wiki_path = base_dir / config["paths"]["processed"]["wiki_parquet"]
    real_wiki_map, other_names_map = _load_wiki_cache(wiki_path)

    tag_to_groups_map: dict = {}
    tag_groups_path = base_dir / config["paths"]["processed"].get("tag_groups", "")
    if tag_groups_path and tag_groups_path.exists():
        try:
            with open(tag_groups_path, "r", encoding="utf-8") as f:
                tag_to_groups_map = json.load(f).get("tag_to_groups", {})
            click.secho(
                f"[LLM Processor] Tag Groups 已加载：覆盖 {len(tag_to_groups_map)} 个标签的分类归属。", fg="blue")
        except Exception as e:
            click.secho(f"[LLM Processor] 警告：无法读取 tag_groups.json: {e}", fg="yellow")

    work_exact_index, work_list = _build_work_cn_index(df)

    indices_entity: list = []
    indices_general: list = []
    for idx, row in df.iterrows():
        name = row["name"]
        if name in history_names or name in temp_records:
            continue
        already_done = len(str(row["wiki"]).strip()) >= 2
        if row["category"] in ("3", "4", 3, 4):
            if not already_done:
                indices_entity.append(idx)
        elif not already_done:
            indices_general.append(idx)

    if preview:
        click.secho(
            f"\n[LLM Processor] 待处理统计 [实体处理: {len(indices_entity)} 条] | [普通标签: {len(indices_general)} 条]",
            fg="magenta",
        )
        return

    current_run_processed: set = set()

    # Entity pass
    if indices_entity:
        click.secho(f"\n开始执行实体处理 (共 {len(indices_entity)} 条)...", fg="cyan")
        for i in range(0, len(indices_entity), batch_size):
            batch_idx = indices_entity[i: i + batch_size]
            payload = [
                _build_entity_payload(
                    df.iloc[idx], other_names_map, real_wiki_map,
                    tag_to_groups_map, work_exact_index, work_list, BANGUMI_TOKEN,
                )
                for idx in batch_idx
            ]
            click.echo(
                f"[LLM Processor] 实体处理进度: "
                f"{min(i + batch_size, len(indices_entity))}/{len(indices_entity)} ..."
            )
            results = _call_llm(client, model_name, "entity", payload)
            _apply_results(df, results, temp_path, current_run_processed)
            time.sleep(1)

    # General / fallback pass
    if indices_general:
        click.secho(f"\n开始执行常规翻译与无Wiki兜底 (共 {len(indices_general)} 条)...", fg="cyan")
        batch_translate: list = []
        batch_fallback: list  = []
        processed_count = 0

        def flush(payload_list: list, mode: str) -> None:
            if not payload_list:
                return
            clean_payload = [{k: v for k, v in item.items() if k != "_idx"} for item in payload_list]
            mode_name = "常规翻译" if mode == "general" else "专属兜底生成"
            click.echo(f"[LLM Processor] [{mode_name}] 进度: {processed_count}/{len(indices_general)} ...")
            results = _call_llm(client, model_name, mode, clean_payload)
            _apply_results(df, results, temp_path, current_run_processed)
            time.sleep(1)
            payload_list.clear()

        for idx in indices_general:
            row = df.iloc[idx]
            entry = _build_general_payload(
                row, other_names_map, real_wiki_map,
                tag_to_groups_map, work_exact_index, work_list,
            )
            entry["_idx"] = idx
            if "wiki_data" in entry:
                batch_translate.append(entry)
            else:
                batch_fallback.append(entry)
            processed_count += 1
            if len(batch_translate) >= batch_size:
                flush(batch_translate, "general")
            if len(batch_fallback) >= batch_size:
                flush(batch_fallback, "fallback")

        flush(batch_translate, "general")
        flush(batch_fallback, "fallback")

    if current_run_processed or temp_records:
        click.secho("\n[LLM Processor] API 环节结束，开始安全合盘...", fg="cyan")
        patch_count = _patch_qualifier_cn_names(
            df, current_run_processed | set(temp_records.keys()),
            work_exact_index, work_list,
        )
        if patch_count:
            click.secho(f"[LLM Processor] 作品限定词补全：修改 {patch_count} 条角色标签。", fg="blue")
        _save_state(df, csv_path, history_path, temp_path,
                    history_names, current_run_processed, temp_records)
    else:
        click.secho("\n  [LLM Processor] 本次运行没有需要更新的数据。", fg="green")