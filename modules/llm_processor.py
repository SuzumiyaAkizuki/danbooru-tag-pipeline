import pandas as pd
import re
import os
import sys
import time
import json
import requests
import urllib.parse
from pathlib import Path
from openai import OpenAI
import click


# ---------------------------------------------------------------------------
# Debug 工具：通过 run(debug=True) 或 CLI --debug 开启
# ---------------------------------------------------------------------------
_DEBUG: bool = False

def dbg(label: str, content=None, *, color: str = "magenta") -> None:
    """仅在 debug 模式下输出详细信息。"""
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


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=['name', 'cn_name', 'wiki', 'post_count', 'category', 'nsfw'])
    for enc in ['utf-8', 'gbk', 'gb18030']:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件（编码未知或文件损坏）：{path}")


def clean_tag_name(tag_name):
    return re.sub(r'_\(.*\)$', '', str(tag_name)).replace('_', ' ').strip().lower()


def clean_wiki_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    # 1. 去除 BBCode 标签（如 [i]、[/i]、[b] 等），保留内容
    #    用负向回顾排除 [[ 开头的情况，避免误吞 wikilink 内容
    text = re.sub(r'(?<!\[)\[/?[a-z]+\]', '', text)

    # 2. 在纯噪声章节处截断正文（这些章节后的内容对描述无价值）
    text = re.sub(
        r'h\d\.\s*(Examples?|See also|Colors?|External links?|Tags?|Related tags?|Aliases?)[^\n]*.*',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )

    # 3. 去除剩余章节标题行（如 h4. Appearance）
    text = re.sub(r'h\d\.\s*\S+[^\n]*\n?', '', text)

    # 4. 去除图片/资源引用（!post #123、!asset #456）
    text = re.sub(r'!\w+\s*#\d+[^\n]*', '', text)

    # 5. 处理 [[链接]] 语法，分三种情况：
    #    [[target|display]]  → display
    #    [[target|]]         → target（去括号限定词和下划线）
    #    [[target]]          → target
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]|]+)\|\]\]',
                  lambda m: m.group(1).split('(')[0].replace('_', ' ').strip(), text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # 6. 去除 Textile 外部链接（"文字":https://...）
    text = re.sub(r'"[^"]+":https?://\S+', '', text)

    # 7. 去除加粗/斜体 Wiki 标记
    text = text.replace("'''", "").replace("''", "")

    # 8. 去除以 * 开头的列表行（通常是标签枚举，无描述价值）
    text = re.sub(r'^\*[^\n]*$', '', text, flags=re.MULTILINE)

    # 9. 折叠空白并截断
    text = re.sub(r'\n{2,}', ' ', text)
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r' {2,}', ' ', text)

    result = text[:400].strip()
    dbg("clean_wiki_text 清洗结果",
        f"原始长度: {len(text)} 字符\n清洗后 ({len(result)} 字符):\n{result}")
    return result


# ---------------------------------------------------------------------------
# 新增：从 other_names 列表中提取第一个中文名（简体或繁体均可）
# ---------------------------------------------------------------------------
_HANZI_RE = re.compile(
    r'[\u4e00-\u9fff'          # CJK 基本汉字
    r'\u3400-\u4dbf'           # CJK 扩展 A
    r'\U00020000-\U0002a6df'   # CJK 扩展 B
    r'\U0002a700-\U0002ceaf'   # CJK 扩展 C/D/E/F
    r'\uf900-\ufaff'           # CJK 兼容汉字
    r']'
)

# 允许出现在"纯汉字"字符串中的非汉字字符：标点、空格、书名号等
_ALLOWED_NON_HANZI_RE = re.compile(
    r'[\s·・•\-—–,，.。、：:；;！!？?「」『』【】《》〈〉""''（）()\u3000]'
)

def _is_pure_chinese(text: str) -> bool:
    """
    判断字符串是否为「纯汉字」名称：
    - 至少包含 2 个汉字
    - 去掉汉字和允许的标点/空格后，不剩任何其他字符
      （即不含假名、拉丁字母、韩文、数字等）
    这样可以正确接受「东方」「东方Project」中的「东方」，
    同时拒绝「東方プロジェクト」「东方Project」「동방」等混合串。
    """
    hanzi_chars = _HANZI_RE.findall(text)
    if len(hanzi_chars) < 2:
        return False
    # 移除所有汉字和允许字符后，若还有剩余，说明含有杂质
    residual = _HANZI_RE.sub('', text)
    residual = _ALLOWED_NON_HANZI_RE.sub('', residual)
    return len(residual) == 0


def extract_chinese_from_other_names(other_names_raw) -> str:
    """
    从 wiki_pages.parquet 的 other_names 字段中提取第一个中文别名。

    other_names 在入库时被序列化为字符串（str(list)），此处需先反序列化。
    返回第一个含有中文的别名；若无则返回空字符串。
    """
    if not other_names_raw:
        return ""

    # other_names 存储为 Python list 的字符串表示，尝试用 json / ast 解析
    names = []
    if isinstance(other_names_raw, list):
        names = other_names_raw
    elif isinstance(other_names_raw, str):
        raw = other_names_raw.strip()
        if not raw or raw in ('[]', 'nan', 'None'):
            return ""
        # 先尝试 JSON（双引号列表）
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                names = parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # 最后兜底：把字符串本身当作单个候选
        if not names:
            names = [raw]

    dbg("extract_chinese_from_other_names 候选列表", names)
    for name in names:
        if isinstance(name, str):
            ok = _is_pure_chinese(name)
            if _DEBUG:
                click.secho(f"    {'✓' if ok else '✗'}  {name!r}", fg="magenta")
            if ok:
                dbg("extract_chinese_from_other_names 命中", name)
                return name.strip()

    dbg("extract_chinese_from_other_names 未命中任何纯中文别名")
    return ""




def validate_and_extract_cn_name(char_data, clean_name, qualifier: str = "") -> str | None:
    """
    从 Bangumi 角色数据中提取并校验中文名。

    qualifier：从 tag_name 括号中提取的作品限定词（如 "arknights"）。
    若存在 qualifier，要求该角色的 aliases 中必须也包含此词，才视为匹配，
    以避免同名角色跨作品误匹配（如用 "sora" 匹配到非明日方舟的角色）。
    """
    aliases = set()
    if not isinstance(char_data, dict):
        return None

    raw_name = char_data.get('name')
    default_name = str(raw_name) if raw_name is not None else ""
    if default_name:
        aliases.add(default_name.lower())

    cn_name = default_name
    infobox = char_data.get('infobox')
    if not isinstance(infobox, list):
        infobox = []

    for info in infobox:
        if not isinstance(info, dict):
            continue

        key = str(info.get('key', ''))
        val = info.get('value')

        val_list = []
        if isinstance(val, str):
            val_list.append(val)
        elif isinstance(val, list):
            for v_item in val:
                if isinstance(v_item, dict) and 'v' in v_item:
                    v_val = v_item['v']
                    if v_val is not None:
                        val_list.append(str(v_val))
                elif isinstance(v_item, str):
                    val_list.append(v_item)
                elif v_item is not None:
                    val_list.append(str(v_item))
        elif val is not None:
            val_list.append(str(val))

        for v in val_list:
            if v:
                aliases.add(str(v).lower())

        if key in ['简体中文名', '中文名'] and val_list:
            cn_name = val_list[0]

    # ── 步骤1：校验角色名是否与查询词匹配 ──
    is_valid = False
    clean_parts = set(clean_name.split())
    # 极短名称（单词且 ≤4 字符）必须精确匹配，不允许子集匹配，避免误命中
    short_name = len(clean_parts) == 1 and len(clean_name) <= 4

    for alias in aliases:
        if not alias:
            continue
        alias_lower = str(alias).lower()
        if clean_name == alias_lower:
            is_valid = True
            break
        if not short_name:
            alias_parts = set(alias_lower.replace(',', ' ').split())
            if clean_parts and clean_parts.issubset(alias_parts):
                is_valid = True
                break

    if not is_valid:
        return None

    # ── 步骤2：若有作品限定词，校验该角色确实属于此作品 ──
    if qualifier:
        qualifier_lower = qualifier.lower().replace('_', ' ')
        qualifier_words = set(qualifier_lower.split())
        found_qualifier = False
        for alias in aliases:
            alias_norm = alias.lower().replace('_', ' ').replace('-', ' ')
            # 限定词的所有词均出现在某个 alias 中，视为归属匹配
            if qualifier_words.issubset(set(alias_norm.split())):
                found_qualifier = True
                break
            # 也接受 alias 包含限定词的连续子串（处理如 "re:zero" 这类带符号的名称）
            if qualifier_lower.replace(' ', '') in alias_norm.replace(' ', '').replace('-', '').replace(':', ''):
                found_qualifier = True
                break
        if not found_qualifier:
            return None

    return cn_name


def fetch_entity_info(tag_name, category, token):
    # 提取括号内的作品限定词，如 "sora_(arknights)" → qualifier="arknights"
    qualifier_match = re.search(r'_\(([^)]+)\)$', str(tag_name))
    qualifier = qualifier_match.group(1) if qualifier_match else ""
    clean_name = clean_tag_name(tag_name)  # 已去除括号部分
    headers = {
        "User-Agent": "YQH/DanbooruTagManager/1.0",
        "Accept": "application/json"
    }
    if token: headers["Authorization"] = f"Bearer {token}"

    result = {"cn_name": "", "summary": ""}

    for attempt in range(3):
        try:
            if category == 3:
                url = f"https://api.bgm.tv/search/subject/{urllib.parse.quote(clean_name)}?responseGroup=large"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('list'):
                        item = data['list'][0]
                        name_lower = str(item.get('name', '')).lower()
                        name_cn_lower = str(item.get('name_cn', '')).lower()

                        is_valid = False
                        if clean_name in name_lower or clean_name in name_cn_lower:
                            is_valid = True
                        else:
                            clean_parts = set(clean_name.split())
                            name_parts = set(name_lower.replace(':', ' ').replace('-', ' ').split())
                            if clean_parts and clean_parts.issubset(name_parts):
                                is_valid = True

                        if is_valid:
                            result["cn_name"] = item.get('name_cn') or item.get('name')
                            if item.get('summary'):
                                result["summary"] = item.get('summary', '').replace('\r', '').replace('\n', '')[:200]
                break

            elif category == 4:
                url = "https://api.bgm.tv/v0/search/characters"
                payload = {"keyword": clean_name}
                resp = requests.post(url, json=payload, headers=headers, timeout=10)

                if resp.status_code == 200 and not resp.json().get('data'):
                    no_space_name = clean_name.replace(" ", "")
                    payload["keyword"] = no_space_name
                    resp = requests.post(url, json=payload, headers=headers, timeout=10)

                if resp.status_code == 200:
                    data = resp.json()
                    char_list = data.get('data', [])
                    for char_data in char_list[:3]:
                        validated_cn = validate_and_extract_cn_name(char_data, clean_name, qualifier)
                        if validated_cn:
                            result["cn_name"] = validated_cn
                            if char_data.get('summary'):
                                result["summary"] = char_data.get('summary', '').replace('\r', '').replace('\n', '')[
                                    :200]
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


def process_batch_smart(client, model_name, batch_data, mode="general", debug: bool = False):
    if mode == "general":
        system_prompt = """
        # Role
        你是一个 Danbooru 标签数据库的专家。

        # Task
        用户会提供一批标签数据，每条包含以下字段：
        - `wiki_data`：官方英文描述，可能缺失
        - `cn_name`：数据库中已有的中文名，可能为空或机翻错误
        - `other_names`：Danbooru Wiki 记录的别名列表（含各语言），可辅助判断含义
        - `cn_hint`：从 other_names 中自动提取的中文别名（若非空，可优先作为中文名参考）

        请完成以下四个动作：

        1. **生成中文描述 (chinese_wiki)**:
           - 提取 `wiki_data` 里的核心信息进行翻译或总结，生成一句简练的中文视觉描述（50字左右）。
           - 如果 `wiki_data` 为空，请根据知识库写一句该标签的中文视觉描述。
           - 如果知识库中没有相关信息且 `wiki_data` 也无效，返回空字符串。
           - 不要在输出里包含任何字数统计或备注信息，只输出描述本身。

        2. **修正中文名 (cn_name)**:
           - 若 `cn_hint` 非空，优先将其作为基础中文名（它来自官方别名列表，可信度较高）。
           - 否则结合 `wiki_data` 和 `other_names` 的真实含义，检查 `cn_name` 是否准确；若存在机翻错误或词不达意，请修正为二次元语境下最准确的基础中文标签名。

        3. **扩展中文名 (extended_cn_name)**:
           - 根据修正后的基础标签含义，生成 3~4 个相关的同义词、风格或大类（用半角逗号分隔）。
           - 只写扩展词，不要包含基础中文名。

        4. **NSFW 判定 (nsfw)**:
           - 包含裸露、性行为、生殖器、恋物癖(Fetish)、血腥暴力则为 1，否则为 0。

        # Output Format (JSON)
        {
            "items": [
                {
                    "name": "原始英文名",
                    "cn_name": "修正后的准确基础中文名",
                    "extended_cn_name": "关联的上位概念或扩展词（逗号分隔）",
                    "chinese_wiki": "生成的中文视觉描述",
                    "nsfw": 0 或 1
                }
            ]
        }
        """
        temp = 0.4
    elif mode == "fallback":
        system_prompt = """
        # Role
        你是一个 Danbooru 标签数据库的资深专家。

        # Task
        用户会提供一批缺失 Wiki 描述的普通标签，每条包含以下字段：
        - `cn_name`：数据库中已有的中文名，可能为空或机翻错误
        - `other_names`：Danbooru Wiki 记录的别名列表（含各语言），可辅助判断含义
        - `cn_hint`：从 other_names 中自动提取的中文别名（若非空，可优先作为中文名参考）

        请根据标签英文名、`cn_name`、`other_names` 及你的内部知识库完成以下任务：

        1. **生成中文描述 (chinese_wiki)**: 解释这个标签的视觉定义或含义，生成一句简练的中文描述（30字左右）。如果完全无法识别该标签，返回空字符串。
        2. **修正中文名 (cn_name)**:
           - 若 `cn_hint` 非空，优先将其作为基础中文名。
           - 否则检查 `cn_name` 是否准确，结合 `other_names` 辅助判断，若存在机翻错误则修正为二次元语境下最准确的基础中文标签名。
        3. **扩展中文名 (extended_cn_name)**: 生成 3~4 个相关的同义词、风格或大类（用半角逗号分隔）。只写扩展词，不要包含基础中文名。
        4. **NSFW 判定 (nsfw)**: 包含裸露、性行为、生殖器、恋物癖(Fetish)、血腥暴力则为 1，否则为 0。

        # Output Format (JSON)
        {"items": [{"name": "...", "cn_name": "...", "extended_cn_name": "...", "chinese_wiki": "...", "nsfw": 0}]}
        """
        temp = 0.5
    else:  # entity 模式
        system_prompt = """
        # Role
        你是一个严谨的 ACG 领域防幻觉整理专家。

        # Task
        用户会提供角色名/作品名的标签数据，每条数据包含以下字段：
        - `ref_cn`：外部数据源（Bangumi）给出的官方中文名，可能为空
        - `ref_wiki`：外部数据源的简介，可能为空
        - `other_names`：Danbooru Wiki 记录的别名列表（含各语言），可能包含中文名，可作为辅助参考
        - `raw_cn_name`：数据库中已有的中文名（可能为空或机翻错误）

        请完成以下四个动作：

        1. **生成中文描述 (chinese_wiki)**:
           - 优先提炼 `ref_wiki` 为 50 字左右的中文简述。若 `ref_wiki` 为空，请根据你的知识库写 50 字简介。
           - 不得在输出的中文描述里包含类似于"(50字)"的字数统计，或任何备注信息，只允许输出描述本身。

        2. **确定中文名 (cn_name)**:
           - 优先级从高到低：`ref_cn`（外部权威数据）> `other_names` 中的中文名 > 你的知识库。
           - 如果 `ref_cn` 存在，直接采纳。
           - 如果 `ref_cn` 为空，检查 `other_names` 中是否有可信的中文名，若有则采纳。
           - 如果以上均为空，且你对该角色/作品的官方汉字名有把握，则填写；否则保留原英文名，绝不瞎猜音译。

        3. **扩展中文名 (extended_cn_name)**:
           - 为该角色或作品生成 2~3 个相关的作品前缀、所属阵营或别名（用半角逗号分隔）。
           - 只写扩展词，不要包含基础中文名。
           - 可参考 `other_names` 中出现的其他语言名称来辅助判断该角色/作品的归属。

        4. **NSFW 判定 (nsfw)**:
           - 包含裸露、性暗示、血腥暴力等则为 1，否则为 0。

        # Output Format (JSON)
        {
            "items": [
                {
                    "name": "原始英文名",
                    "cn_name": "确定的基础中文名",
                    "extended_cn_name": "关联的作品前缀或扩展别名（逗号分隔）",
                    "chinese_wiki": "生成的中文简介",
                    "nsfw": 0 或 1
                }
            ]
        }
        """
        temp = 0.1

    if debug:
        dbg(f"LLM 请求 [mode={mode}] — system prompt",
            system_prompt.strip(), color="cyan")
        dbg(f"LLM 请求 [mode={mode}] — user payload ({len(batch_data)} 条)",
            batch_data, color="cyan")

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(batch_data, ensure_ascii=False)}
                ],
                temperature=temp,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            if debug:
                dbg(f"LLM 响应 [mode={mode}]", raw_content, color="green")
            return json.loads(raw_content).get("items", [])
        except Exception as e:
            if attempt == max_attempts - 1:
                click.secho(f"[LLM Processor] LLM 请求失败，已重试 {max_attempts} 次，放弃本批次: {e}", fg="red")
                break
            # 指数退避：2^attempt 秒，加随机抖动，上限 60 秒
            import random as _random
            wait = min(2 ** attempt + _random.uniform(0, 1), 60)
            click.secho(
                f"[LLM Processor] LLM 请求出错 (尝试 {attempt + 1}/{max_attempts})，"
                f"{wait:.1f}s 后重试: {e}",
                fg="yellow"
            )
            time.sleep(wait)
    return []


def run(config, preview=False, debug=False):
    global _DEBUG
    _DEBUG = debug
    base_dir = Path(__file__).resolve().parent.parent

    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    BANGUMI_TOKEN = os.getenv("BANGUMI_ACCESS_TOKEN")

    if not preview and not API_KEY:
        click.secho("[LLM Processor] 错误：未配置 OPENAI_API_KEY", fg="red")
        sys.exit(1)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY) if not preview else None
    model_name = config['settings']['llm']['model_name']
    batch_size = config['settings']['llm']['batch_size']

    csv_path = base_dir / config['paths']['processed']['tags_enhanced']
    wiki_path = base_dir / config['paths']['processed']['wiki_parquet']
    history_path = base_dir / config['paths']['checkpoint']['llm_history']
    temp_path = base_dir / config['paths']['checkpoint']['llm_temp']

    history_names = set()
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            history_names = set(json.load(f))
        click.secho(f"[LLM Processor] 读取长期历史记录：豁免 {len(history_names)} 个已完工标签。", fg="blue")

    temp_records = {}
    if temp_path.exists():
        with open(temp_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    temp_records[record['name']] = record
        click.secho(f"[LLM Processor] 检测到异常中断，恢复 {len(temp_records)} 条未合并的临时进度。", fg="yellow")

    df_all = read_csv_robust(csv_path)
    if temp_records:
        df_all.set_index('name', inplace=True)
        for name, item in temp_records.items():
            if name in df_all.index:
                base_cn = str(item.get('cn_name', '')).strip()
                ext_cn = str(item.get('extended_cn_name', '')).strip()
                combined_cn = re.sub(r',+', ',', ",".join(filter(None, [base_cn, ext_cn])).strip(','))
                if combined_cn: df_all.at[name, 'cn_name'] = combined_cn
                if item.get('chinese_wiki'): df_all.at[name, 'wiki'] = item['chinese_wiki']
                if 'nsfw' in item: df_all.at[name, 'nsfw'] = str(item.get('nsfw', 0))
        df_all.reset_index(inplace=True)

    # ------------------------------------------------------------------
    # 构建 Wiki 本地缓存：body 文本 + other_names 列表
    # fetch_wiki.py 存储时执行了 str(list)，即 Python repr 格式（单引号列表）
    # fetch_wiki.py 现在统一用 json.dumps() 写入，直接 json.loads() 解析即可
    # ------------------------------------------------------------------
    import ast

    def _parse_other_names(raw) -> list:
        """
        将 fetch_wiki.py 写入的 JSON 字符串还原为 Python list。
        fetch_wiki.py 现在统一用 json.dumps() 序列化 other_names，
        所以只需要处理标准 JSON 格式即可。
        """
        if isinstance(raw, list):
            return raw
        if not isinstance(raw, str):
            return []
        raw = raw.strip()
        if not raw or raw in ('[]', 'nan', 'None'):
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return []

    real_wiki_map = {}   # title -> body str
    other_names_map = {} # title -> List[str]（已反序列化）

    if wiki_path.exists():
        try:
            df_wiki = pd.read_parquet(wiki_path, columns=['title', 'body', 'other_names'])
            has_other_names = True
        except Exception:
            try:
                df_wiki = pd.read_parquet(wiki_path, columns=['title', 'body'])
                df_wiki['other_names'] = ''
                has_other_names = False
                click.secho("[LLM Processor] 警告：Wiki DB 不含 other_names 列，别名提取将跳过。", fg="yellow")
            except Exception as e2:
                df_wiki = pd.DataFrame()
                click.secho(f"[LLM Processor] 警告：无法读取 Wiki DB: {e2}", fg="yellow")

        if not df_wiki.empty:
            # 向量化构建，避免 iterrows 在大 parquet 上的性能问题
            df_wiki = df_wiki.dropna(subset=['title'])
            real_wiki_map = {
                row['title']: row['body']
                for _, row in df_wiki[['title', 'body']].iterrows()
                if isinstance(row['body'], str) and row['body'].strip()
            }
            other_names_map = {
                row['title']: _parse_other_names(row['other_names'])
                for _, row in df_wiki[['title', 'other_names']].iterrows()
            }
            click.secho(
                f"[LLM Processor] Wiki DB 已加载：{len(real_wiki_map)} 条 body，"
                f"{sum(1 for v in other_names_map.values() if v)} 条含 other_names。",
                fg="blue"
            )

    indices_entity = []
    indices_general = []

    for idx, row in df_all.iterrows():
        name = row['name']
        if name in history_names or name in temp_records:
            continue

        cat = row['category']
        current_wiki = str(row['wiki']).strip()
        already_done = len(current_wiki) >= 2  # wiki 字段已有内容，视为已处理

        if cat in ['3', '4', 3, 4]:
            if not already_done:
                indices_entity.append(idx)
        elif not already_done:
            indices_general.append(idx)

    if preview:
        click.secho(
            f"\n[LLM Processor] 待处理统计 [实体处理: {len(indices_entity)} 条] | [普通标签: {len(indices_general)} 条]",
            fg="magenta")
        return

    current_run_processed = set()

    total_entity = len(indices_entity)
    if total_entity > 0:
        click.secho(f"\n开始执行实体处理 (共 {total_entity} 条)...", fg="cyan")
        for i in range(0, total_entity, batch_size):
            batch_idx = indices_entity[i: i + batch_size]
            request_payload = []

            for idx in batch_idx:
                row = df_all.iloc[idx]
                tag_name = row['name']
                category = int(row['category']) if str(row['category']).lstrip('-').isdigit() else -1

                # -------------------------------------------------------
                # ref_cn 三步走 + ref_wiki 三级优先级（均独立处理）
                # -------------------------------------------------------

                # other_names 已在构建缓存时反序列化，直接取用
                other_names_list = other_names_map.get(tag_name, [])

                ref_cn = ""
                ref_wiki = ""

                # === ref_wiki 优先级1：本地 wiki_pages body（原始文本，此处清洗）===
                ref_wiki = clean_wiki_text(real_wiki_map.get(tag_name, ""))
                dbg(f"[{tag_name}] other_names 列表", other_names_list)
                dbg(f"[{tag_name}] ref_wiki (前200字)",
                    ref_wiki[:200] if ref_wiki else "(空)")

                # === ref_cn Step1：从 Wiki other_names 提取中文别名 ===
                cn_from_wiki = extract_chinese_from_other_names(other_names_list)
                if cn_from_wiki:
                    ref_cn = cn_from_wiki
                    click.secho(f"  [Step1/Wiki] {tag_name} → {ref_cn}", fg="blue")

                # === ref_cn Step2：Bangumi 搜索（Step1 未命中时执行）===
                bangumi_summary = ""
                if not ref_cn:
                    ext_info = fetch_entity_info(tag_name, category, BANGUMI_TOKEN)
                    if ext_info["cn_name"]:
                        ref_cn = ext_info["cn_name"]
                        bangumi_summary = ext_info["summary"]
                        click.secho(f"  [Step2/Bangumi] {tag_name} → {ref_cn}", fg="blue")

                # === ref_wiki 优先级2：Bangumi summary（本地 Wiki 无内容时补充）===
                if not ref_wiki and bangumi_summary:
                    ref_wiki = bangumi_summary

                # === ref_cn Step3：交由 LLM 知识库决定（ref_cn 留空即可）===
                if not ref_cn:
                    click.secho(f"  [Step3/LLM] {tag_name} → 交由 LLM 知识库决定", fg="yellow")
                # ref_wiki 优先级3：ref_wiki 同样留空，LLM prompt 会说明自行生成

                request_payload.append({
                    "name": tag_name,
                    "raw_cn_name": str(row['cn_name']),
                    "ref_cn": ref_cn,
                    "ref_wiki": ref_wiki,
                    "other_names": [n for n in other_names_list if isinstance(n, str) and n.strip()],
                })


            click.echo(f"[LLM Processor] 实体处理进度: {min(i + batch_size, total_entity)}/{total_entity} ...")
            results = process_batch_smart(client, model_name, request_payload, "entity", debug=_DEBUG)

            if results:
                with open(temp_path, 'a', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        current_run_processed.add(item['name'])

                for item in results:
                    name = item['name']
                    idx_match = df_all[df_all['name'] == name].index
                    if not idx_match.empty:
                        idx_val = idx_match[0]
                        base_cn = str(item.get('cn_name', '')).strip()
                        ext_cn = str(item.get('extended_cn_name', '')).strip()
                        combined_cn = re.sub(r',+', ',', ",".join(filter(None, [base_cn, ext_cn])).strip(','))
                        if combined_cn: df_all.at[idx_val, 'cn_name'] = combined_cn
                        if item.get('chinese_wiki'): df_all.at[idx_val, 'wiki'] = item['chinese_wiki']
                        if 'nsfw' in item: df_all.at[idx_val, 'nsfw'] = str(item.get('nsfw', 0))
            time.sleep(1)

    total_general = len(indices_general)
    if total_general > 0:
        click.secho(f"\n开始执行常规翻译与无Wiki兜底 (共待筛查 {total_general} 条)...", fg="cyan")

        batch_translate = []
        batch_fallback = []
        processed_count = 0

        def flush_batch(payload_list, mode):
            nonlocal processed_count
            if not payload_list: return

            clean_payload = [{k: v for k, v in item.items() if k != 'idx'} for item in payload_list]
            mode_name = "常规翻译" if mode == "general" else "专属兜底生成"
            click.echo(f"[LLM Processor] [{mode_name}] 进度: {processed_count}/{total_general} ...")

            results = process_batch_smart(client, model_name, clean_payload, mode, debug=_DEBUG)
            if results:
                with open(temp_path, 'a', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        current_run_processed.add(item['name'])

                for item in results:
                    name = item['name']
                    idx_match = df_all[df_all['name'] == name].index
                    if not idx_match.empty:
                        idx_val = idx_match[0]
                        base_cn = str(item.get('cn_name', '')).strip()
                        ext_cn = str(item.get('extended_cn_name', '')).strip()
                        combined_cn = re.sub(r',+', ',', ",".join(filter(None, [base_cn, ext_cn])).strip(','))
                        if combined_cn: df_all.at[idx_val, 'cn_name'] = combined_cn
                        if item.get('chinese_wiki'): df_all.at[idx_val, 'wiki'] = item['chinese_wiki']
                        if 'nsfw' in item: df_all.at[idx_val, 'nsfw'] = str(item.get('nsfw', 0))
            time.sleep(1)
            payload_list.clear()

        for idx in indices_general:
            row = df_all.iloc[idx]
            tag_name = row['name']

            # other_names：本地缓存优先
            other_names_list = other_names_map.get(tag_name, [])

            # body 为原始文本，此处清洗后使用
            clean_wiki = clean_wiki_text(real_wiki_map.get(tag_name, ""))
            dbg(f"[{tag_name}] other_names 列表", other_names_list)
            dbg(f"[{tag_name}] wiki (前200字)",
                clean_wiki[:200] if clean_wiki else "(空)")
            clean_other_names = [n for n in other_names_list if isinstance(n, str) and n.strip()]

            # 同样先尝试从 other_names 提取中文别名，作为翻译参考传入
            cn_hint = extract_chinese_from_other_names(other_names_list)

            if clean_wiki:
                batch_translate.append({
                    "name": tag_name,
                    "cn_name": str(row['cn_name']),
                    "wiki_data": clean_wiki,
                    "other_names": clean_other_names,
                    "cn_hint": cn_hint,   # 可能为空串，LLM 自行参考
                    "idx": idx,
                })
            else:
                batch_fallback.append({
                    "name": tag_name,
                    "cn_name": str(row['cn_name']),
                    "other_names": clean_other_names,
                    "cn_hint": cn_hint,
                    "idx": idx,
                })

            processed_count += 1

            if len(batch_translate) >= batch_size:
                flush_batch(batch_translate, "general")
            if len(batch_fallback) >= batch_size:
                flush_batch(batch_fallback, "fallback")

        flush_batch(batch_translate, "general")
        flush_batch(batch_fallback, "fallback")

    if current_run_processed or temp_records:
        click.secho(f"\n[LLM Processor] API 环节结束，开始安全合盘...", fg="cyan")
        df_all.to_csv(csv_path, index=False, encoding='utf-8')
        click.secho(f"[LLM Processor] 主结果文件已覆写 (总计 {len(df_all)} 条)。", fg="green")

        history_names.update(temp_records.keys())
        history_names.update(current_run_processed)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(list(history_names), f, ensure_ascii=False)
        click.secho(f"[LLM Processor] 长期历史记录已更新 (共豁免 {len(history_names)} 词条)。", fg="green")

        if temp_path.exists():
            os.remove(temp_path)
            click.secho("[LLM Processor] 临时过程文件已清理。", fg="green")
    else:
        click.secho("\n  [LLM Processor] 本次运行没有需要更新的数据。", fg="green")