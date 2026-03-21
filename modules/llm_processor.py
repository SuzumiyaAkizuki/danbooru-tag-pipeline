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


# ================= 🛠️ 辅助函数 =================

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
    return re.sub(r'_\(.*\)$', '', str(tag_name)).replace('_', ' ').strip()


def clean_wiki_text(text):
    if not isinstance(text, str) or not text: return ""
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = text.replace("'''", "")
    text = re.sub(r'h\d\.\s*', '', text)
    return text.replace('\n', ' ').replace('\r', '')[:400].strip()


def fetch_danbooru_wiki_single(tag_name, db_user, db_key):
    """当本地数据库找不到时，直接向 Danbooru 发起单点查询"""
    url = "https://danbooru.donmai.us/wiki_pages.json"
    params = {"search[title]": tag_name}
    if db_user and db_key:
        params['login'] = db_user
        params['api_key'] = db_key

    headers = {"User-Agent": "sakizuki/danbooru_tag_bot"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                return data[0].get('body', '')
    except Exception as e:
        click.secho(f"[LLM Processor] Danbooru 精准打捞失败 ({tag_name}): {e}", fg="yellow")
    return ""


def fetch_entity_info(tag_name, category, token):
    """带 3 次重试机制的 Bangumi API 查询"""
    clean_name = clean_tag_name(tag_name)
    headers = {
        "User-Agent": "sakizuki/danbooru_tag_bot (https://github.com/sakizuki)",
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
                        result["cn_name"] = item.get('name_cn') or item.get('name')
                        if item.get('summary'):
                            result["summary"] = item.get('summary', '').replace('\r', '').replace('\n', '')[:200]
                break

            elif category == 4:
                url = "https://api.bgm.tv/v0/search/characters"
                payload = {"keyword": clean_name, "filter": {"nsfw": True}}
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    char_list = data.get('data', [])
                    if char_list:
                        char_data = char_list[0]
                        if char_data.get('summary'):
                            result["summary"] = char_data.get('summary', '').replace('\r', '').replace('\n', '')[:200]
                        result["cn_name"] = char_data.get('name', '')
                        for info in char_data.get('infobox', []):
                            if info.get('key') == '简体中文名':
                                val = info.get('value')
                                if isinstance(val, str):
                                    result["cn_name"] = val
                                elif isinstance(val, list) and len(val) > 0:
                                    result["cn_name"] = val[0].get('v') if isinstance(val[0], dict) else val[0]
                                break
                break
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2)
            else:
                click.secho(f"[LLM Processor] Bangumi API 彻底失败 ({tag_name}): {e}", fg="yellow")
        except Exception as e:
            click.secho(f"[LLM Processor] Bangumi 解析异常 ({tag_name}): {e}", fg="yellow")
            break
    return result


def process_batch_smart(client, model_name, batch_data, mode="general"):
    if mode == "general":
        system_prompt = """
                # Role
                你是一个 Danbooru 标签数据库的专家。

                # Task
                用户会提供一批标签数据。`wiki_data` 是官方英文描述，但可能缺失。请完成以下四个动作：

                1. **生成中文描述 (chinese_wiki)**:
                   - 提取 `wiki_data` 里的核心信息进行翻译或总结。结合知识库，生成一句简练的中文视觉描述（50字左右）。
                   - 如果 `wiki_data` 为空或无效，请根据你的知识库写一句该标签的中文视觉描述。注意：不要在输出里包含任何字数统计或备注信息，只输出描述本身。
                   - 如果知识库中没有相关信息，`wiki_data` 也无效，返回空字符串。

                2. **修正中文名 (cn_name)**:
                   - 结合 `wiki_data` 的真实含义，严格检查传入的 `cn_name` 是否准确。如果原中文名存在机翻错误、词不达意或完全不贴切，请将其修正为二次元语境下最准确的【基础中文标签名】。

                3. **扩展中文名 (extended_cn_name)**:
                   - 根据修正后的基础标签含义，生成 3~4 个相关的追加【同义词】、【风格】或【大类】（用半角逗号分隔）。注意：这里只写扩展词，不要包含刚才的基础中文名。
                   - 例：对"流苏",生成"中国风, 挂饰".

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
        # 🌟 完美复刻原有的无 Wiki 兜底逻辑
        system_prompt = """
        # Role
        你是一个 Danbooru 标签数据库的资深专家。
        # Task
        用户会提供一批**缺失 Wiki 描述**的普通标签。请根据标签的英文名(name)和现有的中文名(cn_name)，凭借你的内部知识库完成以下任务：
        1. **生成中文描述 (chinese_wiki)**: 解释这个标签的视觉定义或含义。生成一句简练的中文描述（30字左右）。如果完全无法识别该标签，请返回空字符串。
        2. **修正中文名 (cn_name)**: 严格检查传入的 `cn_name` 是否准确。如果存在机翻错误、词不达意，请将其修正为二次元语境下最准确的【基础中文标签名】。
        3. **扩展中文名 (extended_cn_name)**: 生成 3~4 个相关的【同义词】、【风格】或【大类】（用半角逗号分隔）。注意：只写扩展词，不要包含基础中文名。
        4. **NSFW 判定 (nsfw)**: 包含裸露、性行为、生殖器、恋物癖(Fetish)、血腥暴力则为 1，否则为 0。
        # Output Format (JSON)
        {"items": [{"name": "...", "cn_name": "...", "extended_cn_name": "...", "chinese_wiki": "...", "nsfw": "0"}]}
        """
        temp = 0.5
    else:
        system_prompt = """
        # Role
        你是一个严谨的 ACG 领域防幻觉整理专家。

        # Task
        用户会提供【角色名】/【作品名】的标签数据。`ref_cn`是外部API抓取的官方中文名，`ref_wiki`是其简介。请完成以下四个动作：

        1. **生成中文描述 (chinese_wiki)**:
           - 优先提炼 `ref_wiki` 为 50 字左右的中文简述。若 `ref_wiki` 为空，请根据你的知识库写 50 字简介。
           - 不得在输出的中文描述里包含类似于"(50字)"的字数统计，或任何备注信息，只允许输出描述本身。

        2. **确定中文名 (cn_name)**:
           - 如果 `ref_cn` 存在，请采纳其作为基础中文名。若 `ref_cn` 为空且你 **没有百分百把握** 确定官方汉字，必须保留原英文名，绝不瞎猜音译！

        3. **扩展中文名 (extended_cn_name)**:
           - 为该角色或作品生成 2~3 个相关的作品前缀、所属阵营或别名（用半角逗号分隔，例如：对"蝶祈祷"生成"罪恶王冠,祈妹"）。注意：只写扩展词，不要包含基础中文名。
           - 如果 `ref_cn` 不存在，请根据你的知识库在此生成该标签的中文翻译和相关的作品前缀、所属阵营或别名。

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

    for attempt in range(3):
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
            return json.loads(response.choices[0].message.content).get("items", [])
        except Exception as e:
            click.secho(f"[LLM Processor] LLM 出错 (尝试 {attempt + 1}/3): {e}", fg="yellow")
            time.sleep(2)
    return []


# ================= 🚀 核心执行 =================

def run(config, preview=False):
    base_dir = Path(__file__).resolve().parent.parent

    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    BANGUMI_TOKEN = os.getenv("BANGUMI_ACCESS_TOKEN")
    DB_USER = os.getenv("DANBOORU_USER_NAME")
    DB_KEY = os.getenv("DANBOORU_API_KEY")

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
        click.secho(f"[LLM Processor] 检测到异常中断！恢复 {len(temp_records)} 条未合并的临时进度。", fg="yellow")

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

    real_wiki_map = {}
    if wiki_path.exists():
        try:
            df_wiki = pd.read_parquet(wiki_path, columns=['title', 'body'])
            real_wiki_map = {row['title']: row['body'] for _, row in df_wiki.iterrows() if
                             isinstance(row['body'], str) and row['body'].strip()}
        except Exception as e:
            click.secho(f"[LLM Processor] 警告：无法读取 Wiki DB: {e}", fg="yellow")

    indices_entity = []
    indices_general = []

    for idx, row in df_all.iterrows():
        name = row['name']
        if name in history_names or name in temp_records:
            continue

        cat = row['category']
        current_wiki = str(row['wiki']).strip()
        has_real_wiki = name in real_wiki_map

        if cat in [3, 4] and not has_real_wiki:
            indices_entity.append(idx)
        elif len(current_wiki) < 2:
            indices_general.append(idx)

    if preview:
        click.secho(
            f"\n[LLM Processor] 待处理统计 [实体幻觉重写: {len(indices_entity)} 条] | [待定普通标签(将分发至翻译或兜底): {len(indices_general)} 条]",
            fg="magenta")
        return

    current_run_processed = set()

    # === 处理第一部分：实体标签 ===
    total_entity = len(indices_entity)
    if total_entity > 0:
        click.secho(f"\n>>> 开始执行 [实体防幻觉重写] (共 {total_entity} 条)...", fg="cyan")
        for i in range(0, total_entity, batch_size):
            batch_idx = indices_entity[i: i + batch_size]
            request_payload = []

            for idx in batch_idx:
                row = df_all.iloc[idx]
                tag_name = row['name']
                ext_info = fetch_entity_info(tag_name, row['category'], BANGUMI_TOKEN)
                request_payload.append({
                    "name": tag_name, "raw_cn_name": str(row['cn_name']),
                    "ref_cn": ext_info['cn_name'], "ref_wiki": ext_info['summary']
                })

            click.echo(f"[LLM Processor] 实体处理进度: {min(i + batch_size, total_entity)}/{total_entity} ...")
            results = process_batch_smart(client, model_name, request_payload, "entity")

            if results:
                with open(temp_path, 'a', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        current_run_processed.add(item['name'])

                for item in results:
                    name = item['name']
                    idx_match = df_all[df_all['name'] == name].index
                    if not idx_match.empty:
                        idx = idx_match[0]
                        base_cn = str(item.get('cn_name', '')).strip()
                        ext_cn = str(item.get('extended_cn_name', '')).strip()
                        combined_cn = re.sub(r',+', ',', ",".join(filter(None, [base_cn, ext_cn])).strip(','))
                        if combined_cn: df_all.at[idx, 'cn_name'] = combined_cn
                        if item.get('chinese_wiki'): df_all.at[idx, 'wiki'] = item['chinese_wiki']
                        if 'nsfw' in item: df_all.at[idx, 'nsfw'] = str(item.get('nsfw', 0))
            time.sleep(1)

    # === 处理第二部分：动态分桶 (翻译 / 无Wiki兜底) ===
    total_general = len(indices_general)
    if total_general > 0:
        click.secho(f"\n>>> 开始执行 [常规翻译与无Wiki兜底] (共待筛查 {total_general} 条)...", fg="cyan")

        batch_translate = []
        batch_fallback = []
        processed_count = 0

        def flush_batch(payload_list, mode):
            nonlocal processed_count
            if not payload_list: return

            # 清理载荷中的辅助字段
            clean_payload = [{k: v for k, v in item.items() if k != 'idx'} for item in payload_list]
            mode_name = "常规翻译" if mode == "general" else "专属兜底生成"
            click.echo(f"[LLM Processor] [{mode_name}] 进度: {processed_count}/{total_general} ...")

            results = process_batch_smart(client, model_name, clean_payload, mode)
            if results:
                with open(temp_path, 'a', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        current_run_processed.add(item['name'])

                for item in results:
                    name = item['name']
                    idx_match = df_all[df_all['name'] == name].index
                    if not idx_match.empty:
                        idx = idx_match[0]
                        base_cn = str(item.get('cn_name', '')).strip()
                        ext_cn = str(item.get('extended_cn_name', '')).strip()
                        combined_cn = re.sub(r',+', ',', ",".join(filter(None, [base_cn, ext_cn])).strip(','))
                        if combined_cn: df_all.at[idx, 'cn_name'] = combined_cn
                        if item.get('chinese_wiki'): df_all.at[idx, 'wiki'] = item['chinese_wiki']
                        if 'nsfw' in item: df_all.at[idx, 'nsfw'] = str(item.get('nsfw', 0))
            time.sleep(1)
            payload_list.clear()

        for idx in indices_general:
            row = df_all.iloc[idx]
            tag_name = row['name']

            raw_wiki = real_wiki_map.get(tag_name, "")
            # 本地没找到，尝试精确打捞
            if not raw_wiki:
                raw_wiki = fetch_danbooru_wiki_single(tag_name, DB_USER, DB_KEY)

            clean_wiki = clean_wiki_text(raw_wiki)

            # 🌟 核心分桶逻辑
            if clean_wiki:
                batch_translate.append(
                    {"name": tag_name, "cn_name": str(row['cn_name']), "wiki_data": clean_wiki, "idx": idx})
            else:
                batch_fallback.append({"name": tag_name, "cn_name": str(row['cn_name']), "idx": idx})

            processed_count += 1

            # 一旦某个桶满了一个 Batch，立刻发车
            if len(batch_translate) >= batch_size:
                flush_batch(batch_translate, "general")
            if len(batch_fallback) >= batch_size:
                flush_batch(batch_fallback, "fallback")

        # 扫尾发送没装满的桶
        flush_batch(batch_translate, "general")
        flush_batch(batch_fallback, "fallback")

    # ================= 安全合盘 =================
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