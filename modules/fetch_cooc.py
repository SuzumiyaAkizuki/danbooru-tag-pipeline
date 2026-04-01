import requests
import pandas as pd
import time
import os
import json
from pathlib import Path
import click

def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=['name', 'cn_name', 'wiki', 'post_count', 'category', 'nsfw'])
    for enc in ['utf-8', 'gbk', 'gb18030']:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"[Fetch Cooc] 无法读取文件: {path}")

def parse_related_tags(data, tag_a, valid_tags_set, tag_post_counts):
    pairs = []
    query_post_count = data.get("post_count") if isinstance(data, dict) else 0
    if not query_post_count:
        query_post_count = tag_post_counts.get(tag_a, 0)

    if query_post_count <= 0:
        return pairs

    if isinstance(data, dict) and "related_tags" in data:
        for item in data["related_tags"]:
            tag_b = item.get("tag", {}).get("name") or item.get("name")
            if not tag_b or tag_b == tag_a or tag_b not in valid_tags_set:
                continue
            freq = float(item.get("frequency", 0.0))
            cos_sim = float(item.get("cosine_similarity", 0.0))
            pairs.append({"source": tag_a, "target": tag_b, "frequency": freq, "cosine_similarity": cos_sim})

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                tag_b = str(item[0])
                if not tag_b or tag_b == tag_a or tag_b not in valid_tags_set: continue
                try:
                    raw_count = int(item[1])
                    freq = raw_count / query_post_count
                except ValueError:
                    freq = 0.0
                pairs.append({"source": tag_a, "target": tag_b, "frequency": freq, "cosine_similarity": 0.0})
    return pairs

def run(config, full_update=False):
    base_dir = Path(__file__).resolve().parent.parent

    USER_NAME = os.getenv("DANBOORU_USER_NAME")
    API_KEY = os.getenv("DANBOORU_API_KEY")
    if not USER_NAME or not API_KEY:
        click.secho("[Fetch Cooc] 未配置凭证", fg="red")
        return

    HEADERS = {"User-Agent": f"MatrixBuilderBot/3.2 (by {USER_NAME})", "Accept": "application/json"}

    input_csv = base_dir / config['paths']['processed']['tags_enhanced']
    output_raw_csv = base_dir / config['paths']['raw']['cooc_raw_csv']
    progress_file = base_dir / config['paths']['checkpoint']['cooc_progress']
    temp_csv = progress_file.with_name("cooc_temp.csv")
    history_file = base_dir / config['paths']['checkpoint']['cooc_history']

    if not input_csv.exists():
        click.secho("[Fetch Cooc] 找不到标签文件", fg="red")
        return

    df_tags = read_csv_robust(input_csv)
    valid_tags_set = set(df_tags['name'].dropna().unique())
    tag_post_counts = dict(zip(df_tags['name'], pd.to_numeric(df_tags['post_count'], errors='coerce').fillna(0)))

    history_tags = set()
    start_index = 0

    if full_update:
        click.secho("[Fetch Cooc] 全量更新模式", fg="red", bold=True)
        if progress_file.exists(): progress_file.unlink()
    else:
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_tags = set(json.load(f))
            except Exception as e:
                pass

        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    start_index = int(f.read().strip())
            except ValueError:
                pass

    target_tags_list = list(valid_tags_set - history_tags)
    total_tags = len(target_tags_list)

    if total_tags == 0:
        click.secho("[Fetch Cooc] 无需抓取", fg="green")
        return

    new_records_batch = []
    current_run_processed = set()

    for i in range(start_index, total_tags):
        tag_a = target_tags_list[i]
        click.echo(f"[{i + 1}/{total_tags}] 获取 '{tag_a}' ...")

        url = "https://danbooru.donmai.us/related_tag.json"
        params = {'query': tag_a, 'login': USER_NAME, 'api_key': API_KEY}

        success = False
        while not success:
            try:
                resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
                if resp.status_code == 429:
                    time.sleep(180)
                    continue
                elif resp.status_code == 403:
                    return
                elif resp.status_code >= 500:
                    time.sleep(30)
                    continue

                resp.raise_for_status()
                pairs = parse_related_tags(resp.json(), tag_a, valid_tags_set, tag_post_counts)
                new_records_batch.extend(pairs)
                current_run_processed.add(tag_a)
                success = True

            except requests.exceptions.RequestException as e:
                time.sleep(60)

        time.sleep(1.5)

        if (i + 1) % 100 == 0 or (i + 1) == total_tags:
            if new_records_batch:
                df_temp = pd.DataFrame(new_records_batch)
                df_temp.to_csv(temp_csv, mode='a', header=not temp_csv.exists(), index=False, encoding='utf-8')
                new_records_batch.clear()

            with open(progress_file, 'w') as f:
                f.write(str(i + 1))

            history_tags.update(current_run_processed)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(list(history_tags), f, ensure_ascii=False)
            current_run_processed.clear()

            if (i + 1) % 100 == 0:
                time.sleep(30)

    if temp_csv.exists():
        df_new = pd.read_csv(temp_csv, low_memory=False, encoding='utf-8')

        if not full_update and output_raw_csv.exists():
            df_old = pd.read_csv(output_raw_csv, low_memory=False, encoding='utf-8')
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all = df_all.sort_values(by=['frequency', 'cosine_similarity'], ascending=[False, False])
        df_all = df_all.drop_duplicates(subset=['source', 'target'], keep='first')

        df_all.to_csv(output_raw_csv, index=False, encoding='utf-8')
        os.remove(temp_csv)
        if progress_file.exists(): os.remove(progress_file)