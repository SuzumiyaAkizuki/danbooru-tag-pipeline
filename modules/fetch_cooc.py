import requests
import pandas as pd
import time
import os
import json
from pathlib import Path
import click
from modules.trash import trash_file

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
    """返回 (tag_pairs, artist_edges)。

    tag_pairs: 标签-标签共现边，格式不变
    artist_edges: 标签-画师共现边，仅从 dict 格式响应中提取 category=1 的画师
    """
    tag_pairs = []
    artist_edges = []
    query_post_count = data.get("post_count") if isinstance(data, dict) else 0
    if not query_post_count:
        query_post_count = tag_post_counts.get(tag_a, 0)

    if query_post_count <= 0:
        return tag_pairs, artist_edges

    if isinstance(data, dict) and "related_tags" in data:
        for item in data["related_tags"]:
            # ── 解析 tag 信息 ──
            tag_info = item.get("tag", {}) if isinstance(item.get("tag"), dict) else {}
            tag_b = tag_info.get("name") or item.get("name")
            if not tag_b or tag_b == tag_a:
                continue
            freq = float(item.get("frequency", 0.0))
            if freq <= 0.0:
                continue

            category = tag_info.get("category", -1)
            related_post_count = tag_info.get("post_count", 0)

            # ── 画师边（完全独立于标签边）──
            if category == 1 and related_post_count > 100:
                cooc_count = round(freq * query_post_count)
                artist_edges.append({
                    "tag": tag_a,
                    "artist": tag_b,
                    "artist_post_count": related_post_count,
                    "cooc_count": cooc_count,
                    "frequency": round(freq, 6),
                })

            # ── 标签边（category 0/3/4，且在 white list 中）──
            if tag_b not in valid_tags_set:
                continue
            cos_sim = float(item.get("cosine_similarity", 0.0))
            tag_pairs.append({
                "source": tag_a,
                "target": tag_b,
                "frequency": freq,
                "cosine_similarity": cos_sim,
            })

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                tag_b = str(item[0])
                if not tag_b or tag_b == tag_a or tag_b not in valid_tags_set:
                    continue
                try:
                    raw_count = int(item[1])
                    freq = raw_count / query_post_count
                except ValueError:
                    freq = 0.0
                tag_pairs.append({
                    "source": tag_a,
                    "target": tag_b,
                    "frequency": freq,
                    "cosine_similarity": 0.0,
                })

    return tag_pairs, artist_edges

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
    artist_parquet = base_dir / config['paths']['processed']['tag_artist_cooc']
    progress_file = base_dir / config['paths']['checkpoint']['cooc_progress']
    temp_csv = progress_file.with_name("cooc_temp.csv")
    temp_artist_csv = progress_file.with_name("artist_cooc_temp.csv")
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
        if progress_file.exists():
            # 上次全量模式中断，断点续传
            click.secho("[Fetch Cooc] 全量模式 — 从断点续传...", fg="yellow", bold=True)
            if history_file.exists():
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history_tags = set(json.load(f))
                    click.echo(f"[Fetch Cooc] 已恢复 {len(history_tags)} 个已抓取标签")
                except Exception:
                    pass
            try:
                with open(progress_file, 'r') as f:
                    start_index = int(f.read().strip())
            except ValueError:
                pass
        else:
            # 首次启动全量模式，清空旧状态从头开始
            click.secho("[Fetch Cooc] 全量更新模式（首次启动）", fg="red", bold=True)
            if history_file.exists():
                trash_file(base_dir, history_file)
            if temp_csv.exists():
                trash_file(base_dir, temp_csv)
            if temp_artist_csv.exists():
                trash_file(base_dir, temp_artist_csv)
    else:
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_tags = set(json.load(f))
            except Exception:
                pass

        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    start_index = int(f.read().strip())
            except ValueError:
                pass

    # 全量模式不过滤 history（用 start_index 跳过已完成部分），增量模式过滤
    if full_update:
        target_tags_list = list(valid_tags_set)  # set 迭代顺序在同机器上稳定
    else:
        target_tags_list = sorted(valid_tags_set - history_tags)
    total_tags = len(target_tags_list)

    if total_tags == 0:
        click.secho("[Fetch Cooc] 无需抓取", fg="green")
        return

    new_records_batch = []
    artist_records_batch = []
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
                tag_pairs, artist_edges = parse_related_tags(
                    resp.json(), tag_a, valid_tags_set, tag_post_counts
                )
                new_records_batch.extend(tag_pairs)
                if artist_edges:
                    artist_records_batch.extend(artist_edges)
                current_run_processed.add(tag_a)
                success = True

            except requests.exceptions.RequestException as e:
                time.sleep(60)

        time.sleep(1.5)

        if (i + 1) % 100 == 0 or (i + 1) == total_tags:
            # 标签-标签边
            if new_records_batch:
                df_temp = pd.DataFrame(new_records_batch)
                df_temp.to_csv(temp_csv, mode='a', header=not temp_csv.exists(), index=False, encoding='utf-8')
                new_records_batch.clear()

            # 标签-画师边（独立文件，不污染标签数据）
            if artist_records_batch:
                df_artist_temp = pd.DataFrame(artist_records_batch)
                df_artist_temp.to_csv(
                    temp_artist_csv, mode='a',
                    header=not temp_artist_csv.exists(),
                    index=False, encoding='utf-8',
                )
                artist_records_batch.clear()

            with open(progress_file, 'w') as f:
                f.write(str(i + 1))

            history_tags.update(current_run_processed)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(list(history_tags), f, ensure_ascii=False)
            current_run_processed.clear()

            if (i + 1) % 100 == 0:
                time.sleep(30)

    # ── 标签-标签边 merge ──────────────────────────────────────────────
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
        trash_file(base_dir, temp_csv)

    # ── 标签-画师边 merge ──────────────────────────────────────────────
    if temp_artist_csv.exists():
        df_artist_new = pd.read_csv(temp_artist_csv, low_memory=False, encoding='utf-8')

        # 同一 (tag, artist) 对取最大 cooc_count
        df_artist_new = (
            df_artist_new
            .groupby(["tag", "artist"], as_index=False)
            .agg({
                "artist_post_count": "max",
                "cooc_count": "max",
                "frequency": "max",
            })
        )

        # 至少 3 张帖子的共现
        before = len(df_artist_new)
        df_artist_new = df_artist_new[df_artist_new["cooc_count"] >= 3].copy()
        if before > len(df_artist_new):
            click.echo(
                f"[Fetch Cooc] 画师边 cooc_count>=3 过滤: {before:,} → {len(df_artist_new):,}"
            )

        if artist_parquet.exists():
            df_artist_old = pd.read_parquet(str(artist_parquet))
            # 清除旧数据中可能残留的 pmi/npmi 列
            for c in ["pmi", "npmi"]:
                if c in df_artist_old.columns:
                    df_artist_old = df_artist_old.drop(columns=[c])
            # 合并两个视角的数据：同名对取 max cooc_count，旧的无冲突行保留
            old_keys = set(zip(df_artist_old["tag"], df_artist_old["artist"]))
            new_keys = set(zip(df_artist_new["tag"], df_artist_new["artist"]))
            keep_mask = [k not in new_keys for k in old_keys]
            df_artist_final = pd.concat(
                [df_artist_old[keep_mask], df_artist_new], ignore_index=True
            )
        else:
            df_artist_final = df_artist_new

        # 最终去重：合并后可能残留跨来源的同名对，取 max cooc_count
        df_artist_final = (
            df_artist_final
            .groupby(["tag", "artist"], as_index=False)
            .agg({
                "artist_post_count": "max",
                "cooc_count": "max",
                "frequency": "max",
            })
        )
        df_artist_final.sort_values(["tag", "cooc_count"], ascending=[True, False], inplace=True)
        df_artist_final.reset_index(drop=True, inplace=True)

        artist_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_artist_final.to_parquet(str(artist_parquet), index=False, compression="snappy")

        n_artist_edges = len(df_artist_final)
        n_tags = df_artist_final["tag"].nunique()
        n_artists = df_artist_final["artist"].nunique()
        click.secho(
            f"[Fetch Cooc] 画师共现: {n_artist_edges:,} 条边, "
            f"{n_tags} tags, {n_artists} artists → {artist_parquet.name}",
            fg="green",
        )
        trash_file(base_dir, temp_artist_csv)

    if progress_file.exists():
        trash_file(base_dir, progress_file)