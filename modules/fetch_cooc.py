import requests
import pandas as pd
import time
import os
import json
from pathlib import Path
import click

def read_csv_robust(path: Path) -> pd.DataFrame:
    """鲁棒地读取 CSV 文件"""
    if not path.exists():
        return pd.DataFrame(columns=['name', 'cn_name', 'wiki', 'post_count', 'category', 'nsfw'])
    for enc in ['utf-8', 'gbk', 'gb18030']:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"[Fetch Cooc] 无法读取文件（编码未知或文件损坏）：{path}")

def parse_related_tags(data, tag_a, valid_tags_set, tag_post_counts):
    """智能解析关联标签，提取纯共现次数 (raw_count)"""
    pairs = []
    query_post_count = data.get("post_count") if isinstance(data, dict) else 0
    if not query_post_count:
        query_post_count = tag_post_counts.get(tag_a, 0)

    if isinstance(data, dict) and "related_tags" in data:
        for item in data["related_tags"]:
            tag_b = item.get("tag", {}).get("name") or item.get("name")
            if not tag_b or tag_b == tag_a or tag_b not in valid_tags_set:
                continue
            freq = item.get("frequency", 0)
            raw_count = int(round(freq * query_post_count))
            cos_sim = item.get("cosine_similarity", 0.0)
            t1, t2 = sorted([tag_a, tag_b])
            pairs.append({"tag_a": t1, "tag_b": t2, "raw_count": raw_count, "cosine_similarity": cos_sim})

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                tag_b = str(item[0])
                if not tag_b or tag_b == tag_a or tag_b not in valid_tags_set: continue
                try:
                    raw_count = int(item[1])
                except ValueError:
                    raw_count = 0
                t1, t2 = sorted([tag_a, tag_b])
                pairs.append({"tag_a": t1, "tag_b": t2, "raw_count": raw_count, "cosine_similarity": 0.0})
    return pairs


def run(config, full_update=False):
    base_dir = Path(__file__).resolve().parent.parent

    USER_NAME = os.getenv("DANBOORU_USER_NAME")
    API_KEY = os.getenv("DANBOORU_API_KEY")
    if not USER_NAME or not API_KEY:
        click.secho("[Fetch Cooc] 未在 .env 中配置 Danbooru 凭证！", fg="red")
        return

    HEADERS = {"User-Agent": f"MatrixBuilderBot/3.1 (by {USER_NAME})", "Accept": "application/json"}

    input_csv = base_dir / config['paths']['processed']['tags_enhanced']
    output_raw_csv = base_dir / config['paths']['raw']['cooc_raw_csv']
    progress_file = base_dir / config['paths']['checkpoint']['cooc_progress']
    temp_csv = progress_file.with_name("cooc_temp.csv")
    history_file = base_dir / config['paths']['checkpoint']['cooc_history']

    if not input_csv.exists():
        click.secho(f"[Fetch Cooc] 找不到标签文件 {input_csv}！请先执行前置步骤。", fg="red")
        return

    # 1. 读取所有合法标签
    df_tags = read_csv_robust(input_csv)
    valid_tags_set = set(df_tags['name'].dropna().unique())
    tag_post_counts = dict(zip(df_tags['name'], pd.to_numeric(df_tags['post_count'], errors='coerce').fillna(0)))

    # 2. 读取历史记录 (全量模式下直接清空忽略)
    history_tags = set()
    start_index = 0

    if full_update:
        click.secho("[Fetch Cooc] 开启全量更新模式！将无视历史记录，重新抓取所有标签...", fg="red", bold=True)
        # 强制删除旧的断点文件，确保从 0 开始
        if progress_file.exists(): progress_file.unlink()
    else:
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_tags = set(json.load(f))
                click.secho(f"[Fetch Cooc] 读取共现抓取历史：已豁免 {len(history_tags)} 个老标签。", fg="blue")
            except Exception as e:
                click.secho(f"[Fetch Cooc] 无法读取历史记录 ({e})", fg="yellow")

        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    start_index = int(f.read().strip())
                click.secho(f"[Fetch Cooc] 检测到异常断点，从本次任务的第 {start_index} 个标签继续...", fg="yellow")
            except ValueError:
                pass

    # 3. 计算需要真正去抓取的标签
    target_tags_list = list(valid_tags_set - history_tags)
    total_tags = len(target_tags_list)

    if total_tags == 0:
        click.secho("[Fetch Cooc] 所有标签的共现数据均已抓取过，本次无需发请求。", fg="green")
        return

    click.secho(f"[Fetch Cooc] 共有 {total_tags} 个标签需要抓取共现矩阵...", fg="magenta")

    new_records_batch = []
    current_run_processed = set()

    # 4. 抓取循环
    for i in range(start_index, total_tags):
        tag_a = target_tags_list[i]
        click.echo(f"[{i + 1}/{total_tags}] 正在获取 '{tag_a}' ...")

        url = "https://danbooru.donmai.us/related_tag.json"
        params = {'query': tag_a, 'login': USER_NAME, 'api_key': API_KEY}

        success = False
        while not success:
            try:
                resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
                if resp.status_code == 429:
                    click.secho("[Fetch Cooc] 触发频率限制，休眠 3 分钟...", fg="yellow")
                    time.sleep(180)
                    continue
                elif resp.status_code == 403:
                    click.secho("[Fetch Cooc] 收到 403 错误，请检查凭证！", fg="red")
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
                click.secho(f"[Fetch Cooc] 网络异常: {e}，休眠 60 秒后重试...", fg="red")
                time.sleep(60)

        time.sleep(1.5)

        # 🔒 断点续传存盘
        if (i + 1) % 100 == 0 or (i + 1) == total_tags:
            if new_records_batch:
                df_temp = pd.DataFrame(new_records_batch)
                df_temp.to_csv(temp_csv, mode='a', header=not temp_csv.exists(), index=False, encoding='utf-8')
                new_records_batch.clear()

            with open(progress_file, 'w') as f:
                f.write(str(i + 1))

            # 实时更新历史记录
            history_tags.update(current_run_processed)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(list(history_tags), f, ensure_ascii=False)
            current_run_processed.clear()

            if (i + 1) % 100 == 0:
                click.secho(f"[Fetch Cooc] 已处理 {i + 1} 个标签，存盘并强制休息 30 秒...", fg="blue")
                time.sleep(30)

    # ================= 收尾合并与去重 =================
    click.secho("\n[Fetch Cooc] 抓取结束！开始处理共现矩阵...", fg="cyan")

    if temp_csv.exists():
        df_new = pd.read_csv(temp_csv, low_memory=False, encoding='utf-8')

        # 如果是全量更新，我们直接用新抓的数据覆盖旧矩阵；如果是增量，则进行合并
        if not full_update and output_raw_csv.exists():
            click.secho("[Fetch Cooc] 正在与历史共现表进行增量合并...", fg="blue")
            df_old = pd.read_csv(output_raw_csv, low_memory=False, encoding='utf-8')
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            if full_update:
                click.secho("[Fetch Cooc] 全量更新模式：将直接用最新数据覆写旧共现表...", fg="blue")
            df_all = df_new

        # 全局去重
        df_all = df_all.sort_values(by=['raw_count', 'cosine_similarity'], ascending=[False, False])
        df_all = df_all.drop_duplicates(subset=['tag_a', 'tag_b'], keep='first')

        df_all.to_csv(output_raw_csv, index=False, encoding='utf-8')
        click.secho(f"[Fetch Cooc] 原始共现矩阵保存成功！共 {len(df_all)} 条数据。", fg="green")

        os.remove(temp_csv)
        if progress_file.exists(): os.remove(progress_file)
        click.secho("[Fetch Cooc] 临时文件已清理。", fg="green")
    else:
        click.secho("[Fetch Cooc] 没有产生新的缓存文件数据。", fg="green")