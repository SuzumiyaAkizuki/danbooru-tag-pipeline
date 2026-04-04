import json
import requests
import pandas as pd
import time
import os
import random
from dateutil import parser
from pathlib import Path
import click


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def get_local_latest_time(parquet_path):
    if parquet_path.exists():
        try:
            df_local = pd.read_parquet(parquet_path)
            latest_time_str = df_local['updated_at'].dropna().max()
            latest_time = parser.parse(latest_time_str)
            click.secho(f"[Fetch Wiki] 本地主文件最新时间线: {latest_time}", fg="blue")
            return latest_time, df_local
        except Exception as e:
            click.secho(f"[Fetch Wiki] 读取本地 Parquet 出错: {e}，将全量抓取。", fg="yellow")
    else:
        click.secho("[Fetch Wiki] 未找到本地 Wiki 数据库，将进行全量抓取...", fg="yellow")
    return parser.parse("2000-01-01T00:00:00Z"), pd.DataFrame()


def _normalize_entry(entry: dict) -> dict:
    """
    在写入 parquet 前统一处理单条 wiki 记录：
    - other_names：统一序列化为 JSON 字符串（双引号列表），消除 str(list) 的单引号/numpy 格式歧义
    - body：保持原始文本不变，清洗由 llm_processor.py 在读取时完成
    """
    # other_names：确保是列表再序列化
    raw_other = entry.get('other_names', [])
    if isinstance(raw_other, str):
        # 兼容旧数据：尝试反序列化后再重新序列化
        try:
            parsed = json.loads(raw_other)
            if isinstance(parsed, list):
                raw_other = parsed
        except Exception:
            pass
        if isinstance(raw_other, str):
            import ast
            try:
                parsed = ast.literal_eval(raw_other)
                if isinstance(parsed, list):
                    raw_other = parsed
            except Exception:
                raw_other = [raw_other] if raw_other.strip() else []
    if not isinstance(raw_other, list):
        raw_other = []
    entry['other_names'] = json.dumps(raw_other, ensure_ascii=False)

    return entry


def run(config):
    base_dir = Path(__file__).resolve().parent.parent

    USER_NAME = os.getenv("DANBOORU_USER_NAME")
    API_KEY = os.getenv("DANBOORU_API_KEY")

    if not USER_NAME or not API_KEY:
        click.secho("[Fetch Wiki] 错误：未在 .env 中配置 DANBOORU_USER_NAME 或 DANBOORU_API_KEY", fg="red")
        return

    API_URL = "https://danbooru.donmai.us/wiki_pages.json"
    HEADERS = {"User-Agent": f"WikiUpdateBot/1.0 (by {USER_NAME})", "Accept": "application/json"}

    parquet_path = base_dir / config['paths']['processed']['wiki_parquet']
    progress_file = base_dir / config['paths']['checkpoint']['wiki_progress']
    temp_csv_file = progress_file.with_name("wiki_temp.csv")

    last_update_time, df_local = get_local_latest_time(parquet_path)

    current_page = 1
    current_upper_bound = None

    # 断点续传
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                lines = f.read().splitlines()
                if lines:
                    saved_page = int(lines[0].strip())
                    current_page = max(1, saved_page - 2)
                    if len(lines) > 1 and lines[1].strip():
                        current_upper_bound = lines[1].strip()
            click.secho(f"[Fetch Wiki] 检测到中断记录！自动回退，将从第 {current_page} 页恢复请求...", fg="yellow")
        except ValueError:
            pass

    new_records_batch = []
    reached_end = False

    while not reached_end:
        click.echo(f"正在抓取第 {current_page} 页...")
        params = {'limit': 100, 'page': current_page, 'login': USER_NAME, 'api_key': API_KEY}
        if current_upper_bound:
            params['search[updated_at]'] = f"..{current_upper_bound}"

        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                click.secho("[Fetch Wiki] 触发频率限制，休眠 3 分钟...", fg="yellow")
                time.sleep(180)
                continue
            elif resp.status_code == 403:
                click.secho("[Fetch Wiki] 收到 403 错误，凭证可能失效！", fg="red")
                break
            elif resp.status_code == 500:
                time.sleep(60)
                continue

            resp.raise_for_status()
            data = resp.json()

            if not data:
                click.secho("[Fetch Wiki] 已经翻到服务器最后一页。", fg="green")
                break

            for entry in data:
                entry_time = parser.parse(entry['updated_at'])
                if entry_time <= last_update_time:
                    click.secho(f"[Fetch Wiki] 遇到历史数据 ({entry_time})，与本地时间线成功衔接！", fg="green")
                    reached_end = True
                    break
                # 统一处理格式：other_names → JSON字符串，body → 清洗后文本
                new_records_batch.append(_normalize_entry(entry))

            if not reached_end:
                current_page += 1
                time.sleep(1.5 + random.random())

                # 突破 1000 页上限，重置时间轴
                if current_page > 900:
                    current_upper_bound = data[-1]['updated_at']
                    click.secho(f"[Fetch Wiki] 时间轴重置至: {current_upper_bound}", fg="magenta")
                    current_page = 1
                    if new_records_batch:
                        df_temp = pd.DataFrame(new_records_batch)
                        df_temp.to_csv(temp_csv_file, mode='a',
                                       header=not temp_csv_file.exists(), index=False,
                                       encoding='utf_8_sig')
                        new_records_batch.clear()
                    with open(progress_file, 'w') as f:
                        f.write(f"{current_page}\n{current_upper_bound}\n")
                    continue

                # 每 20 页保存一次检查点
                if (current_page - 1) % 20 == 0:
                    if new_records_batch:
                        df_temp = pd.DataFrame(new_records_batch)
                        df_temp.to_csv(temp_csv_file, mode='a',
                                       header=not temp_csv_file.exists(), index=False,
                                       encoding='utf_8_sig')
                        new_records_batch.clear()
                    with open(progress_file, 'w') as f:
                        f.write(f"{current_page - 1}\n")
                        if current_upper_bound:
                            f.write(f"{current_upper_bound}\n")
                    click.secho(f"[Fetch Wiki] 已抓取 {current_page - 1} 页。检查点已保存，强制休息...", fg="blue")
                    time.sleep(60)

        except requests.exceptions.RequestException as e:
            click.secho(f"[Fetch Wiki] 网络请求异常: {e}，休眠 60 秒后重试...", fg="red")
            time.sleep(60)
            continue

    # ── 收尾合并 ──
    click.secho("\n[Fetch Wiki] 抓取结束，开始整合数据...", fg="cyan")
    if new_records_batch:
        df_temp = pd.DataFrame(new_records_batch)
        df_temp.to_csv(temp_csv_file, mode='a',
                       header=not temp_csv_file.exists(), index=False,
                       encoding='utf_8_sig')

    if temp_csv_file.exists():
        df_new_all = pd.read_csv(temp_csv_file, low_memory=False)
        df_new_all['id'] = df_new_all['id'].astype(int)

        if not df_local.empty:
            # 对旧数据中的存量记录也补做格式统一
            # （仅在旧数据是旧格式时需要，新数据已经在抓取时处理过）
            if 'other_names' in df_local.columns:
                click.secho("[Fetch Wiki] 对旧数据存量记录补做格式统一...", fg="blue")
                df_local['other_names'] = df_local['other_names'].apply(
                    lambda v: _normalize_entry({'other_names': v, 'body': ''})['other_names']
                    if not (isinstance(v, str) and v.startswith('[') and '"' in v)
                    else v
                )
            df_local['id'] = df_local['id'].astype(int)
            df_final = pd.concat([df_local, df_new_all], ignore_index=True)
        else:
            df_final = df_new_all

        df_final = df_final.drop_duplicates(subset=['id'], keep='last').sort_values(by='id')
        df_final.to_parquet(parquet_path, index=False)
        click.secho(
            f"[Fetch Wiki] Wiki 数据合并成功！保存为 {parquet_path.name} (总行数: {len(df_final)})",
            fg="green"
        )

        os.remove(temp_csv_file)
        if progress_file.exists():
            os.remove(progress_file)
        click.secho("[Fetch Wiki] 断点记录已清理。", fg="green")
    else:
        click.secho("[Fetch Wiki] 本地数据已经是最新版，没有新数据产生。", fg="green")