import pandas as pd
import sqlite3
import os
import requests
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
    raise ValueError(f"[Sync Tags] 无法读取文件（编码未知或文件损坏）：{path}")


def download_sqlite(save_path="tag.sqlite"):
    """
    从 GitHub 下载 tag.sqlite 文件

    Args:
        save_path: 保存路径，默认为当前目录下的 tag.sqlite
    """
    url = "https://github.com/ffdkj/ffdkj-Danbooru_Tag-Chinese-English-Translation-Table/raw/main/tag.sqlite"

    print(f"[Sync Tags] 正在下载: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = downloaded / total_size * 100
                    print(f"\r[Sync Tags] 下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")

    print(f"\n[Sync Tags] 下载完成！文件保存至: {os.path.abspath(save_path)}")
    return save_path


def run(config):
    """执行标签同步逻辑"""
    # 提取配置路径并转为绝对路径

    base_dir = Path(__file__).resolve().parent.parent
    sqlite_path = base_dir / config['paths']['raw']['sqlite_db']
    csv_path = base_dir / config['paths']['processed']['tags_enhanced']
    download_sqlite(save_path=sqlite_path)
    click.secho(f"[Sync Tags] 读取目标 CSV: {csv_path}", fg="blue")
    df_old = read_csv_robust(csv_path)
    existing_names = set(df_old['name'].tolist())

    click.secho(f"[Sync Tags] 读取源 SQLite: {sqlite_path}", fg="blue")
    if not sqlite_path.exists():
        click.secho(f"[Sync Tags] 找不到 SQLite 数据库文件: {sqlite_path}", fg="red")
        return

    try:
        conn = sqlite3.connect(sqlite_path)
        df_sqlite = pd.read_sql_query("SELECT name, category, cn_name, post_count FROM tags", conn)
        conn.close()
    except Exception as e:
        click.secho(f"[Sync Tags] 读取 SQLite 失败: {e}", fg="red")
        return

    # 格式化 SQLite 数据
    df_sqlite['post_count'] = pd.to_numeric(df_sqlite['post_count'], errors='coerce').fillna(0).astype(int)
    df_sqlite['category'] = pd.to_numeric(df_sqlite['category'], errors='coerce').fillna(-1).astype(int)
    df_sqlite['cn_name'] = df_sqlite['cn_name'].fillna("").astype(str)

    # 提取新增标签
    mask_new = (df_sqlite['post_count'] >= 100) & (df_sqlite['category'].isin([0, 3, 4])) & (
        ~df_sqlite['name'].isin(existing_names))
    df_new = df_sqlite[mask_new].copy()

    if not df_new.empty:
        df_new['wiki'] = ""
        df_new['nsfw'] = "0"
        click.secho(f"[Sync Tags] 发现 {len(df_new)} 个符合条件的新增标签！", fg="green")
    else:
        click.secho("[Sync Tags] 没有发现符合条件的新增标签。", fg="yellow")

    # 合并并更新所有标签的分类和发帖量
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    sqlite_map = df_sqlite.set_index('name')

    # 更新 post_count 和 category
    df_all['post_count'] = df_all['name'].map(sqlite_map['post_count']).fillna(
        pd.to_numeric(df_all['post_count'], errors='coerce').fillna(0)).astype(int)

    df_all['category'] = df_all['name'].map(sqlite_map['category']).fillna(
        pd.to_numeric(df_all['category'], errors='coerce').fillna(-1)).astype(int)

    # 保存更新后的状态
    df_all.to_csv(csv_path, index=False, encoding='utf-8')
    click.secho(f"[Sync Tags] 标签库同步完毕，当前总计 {len(df_all)} 条记录。", fg="green")