"""
modules/fetch_artist_cooc.py
────────────────────────────
步骤 6: 抓取标签-画师共现关系。

从 tag.sqlite 中筛选 post_count > 100 的画师，
调用 /related_tag.json 获取每位画师最常画的标签及频率，
翻转构建 tag→artist 倒排索引，供搜索引擎推荐"常画某题材的画师"。

输出: data/processed/tag_artist_cooc.parquet
  列: tag, artist, artist_post_count, cooc_count, frequency

依赖:
  - data/raw/tag.sqlite（画师名单，category=1, post_count>100）
  - data/processed/tags_enhanced.csv（标签白名单，用于过滤）
  - data/processed/artist_quality_top1000.json（可选，用于优先排序）

"""

import requests
import pandas as pd
import sqlite3
import time
import json
import os
from pathlib import Path
import click
from modules.trash import trash_file, trash_dir


def read_csv_robust(path: Path) -> pd.DataFrame:
    """多编码尝试读取 CSV，失败返回空 DataFrame。"""
    if not path.exists():
        return pd.DataFrame()
    for enc in ['utf-8', 'gbk', 'gb18030']:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"[Artist Cooc] 无法读取文件: {path}")


# ─── 单次 API 请求 ───────────────────────────────────────────────────────

def _fetch_related(session: requests.Session, tag_name: str, max_retries: int = 5):
    """请求 /related_tag.json，返回 JSON 或空列表（失败时）。"""
    url = "https://danbooru.donmai.us/related_tag.json"
    params = {'query': tag_name}

    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                click.secho(f"  [429] 限流，休眠 180s...", fg="yellow")
                time.sleep(180)
                continue
            elif resp.status_code == 403:
                click.secho("  [403] 凭证失效，终止。", fg="red")
                return None
            elif resp.status_code >= 500:
                click.secho(f"  [{resp.status_code}] 服务器错误，休眠 30s...", fg="yellow")
                time.sleep(30)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            click.secho(f"  请求异常 (attempt {attempt+1}): {e}", fg="red")
            if attempt < max_retries - 1:
                time.sleep(10)

    click.secho("  达到最大重试次数。", fg="red")
    return []


# ─── 解析 related_tag 响应 ───────────────────────────────────────────────

def _parse_related_for_artist(
    data: dict | list,
    artist_name: str,
    valid_tags_set: set[str],
    artist_post_count: int,
) -> list[dict]:
    """
    从 /related_tag.json 响应中提取 (tag, artist) 边。

    仅保留 target 标签在 valid_tags_set 中的行，
    cooc_count 由 frequency × artist_post_count 推算。
    优先使用 API 返回的 post_count（比离线 JSON 更新）。
    """
    api_post_count = data.get("post_count", 0) if isinstance(data, dict) else 0
    effective_pc = api_post_count if api_post_count > 0 else artist_post_count

    if isinstance(data, dict) and "related_tags" in data:
        items = data["related_tags"]
    elif isinstance(data, list):
        items = data
    else:
        return []

    edges = []
    for item in items:
        if isinstance(item, dict):
            tag_b = item.get("tag", {}).get("name") if isinstance(item.get("tag"), dict) else item.get("name")
            freq = float(item.get("frequency", 0.0))
        elif isinstance(item, list) and len(item) >= 2:
            tag_b = str(item[0])
            freq = 0.0
        else:
            continue

        if not tag_b or tag_b == artist_name or tag_b not in valid_tags_set:
            continue
        if freq <= 0.0:
            continue

        cooc_count = round(freq * effective_pc)

        edges.append({
            "tag": tag_b,
            "artist": artist_name,
            "artist_post_count": effective_pc,
            "cooc_count": cooc_count,
            "frequency": round(freq, 6),
        })

    return edges


# ─── 主逻辑 ──────────────────────────────────────────────────────────────

def run(config, full_update: bool = False, max_artists: int | None = None,
        min_post_count: int = 100):
    """
    抓取所有 curated 画师的关联标签，构建 tag→artist 共现表。

    Args:
        config:           config.yaml 解析结果
        full_update:      True=忽略断点和历史，全量重抓
        max_artists:      限制抓取画师数量（调试用，None=全部）
        min_post_count:   画师最低发帖量阈值，默认 200
    """
    base_dir = Path(__file__).resolve().parent.parent

    USER_NAME = os.getenv("DANBOORU_USER_NAME")
    API_KEY   = os.getenv("DANBOORU_API_KEY")
    if not USER_NAME or not API_KEY:
        click.secho("[Artist Cooc] 未配置 DANBOORU_USER_NAME / API_KEY", fg="red")
        return

    # ── 路径 ───────────────────────────────────────────────────────────
    sqlite_db     = base_dir / config['paths']['raw']['sqlite_db']
    tags_csv      = base_dir / config['paths']['processed']['tags_enhanced']
    output_parquet = base_dir / config['paths']['processed']['tag_artist_cooc']
    checkpoint_dir = base_dir / "data" / "checkpoint"
    history_file   = checkpoint_dir / "artist_cooc_history.json"
    temp_dir       = checkpoint_dir / "_artist_cooc_temp"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    # ── 从 SQLite 加载画师（category=1, post_count>100）────────────
    if not sqlite_db.exists():
        click.secho(f"[Artist Cooc] 找不到 {sqlite_db}", fg="red")
        return

    conn = sqlite3.connect(str(sqlite_db))
    cursor = conn.execute(
        "SELECT name, post_count FROM tags "
        "WHERE category=1 AND post_count > ? "
        "ORDER BY post_count DESC",
        (min_post_count,)
    )
    artist_rows = cursor.fetchall()
    conn.close()

    all_artist_names = [row[0] for row in artist_rows]
    artist_post_counts = {row[0]: row[1] for row in artist_rows}

    click.echo(f"[Artist Cooc] SQLite 中 post_count>{min_post_count} 的画师: {len(all_artist_names):,} 位")

    # ── 裁剪（调试用）────────────────────────────────────────────────
    if max_artists is not None and max_artists < len(all_artist_names):
        click.secho(f"[Artist Cooc] 调试模式：仅处理前 {max_artists} 位画师", fg="yellow")
        all_artist_names = all_artist_names[:max_artists]
        # 重建 post_count dict（只保留裁剪后的）
        artist_post_counts = {k: artist_post_counts[k] for k in all_artist_names}

    click.echo(f"[Artist Cooc] 本次候选: {len(all_artist_names)} 位画师")

    # ── 加载标签白名单 ─────────────────────────────────────────────────
    df_tags = read_csv_robust(tags_csv)
    valid_tags_set = set(df_tags['name'].dropna().unique())
    click.echo(f"[Artist Cooc] 标签白名单: {len(valid_tags_set)} 个")

    # ── 恢复进度 ───────────────────────────────────────────────────────
    completed_set: set[str] = set()

    if full_update:
        if temp_dir.exists() and list(temp_dir.glob("chunk_*.parquet")):
            # 上次全量模式中断，断点续传
            click.secho("[Artist Cooc] 全量模式 — 从断点续传...", fg="yellow", bold=True)
            if history_file.exists():
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        completed_set = set(json.load(f))
                    click.echo(f"[Artist Cooc] 已恢复 {len(completed_set):,} 位已完成画师")
                except Exception:
                    pass
            # 从 temp chunks 恢复中断时已抓取但未合并的数据
            chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
            if chunk_files:
                dfs_chunks = [pd.read_parquet(str(c)) for c in chunk_files]
                df_tmp = pd.concat(dfs_chunks, ignore_index=True)
                chunk_artists = set(df_tmp["artist"].unique())
                completed_set |= chunk_artists
                click.echo(
                    f"[Artist Cooc] 从 {len(chunk_files)} 个临时 chunk 恢复 "
                    f"{len(chunk_artists):,} 位画师"
                )
        else:
            # 首次启动全量模式，清空旧状态从头开始
            click.secho("[Artist Cooc] 全量模式（首次启动）", fg="red", bold=True)
            if history_file.exists():
                trash_file(base_dir, history_file)
            if temp_dir.exists():
                trash_dir(base_dir, temp_dir)
    else:
        # 1) 从历史文件恢复已完成画师
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    completed_set = set(json.load(f))
                click.echo(f"[Artist Cooc] 历史已完成: {len(completed_set):,} 位")
            except Exception:
                pass

        # 2) 从临时 chunk 恢复中断时已抓但未合并的数据
        if temp_dir.exists():
            chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
            if chunk_files:
                dfs_chunks = [pd.read_parquet(str(c)) for c in chunk_files]
                df_tmp = pd.concat(dfs_chunks, ignore_index=True)
                chunk_artists = set(df_tmp["artist"].unique())
                completed_set |= chunk_artists
                click.echo(
                    f"[Artist Cooc] 从 {len(chunk_files)} 个临时 chunk 恢复 "
                    f"{len(chunk_artists):,} 位画师"
                )

    # ── 计算待处理画师 ─────────────────────────────────────────────────
    remaining = [a for a in all_artist_names if a not in completed_set]
    total_all = len(all_artist_names)
    total_remaining = len(remaining)

    if total_remaining == 0:
        click.secho("[Artist Cooc] 所有画师已处理完毕", fg="green")
        return
    else:
        click.secho(
            f"[Artist Cooc] 待处理: {total_remaining:,} / {total_all:,} 位画师 "
            f"({len(completed_set):,} 已完成)",
            fg="cyan",
        )

    # ── 初始化 session ─────────────────────────────────────────────────
    session = requests.Session()
    session.headers.update({
        "User-Agent": f"ArtistCoocBot/1.0 (by {USER_NAME})",
        "Accept": "application/json",
    })
    session.params = {"login": USER_NAME, "api_key": API_KEY}  # type: ignore[attr-defined]

    # ── 逐画师抓取 ─────────────────────────────────────────────────────
    temp_dir.mkdir(parents=True, exist_ok=True)

    batch: list[dict] = []
    processed_this_run: set[str] = set()

    for i, artist_name in enumerate(remaining):
        n_done = len(completed_set) + len(processed_this_run)
        artist_pc = artist_post_counts.get(artist_name, 0)
        click.echo(
            f"  [{n_done + 1}/{total_all}] {artist_name} "
            f"(post_count={artist_pc})",
            nl=False,
        )

        data = _fetch_related(session, artist_name)
        if data is None:
            click.secho(" ABORT", fg="red")
            break
        if not data:
            click.secho(" -> 无关联标签")
            processed_this_run.add(artist_name)
            continue

        edges = _parse_related_for_artist(data, artist_name, valid_tags_set, artist_pc)
        if edges:
            batch.extend(edges)
            click.secho(f" -> {len(edges)} 条边")
        else:
            click.secho(" -> 0 条有效边")

        processed_this_run.add(artist_name)

        # 每 50 位画师保存一次进度 + 写临时文件
        if len(processed_this_run) % 50 == 0 or (i + 1) == total_remaining:
            if batch:
                df_temp = pd.DataFrame(batch)
                chunk_file = temp_dir / f"chunk_{n_done + len(processed_this_run):05d}.parquet"
                df_temp.to_parquet(chunk_file, index=False)
                click.echo(f"  [checkpoint] 写入 {len(batch)} 条边 → {chunk_file.name}")
                batch.clear()

            completed_set.update(processed_this_run)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(sorted(completed_set), f, ensure_ascii=False)
            processed_this_run.clear()

            # 每 100 位额外休息 30s 避免触限
            if len(completed_set) % 100 == 0:
                click.secho("  [cooldown] 休息 30s...", fg="yellow")
                time.sleep(30)

        time.sleep(1.5)

    # ── 合并 & 输出 ────────────────────────────────────────────────────
    _merge_and_save(temp_dir, output_parquet, full_update)

    # 清理临时文件（保留 history_file 供下次增量使用）
    if temp_dir.exists():
        trash_dir(base_dir, temp_dir)

    click.secho("[Artist Cooc] 完成", fg="green", bold=True)


def _merge_and_save(
    temp_dir: Path,
    output_path: Path,
    full_update: bool,
) -> None:
    """合并所有 chunk 文件并与已有 parquet 合并（同名对取 max cooc_count），输出最终 parquet。"""
    chunks = sorted(temp_dir.glob("chunk_*.parquet"))
    if not chunks:
        click.secho("[Artist Cooc] 无新数据", fg="yellow")
        return

    click.echo(f"[Artist Cooc] 合并 {len(chunks)} 个 chunk...")

    dfs = [pd.read_parquet(str(c)) for c in chunks]
    df_new = pd.concat(dfs, ignore_index=True)

    # 同一 (tag, artist) 对取最大 cooc_count
    df_new = (
        df_new
        .groupby(["tag", "artist"], as_index=False)
        .agg({
            "artist_post_count": "max",
            "cooc_count": "max",
            "frequency": "max",
        })
    )

    # 画师最低共现阈值：至少 3 张帖子同时包含该 tag 和该画师
    before = len(df_new)
    df_new = df_new[df_new["cooc_count"] >= 3].copy()
    if before > len(df_new):
        click.echo(f"  [filter] cooc_count >= 3: {before} → {len(df_new)} 条边")

    # 按 tag 排序，同一 tag 内按 cooc_count 降序
    df_new.sort_values(
        ["tag", "cooc_count"], ascending=[True, False], inplace=True
    )
    df_new.reset_index(drop=True, inplace=True)

    if output_path.exists():
        df_old = pd.read_parquet(str(output_path))
        # 清除旧数据中可能残留的 pmi/npmi 列，避免新旧 schema 不一致
        for c in ["pmi", "npmi"]:
            if c in df_old.columns:
                df_old = df_old.drop(columns=[c])
        # 新的覆盖旧的同名对，旧的无冲突行保留
        old_keys = set(zip(df_old["tag"], df_old["artist"]))
        new_keys = set(zip(df_new["tag"], df_new["artist"]))
        keep_mask = [k not in new_keys for k in old_keys]
        df_merged = pd.concat(
            [df_old[keep_mask], df_new], ignore_index=True
        )
        df_final = df_merged
    else:
        df_final = df_new

    # 最终去重：合并后可能残留跨来源的同名对，取 max cooc_count
    df_final = (
        df_final
        .groupby(["tag", "artist"], as_index=False)
        .agg({
            "artist_post_count": "max",
            "cooc_count": "max",
            "frequency": "max",
        })
    )
    df_final.sort_values(["tag", "cooc_count"], ascending=[True, False], inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    df_final.to_parquet(str(output_path), index=False, compression="snappy")

    # 统计
    n_edges = len(df_final)
    n_tags = df_final["tag"].nunique()
    n_artists = df_final["artist"].nunique()
    click.secho(
        f"[Artist Cooc] 输出 {n_edges:,} 条边, "
        f"{n_tags} tags, {n_artists} artists → {output_path}",
        fg="green",
    )

    # 展示 top-10 画师最多的 tag（数据质量检查）
    tag_artist_count = (
        df_final.groupby("tag").size().sort_values(ascending=False).head(10)
    )
    click.echo("\n  ── Top-10 画师最多的标签 ──")
    for tag, cnt in tag_artist_count.items():
        click.echo(f"    {tag}: {cnt} 位画师")
