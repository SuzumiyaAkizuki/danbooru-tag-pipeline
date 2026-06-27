"""
modules/trim_artist_cooc.py
───────────────────────────
步骤 5.5: 标签-画师共现表 NPMI 降维截断。

对 tag_artist_cooc.parquet 中的每条边计算 NPMI（归一化点互信息），
按画师分组，每位画师仅保留 NPMI 最高的 top_k 个标签，
同时过滤掉 NPMI 低于 min_npmi 阈值的噪声关联。

输出: data/processed/tag_artist_cooc.parquet（覆盖原文件）
  列: tag, artist, artist_post_count, cooc_count, frequency, pmi, npmi

依赖:
  - data/processed/tag_artist_cooc.parquet（fetch_artist_cooc 产出）
  - data/processed/tags_enhanced.csv（tag post_count）
"""

import math
import time
import pandas as pd
import numpy as np
from pathlib import Path
import click


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"[Trim Artist Cooc] 无法读取文件: {path}")


def run(config, top_k=None, min_npmi=None, dry_run=False):
    base_dir = Path(__file__).resolve().parent.parent

    # ── 路径 ───────────────────────────────────────────────────────────
    cooc_path  = base_dir / config['paths']['processed']['tag_artist_cooc']
    tags_path  = base_dir / config['paths']['processed']['tags_enhanced']

    # 默认值可被 CLI 参数覆盖
    actual_top_k = top_k if top_k is not None else config['settings']['artist_cooc_trim']['top_k']
    actual_min_npmi = min_npmi if min_npmi is not None else float(
        config['settings']['artist_cooc_trim']['min_npmi']
    )

    if not cooc_path.exists() or not tags_path.exists():
        click.secho("[Trim Artist Cooc] 找不到源文件", fg="red")
        return

    t0 = time.time()

    # ── 加载标签 post_count ────────────────────────────────────────────
    tags_df = read_csv_robust(tags_path)
    tags_df["post_count"] = pd.to_numeric(tags_df["post_count"], errors="coerce").fillna(0)
    tag_post_count: dict[str, float] = dict(
        zip(tags_df["name"], tags_df["post_count"])
    )
    D = float(tags_df["post_count"].max())  # 语料库大小估算

    # ── 加载画师共现表 ─────────────────────────────────────────────────
    df = pd.read_parquet(str(cooc_path))

    required_cols = {"tag", "artist", "artist_post_count", "cooc_count", "frequency"}
    missing = required_cols - set(df.columns)
    if missing:
        click.secho(
            f"[Trim Artist Cooc] 列缺失: {missing}，请先运行 fetch-artist-cooc",
            fg="red",
        )
        return

    # 清除上一次运行可能残留下的 NaN 列，确保完全重算
    for stale_col in ["pmi", "npmi"]:
        if stale_col in df.columns:
            df = df.drop(columns=[stale_col])
            click.secho(f"[Trim Artist Cooc] 已清除旧版 '{stale_col}' 列，将重新计算。", fg="yellow")

    # 强制类型转换，防止从 parquet 读入非数值类型（如 object/string）
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df["cooc_count"] = pd.to_numeric(df["cooc_count"], errors="coerce")
    df["artist_post_count"] = pd.to_numeric(df["artist_post_count"], errors="coerce")

    original_len = len(df)
    n_artists_orig = df["artist"].nunique()
    n_tags_orig = df["tag"].nunique()
    click.echo(
        f"  原始边数: {original_len:,}  "
        f"(D={D:,.0f}, artists={n_artists_orig}, tags={n_tags_orig})"
    )

    # ── 映射 tag post_count ────────────────────────────────────────────
    tag_pc = df["tag"].map(tag_post_count)
    df["tag_post_count"] = tag_pc

    # 过滤：tag 不在 tags_enhanced 中，或 post_count <= 0
    valid = tag_pc.notna() & (tag_pc > 0) & (df["frequency"] > 0) & (df["cooc_count"] > 0)
    df = df[valid].copy()

    # ── 计算 PMI ───────────────────────────────────────────────────────
    # PMI = log2( cooc * D / (artist_post_count * tag_post_count) )
    #      = log2( frequency * D / tag_post_count )
    # 其中 frequency = cooc_count / artist_post_count
    pmi_ratio = (df["frequency"].to_numpy() * D) / df["tag_post_count"].to_numpy()
    df["pmi"] = np.where(
        pmi_ratio > 0,
        np.log2(pmi_ratio),
        -100.0,
    )

    # ── 计算 NPMI ──────────────────────────────────────────────────────
    # NPMI = PMI / -log2( cooc / D ) = PMI / -log2(P(tag, artist))
    p_ab = df["cooc_count"].to_numpy() / D
    denom = np.where(p_ab > 0, -np.log2(p_ab), 1.0)
    denom = np.where(denom > 0, denom, 1.0)  # 防止除以 0
    df["npmi"] = df["pmi"].to_numpy() / denom

    # NPMI 裁剪到 [−1, 1]（理论上限，浮点误差可能导致略微越界）
    df["npmi"] = df["npmi"].clip(-1.0, 1.0)

    # 数据完整性检查：计算过程中不应产生 NaN
    nan_npmi = df["npmi"].isna().sum()
    nan_pmi = df["pmi"].isna().sum()
    if nan_pmi > 0 or nan_npmi > 0:
        click.secho(
            f"[Trim Artist Cooc] 警告：PMI NaN={nan_pmi}, NPMI NaN={nan_npmi}，"
            f"数据可能存在类型问题，建议重新运行 fetch-artist-cooc。",
            fg="yellow",
        )
        # 丢弃 NaN 行，防止污染输出
        df = df[df["pmi"].notna() & df["npmi"].notna()].copy()
        click.secho(
            f"[Trim Artist Cooc] 已丢弃 NaN 行，剩余 {len(df):,} 条。",
            fg="yellow",
        )

    # ── Dry-run: 不同阈值下的保留量报告 ────────────────────────────────
    if dry_run:
        click.secho(
            f"\n[Trim Artist Cooc] Dry-Run (Top-K = {actual_top_k})",
            fg="magenta", bold=True,
        )
        click.echo(f"  {'NPMI >= ':<10} {'过滤后边数':<12} {'Top-K截断后':<14} {'覆盖画家':<10}")
        click.echo(f"  {'─' * 10} {'─' * 12} {'─' * 14} {'─' * 10}")
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            df_filtered = df[df["npmi"] >= threshold].copy()
            after_pmi = len(df_filtered)

            if after_pmi > 0:
                df_filtered.sort_values(
                    ["artist", "npmi", "cooc_count"],
                    ascending=[True, False, False],
                    inplace=True,
                )
                top_k_df = df_filtered.groupby("artist", sort=False).head(actual_top_k)
                final_kept = len(top_k_df)
                n_art = top_k_df["artist"].nunique()
            else:
                final_kept = 0
                n_art = 0

            click.echo(f"  >= {threshold:<8} {after_pmi:<12,} {final_kept:<14,} {n_art:<10,}")
        return

    # ── 正式裁剪 ───────────────────────────────────────────────────────
    before_pmi = len(df)
    df = df[df["npmi"] >= actual_min_npmi].copy()

    if len(df) == 0:
        click.secho("[Trim Artist Cooc] NPMI 过滤后无数据，检查 min_npmi 阈值", fg="yellow")
        return

    click.echo(f"  NPMI >= {actual_min_npmi}: {before_pmi:,} → {len(df):,} 条边")

    # 按画师降序排列，每画师保留 top_k
    df.sort_values(
        ["artist", "npmi", "cooc_count"],
        ascending=[True, False, False],
        inplace=True,
    )
    df = df.groupby("artist", sort=False).head(actual_top_k).copy()

    # ── 输出 ───────────────────────────────────────────────────────────
    out_cols = [
        "tag", "artist", "artist_post_count", "cooc_count",
        "frequency", "pmi", "npmi",
    ]
    df_out = df[out_cols].reset_index(drop=True)

    # 按 tag 排序，方便引擎做 tag→artist 倒排查询
    df_out.sort_values(["tag", "npmi"], ascending=[True, False], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    df_out.to_parquet(str(cooc_path), index=False, compression="snappy")

    n_edges = len(df_out)
    n_art = df_out["artist"].nunique()
    n_tag = df_out["tag"].nunique()
    elapsed = time.time() - t0

    click.secho(
        f"[Trim Artist Cooc] 完成: {n_edges:,} 条边, "
        f"{n_art} artists, {n_tag} tags, 耗时 {elapsed:.1f}s",
        fg="green",
    )

    # ── 数据质量展示 ───────────────────────────────────────────────────
    click.echo("\n  ── 每画师平均标签数 ──")
    per_artist = df_out.groupby("artist").size()
    click.echo(f"    mean={per_artist.mean():.1f}  median={per_artist.median():.1f}  "
               f"min={per_artist.min()}  max={per_artist.max()}")

    click.echo("\n  ── NPMI 分布 ──")
    npmi_vals = df_out["npmi"]
    click.echo(f"    mean={npmi_vals.mean():.4f}  median={npmi_vals.median():.4f}  "
               f"p10={npmi_vals.quantile(0.10):.4f}  p90={npmi_vals.quantile(0.90):.4f}")
