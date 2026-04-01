import math
import time
import pandas as pd
from pathlib import Path
import click
import numpy as np


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件: {path}")


def run(config, top_k=None, min_pmi=None, dry_run=False):
    base_dir = Path(__file__).resolve().parent.parent

    cooc_path = base_dir / config['paths']['raw']['cooc_raw_csv']
    tags_path = base_dir / config['paths']['processed']['tags_enhanced']
    out_path = base_dir / config['paths']['processed']['cooc_clean']

    actual_top_k = top_k if top_k is not None else config['settings']['cooc_trim']['top_k']
    actual_min_pmi = min_pmi if min_pmi is not None else float(config['settings']['cooc_trim']['min_pmi'])

    if not cooc_path.exists() or not tags_path.exists():
        click.secho("[Trim Cooc] 找不到源文件", fg="red")
        return

    t0 = time.time()
    tags_df = read_csv_robust(tags_path)
    tags_df["post_count"] = pd.to_numeric(tags_df["post_count"], errors="coerce").fillna(0)
    freq = dict(zip(tags_df["name"], tags_df["post_count"]))
    D = float(tags_df["post_count"].max())

    df = read_csv_robust(cooc_path)

    # 兼容性拦截：检查是否为旧版无向矩阵
    if 'source' not in df.columns or 'frequency' not in df.columns:
        click.secho("[Trim Cooc] 检测到旧版格式的共现矩阵", fg="red", bold=True)
        click.secho("请先运行: python main.py fetch-cooc --full 重建矩阵", fg="yellow")
        return

    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0.0)
    original_len = len(df)
    click.echo(f"  原始有向边数: {original_len:,} (总文档估算 D: {D:,.0f})")

    count_target = df["target"].map(freq)
    count_source = df["source"].map(freq)

    valid = count_target.notna() & count_source.notna() & (count_target > 0) & (count_source > 0) & (
                df["frequency"] > 0)
    df = df[valid].copy()

    # 计算不受脱期影响的 PMI
    pmi_values = (df["frequency"] * D / df["target"].map(freq))
    df["pmi"] = pmi_values.apply(lambda x: math.log2(x) if x > 0 else -100)

    # 基于当前最新发帖量和历史稳态频率，实时估算出当前的绝对共现次数
    df["count"] = (df["frequency"] * df["source"].map(freq)).round().astype(int)

    if dry_run:
        click.secho(f"\n[Trim Cooc] 开启 Dry-Run 模式 (Top-K = {actual_top_k})", fg="magenta", bold=True)
        for threshold in range(1, 6):
            th = float(threshold)
            df_filtered = df[df["pmi"] >= th].copy()
            after_pmi_len = len(df_filtered)

            if after_pmi_len > 0:
                df_filtered.sort_values(["source", "pmi", "count"], ascending=[True, False, False], inplace=True)
                top_k_df = df_filtered.groupby("source", sort=False).head(actual_top_k).copy()

                s_vals = top_k_df["source"].to_numpy()
                t_vals = top_k_df["target"].to_numpy()
                mask = s_vals < t_vals
                top_k_df["tag_a"] = np.where(mask, s_vals, t_vals)
                top_k_df["tag_b"] = np.where(mask, t_vals, s_vals)
                final_kept = len(top_k_df.groupby(["tag_a", "tag_b"], as_index=False).first())
            else:
                final_kept = 0

            click.echo(f"  >= {th:<7} | {after_pmi_len:<20,} | {final_kept:<20,}")
        return

    df = df[df["pmi"] >= actual_min_pmi]
    if len(df) == 0:
        click.secho("[Trim Cooc] 过滤后无数据", fg="yellow")
        return

    # 降序并按源节点截断 Top-K
    df.sort_values(["source", "pmi", "count"], ascending=[True, False, False], inplace=True)
    top_k_df = df.groupby("source", sort=False).head(actual_top_k).copy()

    # 有向边折叠回无向边规范化
    s_vals = top_k_df["source"].to_numpy()
    t_vals = top_k_df["target"].to_numpy()
    mask = s_vals < t_vals
    top_k_df["tag_a"] = np.where(mask, s_vals, t_vals)
    top_k_df["tag_b"] = np.where(mask, t_vals, s_vals)

    result = (top_k_df.groupby(["tag_a", "tag_b"], as_index=False)
              .agg({"count": "max", "pmi": "max"})
              .reset_index(drop=True))

    result.sort_values("pmi", ascending=False, inplace=True)
    result = result[["tag_a", "tag_b", "count"]]
    result.to_parquet(out_path, index=False, compression='snappy')

    click.secho("[Trim Cooc] 降维清洗成功", fg="green")