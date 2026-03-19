import math
import time
import pandas as pd
from pathlib import Path
import click


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

    # 优先使用命令行传入的参数，如果没有传，则使用 config.yaml 里的默认值
    actual_top_k = top_k if top_k is not None else config['settings']['cooc_trim']['top_k']
    actual_min_pmi = min_pmi if min_pmi is not None else float(config['settings']['cooc_trim']['min_pmi'])

    if not cooc_path.exists() or not tags_path.exists():
        click.secho("[Trim Cooc] 找不到源文件 (原始共现矩阵或标签文件)，请先执行前置抓取步骤！", fg="red")
        return

    t0 = time.time()
    click.secho(f"[Trim Cooc] 正在读取标签频数表与共现表... (当前设置: Top-K={actual_top_k}, Min-PMI={actual_min_pmi})", fg="blue")

    # 1. 读取标签发帖量 (用于估算总文档数 D)
    tags_df = read_csv_robust(tags_path)
    tags_df["post_count"] = pd.to_numeric(tags_df["post_count"], errors="coerce").fillna(0)
    freq = dict(zip(tags_df["name"], tags_df["post_count"]))
    D = float(tags_df["post_count"].max())  # 总文档数

    # 2. 读取共现表
    df = read_csv_robust(cooc_path)
    # 兼容历史版本的列名 (把 raw_count 视作 count)
    if 'raw_count' in df.columns and 'count' not in df.columns:
        df = df.rename(columns={'raw_count': 'count'})

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    original_len = len(df)
    click.echo(f"  原始行数: {original_len:,} (总发帖量估算 D: {D:,.0f})")

    # 3. 计算所有基础组合的 PMI
    click.echo("[Trim Cooc] 正在计算全量 PMI ...")
    count_a = df["tag_a"].map(freq)
    count_b = df["tag_b"].map(freq)
    cooc = df["count"].astype(float)

    valid = count_a.notna() & count_b.notna() & (count_a > 0) & (count_b > 0) & (cooc > 0)
    df = df[valid].copy()

    df["pmi"] = (df["count"].astype(float) * D / (df["tag_a"].map(freq) * df["tag_b"].map(freq))).apply(math.log2)

    # ==========================================
    # 🌟 核心新增：Dry-Run 测试模式
    # ==========================================
    if dry_run:
        click.secho(f"\n[Trim Cooc] 开启 Dry-Run 模式，开始模拟截断测试 (设定 Top-K = {actual_top_k})", fg="magenta", bold=True)
        click.secho("-" * 65, fg="magenta")
        click.secho(f"{'PMI 阈值':<10} | {'PMI过滤后剩余行数':<20} | {'Top-K截断后最终行数':<20}", fg="cyan")
        click.secho("-" * 65, fg="magenta")

        # 遍历 1 到 5 的阈值
        for threshold in range(1, 6):
            th = float(threshold)
            df_filtered = df[df["pmi"] >= th].copy()
            after_pmi_len = len(df_filtered)

            if after_pmi_len > 0:
                # 模拟 Top-K 截断
                fwd = df_filtered[["tag_a", "tag_b", "count", "pmi"]].rename(columns={"tag_a": "src", "tag_b": "dst"})
                rev = df_filtered[["tag_b", "tag_a", "count", "pmi"]].rename(columns={"tag_b": "src", "tag_a": "dst"})
                both = pd.concat([fwd, rev], ignore_index=True)

                both.sort_values(["src", "pmi", "count"], ascending=[True, False, False], inplace=True)
                both = both.groupby("src", sort=False).head(actual_top_k)

                both["tag_a"] = both[["src", "dst"]].min(axis=1)
                both["tag_b"] = both[["src", "dst"]].max(axis=1)

                result_sim = both.groupby(["tag_a", "tag_b"], as_index=False).first()
                final_kept = len(result_sim)
            else:
                final_kept = 0

            click.echo(f"  >= {th:<7} | {after_pmi_len:<20,} | {final_kept:<20,}")

        click.secho("-" * 65, fg="magenta")
        click.secho(f"[Trim Cooc] 测试完毕，耗时 {time.time() - t0:.2f} 秒。未向磁盘写入任何文件。", fg="green")
        return

    # ==========================================
    # 正常执行模式
    # ==========================================
    df = df[df["pmi"] >= actual_min_pmi]
    after_pmi = len(df)
    click.echo(f"  PMI >= {actual_min_pmi} 过滤后剩余: {after_pmi:,} 行")

    if after_pmi == 0:
        click.secho("[Trim Cooc] 过滤后无数据，请检查设定的 PMI 阈值！", fg="yellow")
        return

    # 4. 双向展开，降序截断 Top-K
    click.echo(f"[Trim Cooc] 正在执行 Top-{actual_top_k} 截断 ...")
    fwd = df[["tag_a", "tag_b", "count", "pmi"]].rename(columns={"tag_a": "src", "tag_b": "dst"})
    rev = df[["tag_b", "tag_a", "count", "pmi"]].rename(columns={"tag_b": "src", "tag_a": "dst"})
    both = pd.concat([fwd, rev], ignore_index=True)

    both.sort_values(["src", "pmi", "count"], ascending=[True, False, False], inplace=True)
    both = both.groupby("src", sort=False).head(actual_top_k)

    # 5. 统一字典序并去重
    both["tag_a"] = both[["src", "dst"]].min(axis=1)
    both["tag_b"] = both[["src", "dst"]].max(axis=1)

    result = (both.groupby(["tag_a", "tag_b"], as_index=False)
              .agg({"count": "first", "pmi": "max"})
              .reset_index(drop=True))

    result.sort_values("pmi", ascending=False, inplace=True)
    result = result[["tag_a", "tag_b", "count"]]  # 最终抛弃多余列，保持极简
    result.reset_index(drop=True, inplace=True)

    kept = len(result)

    # 6. 保存为 Parquet
    result.to_parquet(out_path, index=False, compression='snappy')
    size_mb = out_path.stat().st_size / 1024 / 1024

    click.secho(f"[Trim Cooc] 降维清洗成功！", fg="green")
    click.secho(f"[Trim Cooc] 最终保留 {kept:,} 行 ({kept / original_len * 100:.1f}%)，耗时 {time.time() - t0:.2f} 秒。")
    click.secho(f"[Trim Cooc] 成品已保存至: {out_path.name} ({size_mb:.1f} MB)", fg="green")