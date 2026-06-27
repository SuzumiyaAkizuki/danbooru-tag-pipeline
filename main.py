import click
import yaml
import os
import zipfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

from modules import sync_tags as mod_sync_tags
from modules import fetch_wiki as mod_fetch_wiki
from modules import llm_processor as mod_llm_processor
from modules import fetch_cooc as mod_fetch_cooc
from modules import trim_cooc as mod_trim_cooc
from modules import fetch_artist_cooc as mod_fetch_artist_cooc
from modules import trim_artist_cooc as mod_trim_artist_cooc
from modules import fetch_tag_groups as mod_fetch_tag_groups


def load_config():
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        click.secho("[Main] 找不到 config.yaml 配置文件！", fg="red")
        raise click.Abort()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config):
    base_dir = Path(__file__).resolve().parent
    for category in ['raw', 'checkpoint', 'processed']:
        dir_path = base_dir / "data" / category
        dir_path.mkdir(parents=True, exist_ok=True)


def backup_data():
    """将 data/ 文件夹打包备份为 data_backup_YYYYMMDD_HHMMSS.zip"""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    if not data_dir.exists():
        click.secho("[Backup] data/ 目录不存在，跳过备份。", fg="yellow")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = base_dir / f"data_backup_{timestamp}.zip"

    click.secho(f"[Backup] 正在备份 data/ → {backup_path.name} ...", fg="cyan")
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(base_dir)
                zf.write(file_path, arcname)
    size_mb = backup_path.stat().st_size / 1024 / 1024
    click.secho(f"[Backup] 备份完成 ({size_mb:.1f} MB)", fg="green")


class OrderedGroup(click.Group):
    """按定义顺序显示命令的 Group"""
    def list_commands(self, ctx):
        return list(self.commands.keys())


@click.group(cls=OrderedGroup)
def cli():
    """Danbooru 标签工程自动化管理工具 (CLI)"""
    pass


@cli.command()
def sync_tags():
    """步骤 1: 根据 sqlite 更新标签基础信息"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> [Step 1] 开始同步 SQLite 标签数据...", fg="cyan")
    mod_sync_tags.run(config)
    click.secho("[Main] 同步完成！", fg="green")


@cli.command()
def fetch_tag_groups():
    """步骤 2: 获取标签组信息"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> [Step 2] 开始抓取标签组信息...", fg="cyan")
    mod_fetch_tag_groups.run(config)
    click.secho("[Main] 标签组信息抓取完成！", fg="green")


@cli.command()
def fetch_wiki():
    """步骤 3: 增量抓取 Danbooru Wiki"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> [Step 3] 开始抓取 Wiki...", fg="cyan")
    mod_fetch_wiki.run(config)


@cli.command()
@click.option('--preview', is_flag=True, help="开启预览模式，仅查看修改名单，不发请求")
@click.option('--debug', is_flag=True, help="开启调试模式，输出 wiki 清洗、other_names 匹配、LLM 收发的详细过程")
@click.option('--reprocess-wiki-updates', is_flag=True, help="重新处理 Wiki 有更新的标签（即使已处理过），默认仅处理新增标签")
def llm_process(preview, debug, reprocess_wiki_updates):
    """步骤 4: 调用 LLM 与 Bangumi 翻译、重写、补齐标签"""
    config = load_config()
    ensure_directories(config)
    click.secho(f">>> [Step 4] 开始 LLM 处理流程 (预览: {preview}, 调试: {debug}, Wiki更新重处理: {reprocess_wiki_updates})...", fg="cyan")
    mod_llm_processor.run(config, preview=preview, debug=debug, reprocess_wiki_updates=reprocess_wiki_updates)


@cli.command()
@click.option('--full', is_flag=True, help="开启全量更新模式：无视历史记录，重新抓取所有标签的最新共现数据")
def fetch_cooc(full):
    """步骤 5: 抓取标签共现关系矩阵（同时顺带抓取标签-画师共现）"""
    config = load_config()
    ensure_directories(config)
    mode_text = "全量模式" if full else "增量模式"
    click.secho(f">>> [Step 5] 开始抓取共现矩阵 + 画师共现 ({mode_text})...", fg="cyan", bold=True)
    mod_fetch_cooc.run(config, full_update=full)


@cli.command()
@click.option('--full', is_flag=True, help="全量模式：忽略断点和历史，从头重抓所有画师")
@click.option('--max-artists', type=int, default=None, help="仅处理前 N 位画师（调试用）")
@click.option('--min-post-count', type=int, default=100, help="画师最低发帖量阈值（默认 100）")
def fetch_artist_cooc(full, max_artists, min_post_count):
    """步骤 6: 按画师逐一抓取标签-画师共现关系"""
    config = load_config()
    ensure_directories(config)
    mode_text = "全量模式" if full else "增量模式"
    click.secho(f">>> [Step 6] 开始抓取标签-画师共现 ({mode_text}, post_count>{min_post_count})...", fg="cyan", bold=True)
    mod_fetch_artist_cooc.run(config, full_update=full, max_artists=max_artists,
                              min_post_count=min_post_count)


@cli.command()
@click.option('--top-k', type=int, default=None, help="覆盖 config.yaml 中的 top_k 设置")
@click.option('--min-pmi', type=float, default=None, help="覆盖 config.yaml 中的 min_pmi 设置")
@click.option('--dry-run', is_flag=True, help="开启测试模式：输出 PMI 为 1~5 时的截断保留量报告，不保存文件")
def trim_cooc(top_k, min_pmi, dry_run):
    """步骤 7: 共现矩阵 PMI 降维截断"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> [Step 7] 开始执行共现矩阵清洗与降维...", fg="cyan")
    mod_trim_cooc.run(config, top_k=top_k, min_pmi=min_pmi, dry_run=dry_run)


@cli.command()
@click.option('--top-k', type=int, default=None, help="每位画师保留的标签数上限（覆盖 config）")
@click.option('--min-npmi', type=float, default=None, help="NPMI 最低阈值（覆盖 config）")
@click.option('--dry-run', is_flag=True, help="测试模式：输出不同阈值下的保留量，不保存")
def trim_artist_cooc(top_k, min_npmi, dry_run):
    """步骤 8: 标签-画师共现表 NPMI 降维截断"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> [Step 8] 开始执行标签-画师共现 NPMI 裁剪...", fg="cyan")
    mod_trim_artist_cooc.run(config, top_k=top_k, min_npmi=min_npmi, dry_run=dry_run)


@cli.command()
def pipeline():
    """一键全自动工作流: 自动备份 data/ 后顺序执行 Step 1-8"""
    config = load_config()
    ensure_directories(config)

    click.secho("=" * 60, fg="magenta", bold=True)
    click.secho("[Pipeline] 启动一键全自动工作流", fg="magenta", bold=True)
    click.secho("=" * 60, fg="magenta", bold=True)

    backup_data()

    ctx = click.get_current_context()
    steps = [
        ("Step 1/8", sync_tags),
        ("Step 2/8", fetch_tag_groups),
        ("Step 3/8", fetch_wiki),
        ("Step 4/8", llm_process),
        ("Step 5/8", fetch_cooc),
        ("Step 6/8", fetch_artist_cooc),
        ("Step 7/8", trim_cooc),
        ("Step 8/8", trim_artist_cooc),
    ]

    for label, cmd in steps:
        click.secho(f"\n{'─' * 40}", fg="magenta")
        click.secho(f"[Pipeline] 执行 {label}", fg="magenta", bold=True)
        click.secho(f"{'─' * 40}", fg="magenta")
        # 对带 option 的命令传入默认值
        if cmd == llm_process:
            ctx.invoke(cmd, preview=False, debug=False, reprocess_wiki_updates=False)
        elif cmd == fetch_cooc:
            ctx.invoke(cmd, full=False)
        elif cmd == fetch_artist_cooc:
            ctx.invoke(cmd, full=False, max_artists=None, min_post_count=100)
        elif cmd == trim_cooc:
            ctx.invoke(cmd, top_k=None, min_pmi=None, dry_run=False)
        elif cmd == trim_artist_cooc:
            ctx.invoke(cmd, top_k=None, min_npmi=None, dry_run=False)
        else:
            ctx.invoke(cmd)

    click.secho("\n" + "=" * 60, fg="magenta", bold=True)
    click.secho("[Pipeline] 所有任务执行完毕！", fg="magenta", bold=True)
    click.secho("=" * 60, fg="magenta", bold=True)


if __name__ == "__main__":
    cli()
