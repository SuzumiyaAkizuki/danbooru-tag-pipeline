import click
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

from modules import sync_tags as mod_sync_tags
from modules import fetch_wiki as mod_fetch_wiki
from modules import llm_processor as mod_llm_processor
from modules import fetch_cooc as mod_fetch_cooc
from modules import trim_cooc as mod_trim_cooc
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


@click.group()
def cli():
    """🚀 Danbooru 标签工程自动化管理工具 (CLI)"""
    pass


@cli.command()
def sync_tags():
    """步骤 1: 根据 sqlite 更新标签基础信息"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> 开始同步 SQLite 标签数据...", fg="cyan")
    mod_sync_tags.run(config)
    click.secho("[Main] 同步完成！", fg="green")

@cli.command()
def fetch_tag_groups():
    """步骤 1.5: 获取标签组信息"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> 开始抓取标签组信息...", fg="cyan")
    mod_fetch_tag_groups.run(config)
    click.secho("[Main] 标签组信息抓取完成！", fg="green")

@cli.command()
def fetch_wiki():
    """步骤 2: 增量抓取 Danbooru Wiki"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> 开始抓取 Wiki...", fg="cyan")
    mod_fetch_wiki.run(config)


@cli.command()
@click.option('--preview', is_flag=True, help="开启预览模式，仅查看修改名单，不发请求")
@click.option('--debug', is_flag=True, help="开启调试模式，输出 wiki 清洗、other_names 匹配、LLM 收发的详细过程")
def llm_process(preview, debug):
    """步骤 3: 调用 LLM 与 Bangumi 翻译、重写、补齐标签"""
    config = load_config()
    ensure_directories(config)
    click.secho(f">>> 开始 LLM 处理流程 (预览模式: {preview}, 调试模式: {debug})...", fg="cyan")
    mod_llm_processor.run(config, preview=preview, debug=debug)


@cli.command()
@click.option('--full', is_flag=True, help="开启全量更新模式：无视历史记录，重新抓取所有标签的最新共现数据")
def fetch_cooc(full):
    """步骤 4: 抓取标签共现关系矩阵"""
    config = load_config()
    ensure_directories(config)
    mode_text = "全量模式" if full else "增量模式"
    click.secho(f">>> 开始抓取共现矩阵 ({mode_text})...", fg="cyan", bold=True)
    mod_fetch_cooc.run(config, full_update=full)


@cli.command()
@click.option('--top-k', type=int, default=None, help="覆盖 config.yaml 中的 top_k 设置")
@click.option('--min-pmi', type=float, default=None, help="覆盖 config.yaml 中的 min_pmi 设置")
@click.option('--dry-run', is_flag=True, help="开启测试模式：输出 PMI 为 1~5 时的截断保留量报告，不保存文件")
def trim_cooc(top_k, min_pmi, dry_run):
    """步骤 5: 共现矩阵 PMI 降维截断"""
    config = load_config()
    ensure_directories(config)
    click.secho(">>> 开始执行共现矩阵清洗与降维...", fg="cyan")
    mod_trim_cooc.run(config, top_k=top_k, min_pmi=min_pmi, dry_run=dry_run)


@cli.command()
def pipeline():
    """⭐ 一键全自动工作流: 自动顺序执行 1 到 5 步"""
    config = load_config()
    ensure_directories(config)
    click.secho("[Main] 启动一键全自动工作流 Pipeline...", fg="magenta", bold=True)
    ctx = click.get_current_context()
    ctx.invoke(sync_tags)
    ctx.invoke(fetch_wiki)
    ctx.invoke(llm_process, preview=False, debug=False)
    ctx.invoke(fetch_cooc)
    ctx.invoke(trim_cooc)
    click.secho("\n[Main] 所有 Pipeline 任务执行完毕！", fg="magenta", bold=True)


if __name__ == "__main__":
    cli()