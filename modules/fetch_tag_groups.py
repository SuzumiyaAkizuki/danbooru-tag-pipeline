import re
import json
import requests
import time
import os
from pathlib import Path
import click


def fetch_wiki_page(title: str, headers: dict, auth: dict) -> dict | None:
    url = "https://danbooru.donmai.us/wiki_pages.json"
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params={**auth, 'search[title]': title, 'limit': 1},
                headers=headers,
                timeout=30
            )
            if resp.status_code == 429:
                click.secho("[Tag Groups] 429 限流，休眠 3 分钟...", fg="yellow")
                time.sleep(180)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data[0] if data else None
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(5)
            else:
                click.secho(f"[Tag Groups] 请求失败 ({title}): {e}", fg="yellow")
    return None


def parse_group_titles(index_body: str) -> list[str]:
    """从 tag_groups 主目录页 body 解析所有子 group 的 wiki title。"""
    titles = []
    # body 中格式为 [[Tag group:Xxx yyy]] 或 [[Tag Group:Xxx]]，大小写不固定
    for m in re.finditer(r'\[\[Tag [Gg]roup:([^\]|]+?)(?:\|[^\]]*)?\]\]', index_body, re.IGNORECASE):
        raw = m.group(1).strip()
        # 转为小写下划线形式，与 API search[title] 匹配
        title = "tag_group:" + raw.lower().replace(' ', '_')
        titles.append(title)
    return list(dict.fromkeys(titles))  # 保序去重


def parse_group_members(body: str) -> list[str]:
    """从 tag group 页 body 中提取成员标签名。"""
    tags = []
    for m in re.finditer(r'\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]', body):
        raw = m.group(1).strip()
        normalized = raw.lower().replace(' ', '_')
        # 排除导航性 wiki 页面引用
        if any(normalized.startswith(p) for p in (
            'tag_group:', 'list_of_', 'help:', 'pool_', 'tag_groups'
        )):
            continue
        tags.append(normalized)
    return list(dict.fromkeys(tags))


def run(config):
    base_dir = Path(__file__).resolve().parent.parent

    USER_NAME = os.getenv("DANBOORU_USER_NAME")
    API_KEY = os.getenv("DANBOORU_API_KEY")
    if not USER_NAME or not API_KEY:
        click.secho("[Tag Groups] 未配置凭证", fg="red")
        return

    out_path = base_dir / config['paths']['processed']['tag_groups']
    headers = {"User-Agent": f"TagGroupBot/1.0 (by {USER_NAME})", "Accept": "application/json"}
    auth = {'login': USER_NAME, 'api_key': API_KEY}

    # Step 1: 抓取主目录页，解析所有子 group title
    click.echo("[Tag Groups] 正在抓取主目录页 tag_groups ...")
    index_page = fetch_wiki_page('tag_groups', headers, auth)
    if not index_page:
        click.secho("[Tag Groups] 无法获取主目录页", fg="red")
        return

    group_titles = parse_group_titles(index_page.get('body', ''))
    click.echo(f"[Tag Groups] 目录页解析到 {len(group_titles)} 个 group")
    if not group_titles:
        click.secho("[Tag Groups] 未解析到任何 group，终止", fg="red")
        return

    # Step 2: 逐个精确查询每个 tag group 页，提取成员标签
    tag_to_groups: dict[str, list[str]] = {}
    group_to_tags: dict[str, list[str]] = {}

    for i, title in enumerate(group_titles):
        click.echo(f"  [{i + 1}/{len(group_titles)}] {title}")
        page = fetch_wiki_page(title, headers, auth)
        if not page:
            click.secho(f"  跳过（页面不存在或请求失败）", fg="yellow")
            time.sleep(1.0)
            continue

        members = parse_group_members(page.get('body', ''))
        group_to_tags[title] = members

        for tag in members:
            tag_to_groups.setdefault(tag, []).append(title)

        time.sleep(1.5)

    result = {
        'tag_to_groups': tag_to_groups,
        'group_to_tags': group_to_tags,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    click.secho(
        f"[Tag Groups] 完成：{len(group_to_tags)} 个 group，覆盖 {len(tag_to_groups)} 个标签",
        fg="green"
    )