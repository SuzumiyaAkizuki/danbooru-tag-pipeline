"""
统一回收站模块：所有文件/目录删除操作先移入 data/.trash/，避免误删无法恢复。
"""

import shutil
from pathlib import Path
from datetime import datetime


def _trash_root(base_dir: Path) -> Path:
    return base_dir / "data" / ".trash"


def trash_file(base_dir: Path, path: Path) -> None:
    """将文件移入回收站。若文件不存在则静默跳过。"""
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_dir = _trash_root(base_dir) / ts
    trash_dir.mkdir(parents=True, exist_ok=True)
    dest = trash_dir / path.name
    counter = 0
    while dest.exists():
        counter += 1
        dest = trash_dir / f"{path.stem}_{counter}{path.suffix}"
    shutil.move(str(path), str(dest))


def trash_dir(base_dir: Path, path: Path) -> None:
    """将整个目录移入回收站。若目录不存在则静默跳过。"""
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_dir = _trash_root(base_dir) / ts / path.name
    trash_dir.mkdir(parents=True, exist_ok=True)
    # 将目录内的所有内容移入回收站子目录
    shutil.move(str(path), str(trash_dir))
