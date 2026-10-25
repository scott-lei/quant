"""Tushare Pro 客户端封装。

约定（与 .gitignore、pyproject.toml 配套）：

- token 只从环境变量 ``TUSHARE_TOKEN`` 读取，**不硬编码、不写进仓库**。
- 若项目根目录存在 ``.env``，首次读取时自动加载；已存在的环境变量优先，不会被覆盖。
- 同一进程内复用同一个 pro 对象，避免重复鉴权。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ENV_TOKEN = "TUSHARE_TOKEN"
"""存放 token 的环境变量名，唯一来源。"""


class MissingTokenError(RuntimeError):
    """未配置 TUSHARE_TOKEN 时抛出，避免带着空 token 发出无意义的请求。"""


def _project_root() -> Path:
    """src/data/tushare_client.py → 上溯三级即项目根。"""
    return Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """极简 ``.env`` 加载器，避免为一行读取引入 python-dotenv。

    只支持 ``KEY=VALUE`` 行，忽略空行与 ``#`` 注释；已存在于 ``os.environ``
    的键不覆盖，保证真实环境变量（如 CI 注入）优先级更高。
    """
    dotenv = path or _project_root() / ".env"
    if not dotenv.is_file():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_token() -> str:
    """读取并校验 token。缺失时立即报错，而不是让 tushare 返回权限错误。"""
    load_dotenv()
    token = os.environ.get(ENV_TOKEN, "").strip()
    if not token:
        raise MissingTokenError(
            f"未找到环境变量 {ENV_TOKEN}。"
            f"请在项目根目录的 .env 中设置 {ENV_TOKEN}=<你的 token>，"
            f"或先 export {ENV_TOKEN}=... 再运行。"
        )
    return token


@lru_cache(maxsize=1)
def pro_api():
    """返回已鉴权的 tushare pro 对象（进程内单例）。

    token 变更后需重启进程；这是刻意为之——token 在运行期变化属于配置事故。
    """
    token = get_token()  # 先校验 token，缺 token 时不必关心 tushare 是否安装

    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "未安装 tushare，请先执行：python -m pip install -e ."
        ) from exc

    return ts.pro_api(token)
