"""策略读取层：按「截止日」取单只标的的历史 bar。

这个模块是 **as-of 规则在代码层面的落点**（见 ``docs/data-dictionary.md`` §4）。
所有策略取数都应经过这里，而不是自己拼日期条件——否则未来函数很容易悄悄混进来。

典型用法::

    bars = get_bars_before("000001.SZ", "20250102", lookback=100)  # 含 2025-01-02
    prev = get_bars_before("000001.SZ", "20250102", include_as_of=False)  # 不含当日

**关于是否包含当日（重要）**

默认 ``include_as_of=True``，即取 ``trade_date <= as_of_date``。这个选择与
「日线 bar 在收盘后才可用」的事实放在一起看，就有一个必须由调用方承担的语义：

- 日线 bar 的 ``available_at`` 是**当日 15:00**（见数据字典 §4.2）。
- 因此若策略在 ``T`` 日**盘中**做决策（例如 14:30 生成信号），传入 ``as_of_date = T``
  就会把 ``T`` 日**尚未收完的收盘价**读进来——这是典型的未来函数。
- 若策略在 ``T`` 日**盘后**做决策（收盘后跑批、用于次日开盘下单），传入
  ``as_of_date = T`` 才是正确的。

判断口诀：**决策发生在当日收盘前，就传前一交易日；发生在收盘后，才传当日。**
拿不准时用 ``include_as_of=False``，它等价于「只看到昨天为止」的保守口径。

单位：默认返回**规范层**（``volume`` 单位股、``amount`` 单位元），与
``docs/data-dictionary.md`` §1.4 一致；需要 Tushare 原始单位时传 ``canonical=False``。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from . import storage

#: 返回列（规范层），空结果时也按这套 schema 返回
_CANONICAL_SCHEMA: dict[str, str] = {
    "ts_code": "string",
    "trade_date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
}

#: 原始层 schema（canonical=False 时使用），列序与 storage.REQUIRED_COLUMNS 一致
_RAW_SCHEMA: dict[str, str] = {
    "ts_code": "string",
    "trade_date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "vol": "float64",
    "amount": "float64",
}


def _to_yyyymmdd(value: str | date | datetime | pd.Timestamp) -> str:
    """YYYYMMDD / YYYY-MM-DD / date 统一成 YYYYMMDD 字符串。"""
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, str):
        digits = value.strip().replace("-", "").replace("/", "")
        if len(digits) == 8 and digits.isdigit():
            try:
                datetime.strptime(digits, "%Y%m%d")
            except ValueError as exc:
                raise ValueError(f"非法日期：{value!r}") from exc
            return digits
    raise ValueError(f"无法解析日期：{value!r}，期望 YYYYMMDD / YYYY-MM-DD")


def _partition_files(upper: str, root: Path) -> list[str]:
    """列出 ``trade_date <= upper`` 且存在 data.parquet 的分区文件，升序。

    在 Python 侧先按目录名筛一遍，避免让 DuckDB 打开用不上的分区元数据
    （数据字典 §6：显式文件列表比通配符少读文件）。
    """
    if not root.exists():
        return []
    files: list[tuple[str, str]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("trade_date="):
            continue
        token = child.name.split("=", 1)[1]
        if len(token) != 8 or not token.isdigit() or token > upper:
            continue
        parquet = child / "data.parquet"
        if parquet.is_file():
            files.append((token, parquet.as_posix().replace("'", "''")))
    return [path for _, path in sorted(files)]


def _empty_like(canonical: bool) -> pd.DataFrame:
    schema = _CANONICAL_SCHEMA if canonical else _RAW_SCHEMA
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in schema.items()}).loc[
        :, list(schema)
    ]


def get_bars_before(
    symbol: str,
    as_of_date: str | date | datetime | pd.Timestamp,
    lookback: int = 100,
    *,
    include_as_of: bool = True,
    canonical: bool = True,
    root: Path | None = None,
) -> pd.DataFrame:
    """取 ``symbol`` 在 ``as_of_date`` 及之前最近 ``lookback`` 根日线 bar。

    过滤条件默认为 ``trade_date <= as_of_date``（含当日）。**是否该含当日取决于
    决策时点**：盘中决策必须传前一交易日或用 ``include_as_of=False``，详见模块
    docstring 与数据字典 §4。

    Args:
        symbol: 标的代码，如 ``000001.SZ``（大小写与首尾空格会被归一）。
        as_of_date: 截止日（含），YYYYMMDD 或 YYYY-MM-DD。
        lookback: 期望返回的最大行数；数据不足时返回**全部可用数据**，不报错。
        include_as_of: ``True`` → ``<=``；``False`` → ``<``（严格早于当日）。
        canonical: ``True`` → 规范层单位（volume=股、amount=元）；``False`` → 原始层。
        root: 数据根目录，默认 ``data/raw/daily``（测试可指向临时目录）。

    Returns:
        按 ``trade_date`` **升序**排列的 DataFrame，最多 ``lookback`` 行。
        无数据时返回同 schema 的空表（不返回 None，便于调用方直接 rolling）。

    Raises:
        ValueError: ``symbol`` 为空、日期非法、或 ``lookback`` 不是正整数。
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol 不能为空")
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback <= 0:
        raise ValueError(f"lookback 必须是正整数，收到：{lookback!r}")

    code = symbol.strip().upper()
    as_of = _to_yyyymmdd(as_of_date)

    # include_as_of=True → 分区目录名 <= as_of；False → 严格 <
    # 未来数据有两层拦截，缺一不可：
    #   第一层：这里按目录名裁剪分区，晚于 as_of 的分区根本不会被打开；
    #   第二层：下面 SQL 里再做一次日期比较，防止将来有人改回通配符读取时漏掉。
    # 变异测试结论（tests/test_accessor.py）：只删掉第二层，单日用例仍会通过
    # （第一层挡住了）；两层都删才会泄漏未来行，届时 test_future_day_is_not_returned 会失败。
    upper = as_of
    files = _partition_files(upper, Path(root) if root is not None else storage.DAILY_ROOT)
    if not files:
        return _empty_like(canonical)

    cmp = "<=" if include_as_of else "<"
    sql = f"""
        SELECT *
        FROM read_parquet({files!r}, hive_partitioning = false)
        WHERE ts_code = ?
          AND CAST(trade_date AS DATE) {cmp} DATE '{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}'
        ORDER BY trade_date DESC
        LIMIT ?
    """

    with duckdb.connect() as con:
        df = con.execute(sql, [code, int(lookback)]).df()

    if df.empty:
        return _empty_like(canonical)

    # 上面按 DESC 取「最近 N 根」，这里翻回升序；索引重置成 0..n-1
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    if canonical:
        df = storage.to_canonical(df)
    return df.loc[:, list(_CANONICAL_SCHEMA if canonical else _RAW_SCHEMA)]
