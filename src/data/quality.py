"""日线落地数据的质量检查。

用 DuckDB 直接扫 Parquet 分区，产出「按交易日」的异常计数表，用于在回测前
确认落盘数据没有重复行、非法价格、负成交量。

用法::

    from data.quality import check_daily
    print(check_daily("20261019", "20261023"))

或命令行::

    python -m data.quality 20261019 20261023

返回的 DataFrame 每个交易日一行，列为：

===================  ==========================================================
``trade_date``       交易日
``total_rows``       当日总行数（正常应等于当日标的总数）
``duplicate_codes``  同一 ``ts_code`` 出现多次的多余行数（0 才算干净）
``null_close``       ``close`` 为 NULL **或 ≤ 0** 的行数（0 才算干净）
``negative_vol``     ``vol`` 为负的行数（0 才算干净）
===================  ==========================================================

注意：区间内没有任何分区时返回**空表**并给出 warning。空表 ≠ 数据干净，
它通常意味着「忘了先跑 fetch_daily.py」，所以不以静默通过的方式放过。
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from . import storage

#: 结果列与 dtype，空结果时也按这套 schema 返回，调用方不必判空
_RESULT_SCHEMA: dict[str, str] = {
    "trade_date": "datetime64[ns]",
    "total_rows": "int64",
    "duplicate_codes": "int64",
    "null_close": "int64",
    "negative_vol": "int64",
}


def _to_yyyymmdd(value: str | date | datetime | pd.Timestamp) -> str:
    """把 YYYYMMDD / YYYY-MM-DD / date 统一成 YYYYMMDD 字符串。"""
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


def _glob_pattern(root: Path) -> str:
    """拼出给 DuckDB 的通配符路径，并把反斜杠统一成正斜杠（Windows 下更稳）。"""
    pattern = (root / "trade_date=*" / "data.parquet").as_posix()
    return pattern.replace("'", "''")


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _RESULT_SCHEMA.items()})


def check_daily(
    start_date: str | date | datetime | pd.Timestamp,
    end_date: str | date | datetime | pd.Timestamp,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """检查 ``[start_date, end_date]`` 闭区间内每个交易日的数据质量。

    Args:
        start_date: 起始日期，YYYYMMDD 或 YYYY-MM-DD。
        end_date: 结束日期，含当日。
        root: 数据根目录，默认 ``data/raw/daily``（便于测试指向临时目录）。

    Returns:
        按 ``trade_date`` 升序的 DataFrame，列见模块 docstring。
        区间内无分区时返回同 schema 的空表。

    Raises:
        ValueError: 日期格式非法，或 start_date 晚于 end_date。
    """
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    if start > end:
        raise ValueError(f"start_date ({start}) 不能晚于 end_date ({end})")

    base = Path(root) if root is not None else storage.DAILY_ROOT
    if not base.exists():
        warnings.warn(f"数据目录不存在：{base}", RuntimeWarning, stacklevel=2)
        return _empty_result()

    pattern = _glob_pattern(base)

    sql = f"""
        WITH src AS (
            SELECT
                CAST(trade_date AS DATE) AS trade_date,
                ts_code,
                close,
                vol
            FROM read_parquet('{pattern}', hive_partitioning = false)
            WHERE CAST(trade_date AS DATE) BETWEEN DATE '{start[:4]}-{start[4:6]}-{start[6:]}'
                                                AND DATE '{end[:4]}-{end[4:6]}-{end[6:]}'
        )
        SELECT
            trade_date,
            CAST(COUNT(*) AS BIGINT)                                AS total_rows,
            CAST(COUNT(*) - COUNT(DISTINCT ts_code) AS BIGINT)      AS duplicate_codes,
            CAST(SUM(CASE WHEN close IS NULL OR close <= 0
                          THEN 1 ELSE 0 END) AS BIGINT)             AS null_close,
            CAST(SUM(CASE WHEN vol < 0
                          THEN 1 ELSE 0 END) AS BIGINT)             AS negative_vol
        FROM src
        GROUP BY trade_date
        ORDER BY trade_date
    """

    # duckdb.query 是模块级快捷方式，内部使用默认连接；用完即关不涉及持久状态。
    # 这里用 connect() 包裹以显式管理连接，语义与 duckdb.query 一致。
    with duckdb.connect() as con:
        try:
            df = con.execute(sql).df()
        except duckdb.IOException as exc:  # 通配符匹配不到任何文件
            warnings.warn(
                f"区间 {start}~{end} 没有匹配到任何分区文件（{pattern}）：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _empty_result()

    if df.empty:
        warnings.warn(
            f"区间 {start}~{end} 内没有数据。空表不代表数据干净，"
            f"请先运行 scripts/fetch_daily.py 拉取该区间。",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_result()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ("total_rows", "duplicate_codes", "null_close", "negative_vol"):
        df[col] = df[col].astype("int64")

    return df.loc[:, list(_RESULT_SCHEMA)].reset_index(drop=True)


def is_clean(df: pd.DataFrame) -> bool:
    """三条异常计数是否全为 0（空表返回 False，避免「空即干净」的误判）。"""
    if df.empty:
        return False
    return bool(
        (df["duplicate_codes"] == 0).all()
        and (df["null_close"] == 0).all()
        and (df["negative_vol"] == 0).all()
    )


if __name__ == "__main__":  # pragma: no cover - 手工检查用
    import sys

    if len(sys.argv) != 3:
        print("用法：python -m data.quality YYYYMMDD YYYYMMDD", file=sys.stderr)
        raise SystemExit(2)

    report = check_daily(sys.argv[1], sys.argv[2])
    print(report.to_string(index=False))
    print(f"\n共 {len(report)} 个交易日 | 是否干净：{is_clean(report)}")
