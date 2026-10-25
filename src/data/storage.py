"""日线行情落盘与读取（Parquet 分区）。

目录约定::

    data/raw/daily/
        trade_date=20250102/
            data.parquet
        trade_date=20250105/
            data.parquet

设计要点：

- **只存 Tushare 原始单位**（不做任何换算）：``vol`` 为「手」、``amount`` 为「千元」，
  价格字段为不复权价。单位换算统一放在策略/回测读取层完成，避免同一份数据在不同
  模块里被重复换算两次（这类错误极难发现，且会让回测结论整体偏移）。
- **一个分区一个交易日**：``trade_date`` 即分区键，文件名固定 ``data.parquet``，
  不用 ``part-0.parquet`` 之类的随机名，方便 DuckDB 按目录通配、也方便人工核对。
- **重复拉取可覆盖**：同一 ``trade_date`` 重写即幂等覆盖，无需先删。
- ``trade_date`` 列统一存为 ``datetime64[ns]``，便于 DuckDB 与 pandas 直接比较。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

# 落盘根目录：src/data/storage.py → 上溯三级为项目根
DAILY_ROOT: Path = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily"
"""可通过修改此变量（或用 monkeypatch）重定向存储位置，测试时用临时目录。"""

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)
"""入库只保留这些列。多余列（如 pre_close / pct_chg）一律丢弃，保证 schema 稳定。"""

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "vol", "amount")


def _to_yyyymmdd(value: str | date | datetime | pd.Timestamp) -> str:
    """把多种日期表示归一成 ``YYYYMMDD`` 字符串（分区键格式）。"""
    if isinstance(value, str):
        digits = value.strip().replace("-", "").replace("/", "")
        if len(digits) != 8 or not digits.isdigit():
            raise ValueError(f"无法解析日期：{value!r}，期望 YYYYMMDD / YYYY-MM-DD")
        try:  # 长度对不代表日期合法，20251399 这种必须拦住
            datetime.strptime(digits, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"非法日期：{value!r}") from exc
        return digits
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y%m%d")
    raise TypeError(f"不支持的日期类型：{type(value)!r}")


def save_daily(df: pd.DataFrame, trade_date: str | date | datetime) -> Path | None:
    """把一个交易日的日线行情写入 ``data/raw/daily/trade_date=YYYYMMDD/data.parquet``。

    Args:
        df: 至少包含 :data:`REQUIRED_COLUMNS` 的 DataFrame；多余列会被丢弃。
        trade_date: 该批数据所属交易日，``YYYYMMDD`` 或 ``YYYY-MM-DD``。

    Returns:
        写入的文件路径；``df`` 为空时不落盘，返回 ``None``（休市日批量回补时很常见）。

    Raises:
        KeyError: 缺少必要字段。
        ValueError: 数据跨多个交易日（分区只能对应单日）、字段类型无法转换、或日期非法。
    """
    yyyymmdd = _to_yyyymmdd(trade_date)

    if df is None or df.empty:
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要字段：{missing}；当前列：{list(df.columns)}")

    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    # trade_date 统一成 datetime64[ns]，且必须与分区键一致
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="raise")
    unique_dates = out["trade_date"].dt.strftime("%Y%m%d").unique()
    if len(unique_dates) > 1:
        raise ValueError(f"数据跨越多个交易日 {sorted(unique_dates)}，无法写入单个分区")

    for col in _PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    out["ts_code"] = out["ts_code"].astype("string")

    # 同一 ts_code 重复出现时保留最后一条（重复拉取/修正数据的常见形态）
    out = out.drop_duplicates(subset=["ts_code"], keep="last")
    out = out.sort_values("ts_code").reset_index(drop=True)

    target_dir = DAILY_ROOT / f"trade_date={yyyymmdd}"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "data.parquet"
    out.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path


def load_daily(start_date: str | date | datetime, end_date: str | date | datetime) -> pd.DataFrame:
    """读取 ``[start_date, end_date]`` 闭区间内的日线数据。

    只读取落在区间内的分区目录，不会扫描全库；缺失的交易日自然跳过（休市日没有分区）。

    Returns:
        按 ``trade_date``、``ts_code`` 升序排列的 DataFrame。区间内无数据时返回
        **空 DataFrame 但保留完整列与 dtype**，调用方不必额外判空。

    Raises:
        ValueError: ``start_date > end_date``。
    """
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    if start > end:
        raise ValueError(f"起始日期 {start} 晚于结束日期 {end}")

    frames: list[pd.DataFrame] = []
    if DAILY_ROOT.is_dir():
        for part in sorted(DAILY_ROOT.glob("trade_date=*")):
            key = part.name.split("=", 1)[1]
            # 分区名是定长 YYYYMMDD，字符串比较等价于日期比较
            if not (start <= key <= end):
                continue
            parquet = part / "data.parquet"
            if not parquet.is_file():
                continue
            frames.append(pd.read_parquet(parquet, engine="pyarrow", columns=list(REQUIRED_COLUMNS)))

    if not frames:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in _PRICE_COLUMNS})
        empty.insert(0, "trade_date", pd.Series(dtype="datetime64[ns]"))
        empty.insert(0, "ts_code", pd.Series(dtype="string"))
        return empty

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
