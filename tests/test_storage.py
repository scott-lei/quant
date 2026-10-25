"""storage.save_daily / load_daily 的行为约束。"""

from __future__ import annotations

import pandas as pd
import pytest

from data import storage


def _make_df(trade_date: str, codes: tuple[str, ...] = ("000001.SZ", "600519.SH")) -> pd.DataFrame:
    """造一份带「多余列」的假数据，用于验证字段裁剪。"""
    return pd.DataFrame(
        {
            "ts_code": list(codes),
            "trade_date": [trade_date] * len(codes),
            "open": [10.0, 1500.0][: len(codes)],
            "high": [10.5, 1520.0][: len(codes)],
            "low": [9.8, 1480.0][: len(codes)],
            "close": [10.2, 1488.0][: len(codes)],
            "vol": [123456.7, 50028.7][: len(codes)],      # 单位：手（Tushare 原始）
            "amount": [1234.5, 7490883.7][: len(codes)],   # 单位：千元（Tushare 原始）
            # 下面两列应被丢弃
            "pre_close": [9.9, 1490.0][: len(codes)],
            "pct_chg": [3.03, -0.13][: len(codes)],
        }
    )


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """每个用例都在临时目录里跑，绝不碰项目真实的 data/raw。"""
    monkeypatch.setattr(storage, "DAILY_ROOT", tmp_path / "daily")


def test_save_writes_expected_partition_path():
    path = storage.save_daily(_make_df("20250102"), "20250102")
    assert path == storage.DAILY_ROOT / "trade_date=20250102" / "data.parquet"
    assert path.is_file()


def test_save_keeps_only_required_columns():
    """只保留必要字段，pre_close / pct_chg 这类多余列必须被裁掉。"""
    path = storage.save_daily(_make_df("20250102"), "20250102")
    saved = pd.read_parquet(path)
    assert list(saved.columns) == list(storage.REQUIRED_COLUMNS)
    assert "pre_close" not in saved.columns
    assert "pct_chg" not in saved.columns


def test_save_preserves_tushare_raw_units():
    """存储层不做单位换算：vol 仍是「手」、amount 仍是「千元」。"""
    path = storage.save_daily(_make_df("20250102"), "20250102")
    saved = pd.read_parquet(path).set_index("ts_code")
    assert saved.loc["600519.SH", "vol"] == pytest.approx(50028.7)
    assert saved.loc["600519.SH", "amount"] == pytest.approx(7490883.7)


def test_save_is_idempotent_and_overwrites():
    """同一交易日重复拉取：覆盖而非追加，行数不翻倍。"""
    storage.save_daily(_make_df("20250102"), "20250102")
    storage.save_daily(_make_df("20250102"), "20250102")
    got = storage.load_daily("20250102", "20250102")
    assert len(got) == 2


def test_save_empty_df_writes_nothing():
    assert storage.save_daily(pd.DataFrame(), "20250102") is None
    assert not (storage.DAILY_ROOT / "trade_date=20250102").exists()


def test_save_missing_column_raises():
    df = _make_df("20250102").drop(columns=["amount"])
    with pytest.raises(KeyError, match="amount"):
        storage.save_daily(df, "20250102")


def test_save_multi_date_raises():
    df = pd.concat([_make_df("20250102"), _make_df("20250103")], ignore_index=True)
    with pytest.raises(ValueError, match="多个交易日"):
        storage.save_daily(df, "20250102")


def test_save_rejects_bad_date():
    with pytest.raises(ValueError):
        storage.save_daily(_make_df("20250102"), "2025-13-99")


def test_load_filters_by_date_range():
    """区间过滤：写入 3 天，只取中间 2 天，且按日期升序。"""
    for d in ("20250102", "20250103", "20250106"):
        storage.save_daily(_make_df(d), d)

    got = storage.load_daily("20250102", "20250103")
    assert got["trade_date"].dt.strftime("%Y%m%d").unique().tolist() == ["20250102", "20250103"]
    assert len(got) == 4  # 2 天 × 2 只

    assert len(storage.load_daily("20250103", "20250103")) == 2
    assert len(storage.load_daily("20250101", "20250131")) == 6  # 闭区间，含首尾


def test_load_accepts_hyphenated_dates():
    storage.save_daily(_make_df("20250102"), "20250102")
    got = storage.load_daily("2025-01-01", "2025-01-31")
    assert len(got) == 2


def test_load_missing_range_returns_empty_with_schema():
    """区间内没有任何分区时，返回带完整 schema 的空表，调用方无需判空。"""
    storage.save_daily(_make_df("20250102"), "20250102")
    got = storage.load_daily("20250601", "20250630")
    assert got.empty
    assert list(got.columns) == list(storage.REQUIRED_COLUMNS)
    assert str(got["trade_date"].dtype) == "datetime64[ns]"


def test_load_reversed_range_raises():
    with pytest.raises(ValueError, match="晚于"):
        storage.load_daily("20250110", "20250102")
