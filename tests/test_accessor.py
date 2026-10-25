"""get_bars_before 的 as-of 行为约束。

核心一条：**决策日之后的数据一根都不能进来**。为了让这条断言不是「空跑」，
每个用例都先确认构造的数据里真的有「未来那一天」，再断言它没被返回——
否则数据没写进去，断言也会通过，等于什么都没测。
"""

from __future__ import annotations

import pandas as pd
import pytest

from data import accessor, storage


def _bars(trade_date: str, code: str = "000001.SZ", close: float = 10.0) -> pd.DataFrame:
    """造一根原始层（Tushare 单位）日线，字段与 storage.REQUIRED_COLUMNS 对齐。"""
    return pd.DataFrame(
        {
            "ts_code": [code],
            "trade_date": [trade_date],
            "open": [close - 0.1],
            "high": [close + 0.2],
            "low": [close - 0.2],
            "close": [close],
            "vol": [1000.0],       # 手
            "amount": [1000.0],    # 千元
        }
    )


def _save(*frames: pd.DataFrame) -> None:
    for df in frames:
        storage.save_daily(df, df["trade_date"].iloc[0])


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """每个用例在临时目录里跑，绝不碰项目真实的 data/raw。"""
    monkeypatch.setattr(storage, "DAILY_ROOT", tmp_path / "daily")


def test_future_day_is_not_returned():
    """as_of=20250101 时，20250102 的行不得出现在结果里。

    双重保险：先断言 20250102 这份分区**确实已经落盘**（否则本用例会空跑通过），
    再断言查询结果既不含它、也不是空表。
    """
    _save(_bars("20250101", close=10.0), _bars("20250102", close=11.0))

    # 前提校验：未来那天真的在库里
    future_partition = storage.DAILY_ROOT / "trade_date=20250102" / "data.parquet"
    assert future_partition.is_file(), "测试数据没写进去，本用例会空跑通过"
    assert len(pd.read_parquet(future_partition)) == 1

    got = accessor.get_bars_before("000001.SZ", "20250101")

    assert not got.empty, "结果为空说明连 20250101 都没取到，断言会失去意义"
    assert sorted(got["trade_date"].dt.strftime("%Y%m%d")) == ["20250101"]
    assert "20250102" not in set(got["trade_date"].dt.strftime("%Y%m%d"))
    # 未来那天的价格没有以任何形式泄漏进来
    assert 11.0 not in set(got["close"])


def test_as_of_day_is_included_by_default():
    """默认 include_as_of=True，即 ``<=``：as_of=20250102 时应含当日。

    这条把「含当日」的语义钉死。若有人把条件从 ``<=`` 改成 ``<``，这里会失败，
    而不是让语义悄悄变形。
    """
    _save(_bars("20250101", close=10.0), _bars("20250102", close=11.0))

    got = accessor.get_bars_before("000001.SZ", "20250102")

    assert [d.strftime("%Y%m%d") for d in got["trade_date"]] == ["20250101", "20250102"]
    assert got["close"].iloc[-1] == pytest.approx(11.0)


def test_include_as_of_false_excludes_same_day():
    """保守口径：盘中决策应使用严格早于当日的口径。"""
    _save(_bars("20250101", close=10.0), _bars("20250102", close=11.0))

    got = accessor.get_bars_before("000001.SZ", "20250102", include_as_of=False)

    assert [d.strftime("%Y%m%d") for d in got["trade_date"]] == ["20250101"]


def test_lookback_returns_last_n_in_ascending_order():
    """lookback 截取「最近 N 根」，且返回顺序是升序（最新一根在末尾）。"""
    days = ["20250102", "20250103", "20250106", "20250107", "20250108"]
    _save(*(_bars(d, close=10.0 + i) for i, d in enumerate(days)))

    got = accessor.get_bars_before("000001.SZ", "20250108", lookback=3)

    assert [d.strftime("%Y%m%d") for d in got["trade_date"]] == ["20250106", "20250107", "20250108"]
    assert got["trade_date"].is_monotonic_increasing
    assert list(got.index) == [0, 1, 2]


def test_insufficient_data_returns_all_available():
    """数据不足 lookback 时返回全部，不抛错。"""
    _save(_bars("20250102"), _bars("20250103"))

    got = accessor.get_bars_before("000001.SZ", "20250103", lookback=100)

    assert len(got) == 2


def test_columns_follow_canonical_units_by_default():
    """默认返回规范层：vol(手) → volume(股)，amount(千元) → 元。"""
    _save(_bars("20250102"))

    canon = accessor.get_bars_before("000001.SZ", "20250102")
    raw = accessor.get_bars_before("000001.SZ", "20250102", canonical=False)

    assert "volume" in canon.columns and "vol" not in canon.columns
    assert canon["volume"].iloc[0] == pytest.approx(1000.0 * 100)
    assert canon["amount"].iloc[0] == pytest.approx(1000.0 * 1000)
    assert "vol" in raw.columns and "volume" not in raw.columns


def test_empty_range_returns_empty_frame_with_schema():
    """无数据时返回同 schema 空表，不是 None，调用方可直接 rolling。"""
    got = accessor.get_bars_before("000001.SZ", "20250102")

    assert got.empty
    assert list(got.columns) == list(storage.CANONICAL_COLUMNS)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="symbol"):
        accessor.get_bars_before("   ", "20250102")
    with pytest.raises(ValueError, match="lookback"):
        accessor.get_bars_before("000001.SZ", "20250102", lookback=0)
    with pytest.raises(ValueError, match="非法日期"):
        accessor.get_bars_before("000001.SZ", "2025-13-99")


def test_same_slice_read_twice_is_identical():
    """同一时间切片重复读取，两次结果必须完全相同。

    每次调用都会新建 DuckDB 连接、重新解析 Parquet 文件列表，所以这不是
    「读同一个内存对象」的假一致。将来若改动存储格式或查询逻辑导致行为不稳定，
    这里会失败。
    """
    _save(_bars("20250101", close=10.0), _bars("20250102", close=11.0))

    first = accessor.get_bars_before("000001.SZ", "20250102")
    second = accessor.get_bars_before("000001.SZ", "20250102")

    # 先确认不是空表，否则「两个空表相等」会给出无意义的通过
    assert not first.empty
    pd.testing.assert_frame_equal(first, second)


def test_same_slice_read_twice_is_identical_when_truncated():
    """重复读取一致性，但**必须让 LIMIT 真的截断**。

    上一条用例覆盖不到非确定性：数据只有 2 行且全部返回时，即使 SQL 的排序完全
    随机，Python 侧 ``sort_values`` 也会把顺序抹平，断言照样通过（已用
    ``ORDER BY random()`` 变异验证过，确实抓不到）。

    非确定性真正暴露的地方是**截断**：若 SQL 未按 trade_date 稳定排序，``LIMIT n``
    每次可能挑中不同的 n 根，之后再排序也救不回来。所以这里造 5 天数据、只取 3 行。
    """
    days = ["20250102", "20250103", "20250106", "20250107", "20250108"]
    _save(*(_bars(d, close=10.0 + i) for i, d in enumerate(days)))

    first = accessor.get_bars_before("000001.SZ", "20250108", lookback=3)
    second = accessor.get_bars_before("000001.SZ", "20250108", lookback=3)

    assert len(first) == 3, "未发生截断，本用例会退化成上一条，抓不到非确定性"
    assert [d.strftime("%Y%m%d") for d in first["trade_date"]] == [
        "20250106",
        "20250107",
        "20250108",
    ]
    pd.testing.assert_frame_equal(first, second)
