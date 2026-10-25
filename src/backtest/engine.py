"""回测主循环：单标的、日线、T 日收盘出信号 / T+1 开盘成交。

**时序（本模块最容易被写错的地方）**

每个交易日的处理顺序固定为：

1. **先成交**：把**昨日**的信号用**今日开盘价**撮合（T+1 的真实落地）；
2. **再决策**：用截至今日收盘的历史调用策略，得到今日信号；
3. **最后记净值**：按今日收盘价盯市，写入 ``equity/cash/position``。

顺序不能换。若「先决策、立刻用今日 open 成交、再记今日 close」，等于用一个开盘时
还不知道的信号在当天开盘成交——比未来函数更隐蔽，因为净值曲线看起来完全正常。
若「先记今日净值、再让今日信号成交」，持仓又会提前一天出现在净值里。

**为什么要多读一天**

最后一根 bar（``end_date``）的信号，其 T+1 落在区间之外。不额外取一根，这笔信号
永远无法撮合，回测会系统性少一次交易（双均线策略可能正好少掉最后一段趋势）。
因此取数时向后多留 :data:`EXTRA_CALENDAR_DAYS` 个自然日，只为定位 ``end_date`` 之后
**紧邻的一根** bar：

- 该 bar 只用作 ``next_bar``，**永远不会**进入 ``strategy_func`` 的入参（入参按位置切片）；
- 若它确实是下一交易日，末笔信号能成交，记入 ``result.attrs["trades"]``；
- 净值序列仍严格截止于 ``end_date``，不掺入区间外的日期。

**数量从哪来**

``SignalEvent`` 只有方向，没有股数。本模块内置一个最简 Sizer：

- 买入：``cash × max_position_fraction`` 除以「开盘价×(1+滑点)×(1+费率)」，向下取整到
  ``lot_size``（默认 100 股）的整数倍；买不起 1 手就不下单；
- 卖出：清掉全部持仓（只做多，没有分批减仓语义）。

这不是「好的」仓位管理，只是让回测能跑通的最小实现。要做等权、波动率目标、
固定风险预算，应把 Sizer 抽出来替换这一处，而不是改撮合或记账。
"""

from __future__ import annotations

import math
import numbers
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from backtest.events import BarEvent, SignalEvent
from backtest.matcher import Matcher
from backtest.portfolio import Portfolio
from data.accessor import get_bars_before

__all__ = ["run_backtest", "EXTRA_CALENDAR_DAYS", "DEFAULT_LOT_SIZE"]

#: 取数时向 ``end_date`` 之后多留的自然日数。只用于定位「下一个交易日」，
#: 覆盖周末与春节/国庆长假（A 股最长连续休市约 9~10 天，30 天留足余量）。
EXTRA_CALENDAR_DAYS: int = 30

#: A 股最小交易单位（股）。买入必须为其整数倍；卖出允许零股，但本引擎持仓恒为整数倍。
DEFAULT_LOT_SIZE: int = 100

_HOLD = 0
_COLUMNS = ["trade_date", "equity", "cash", "position"]
_TRADE_COLUMNS = [
    "signal_date",
    "fill_date",
    "direction",
    "quantity",
    "fill_price",
    "fee",
    "slippage",
    "cash_after",
    "position_after",
]


def _to_date(value: object, name: str) -> date:
    """把 str / date / datetime / Timestamp 统一成 ``datetime.date``。

    与 ``events._as_date`` 同一口径：只做日期截断，不做时区转换。额外接受
    ``YYYYMMDD`` 字符串，因为命令行传参通常是字符串。
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        digits = value.strip().replace("-", "").replace("/", "")
        if len(digits) == 8 and digits.isdigit():
            try:
                return datetime.strptime(digits, "%Y%m%d").date()
            except ValueError as exc:
                raise ValueError(f"{name} 非法日期：{value!r}") from exc
        raise ValueError(f"{name} 无法解析：{value!r}，期望 YYYYMMDD / YYYY-MM-DD")
    raise TypeError(f"{name} 需要 str/date/datetime，收到 {type(value).__name__}: {value!r}")


def _make_bar(symbol: str, row: pd.Series) -> BarEvent | None:
    """把一行行情转成 BarEvent；脏数据返回 ``None`` 而不是抛异常。

    停牌日部分数据源会给全 0 或 NaN 的 OHLC，这类行无法参与撮合。跳过比让
    BarEvent 的校验炸掉整轮回测更合适——跳过行数记在 ``attrs["skipped_bars"]``。
    """
    values = [row["open"], row["high"], row["low"], row["close"], row["volume"]]
    if any(pd.isna(v) or not math.isfinite(float(v)) for v in values):
        return None
    if min(float(row["open"]), float(row["low"]), float(row["close"])) <= 0:
        return None
    try:
        return BarEvent(
            symbol=symbol,
            trade_date=row["trade_date"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
    except ValueError:
        # OHLC 不自洽（high < max(open, close) 之类）同样按脏数据跳过
        return None


def _read_signal(raw: object) -> int:
    """校验策略返回值：必须是 ``1`` / ``-1`` / ``0`` 的整数。

    不做 ``== 1`` 这类松散判断：``True`` 会被当成 1、``1.0`` 会被当成 1，这些
    「看起来能跑」的返回值迟早在某处改变语义。策略接口就该是严格的。
    """
    if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
        raise TypeError(
            f"strategy_func 必须返回整数信号 1/-1/0，收到 {type(raw).__name__}: {raw!r}"
        )
    value = int(raw)
    if value not in (-1, 0, 1):
        raise ValueError(f"strategy_func 返回 {value}，只允许 1(买) / -1(卖) / 0(持有)")
    return value


def _buy_quantity(
    cash: float, bar: BarEvent, matcher: Matcher, lot_size: int, fraction: float
) -> float:
    """按可用现金估算可买股数，向下取整到整手；买不起 1 手返回 0。

    先粗算再逐手回退：滑点与佣金让实际成本高于粗算值，差一手就退一手，宁可少买
    也不让 ``Portfolio`` 抛「现金不足」把整轮回测打断。
    """
    est_price = bar.open * (1 + matcher.slippage_rate) * (1 + matcher.commission_rate)
    if est_price <= 0:
        return 0.0
    qty = math.floor(cash * fraction / est_price / lot_size) * lot_size

    fill_price = bar.open * (1 + matcher.slippage_rate)
    while qty >= lot_size:
        notional = fill_price * qty
        if notional + matcher.estimate_fee(notional) <= cash + 1e-9:
            return float(qty)
        qty -= lot_size
    return 0.0


def _empty_result(code: str, start: date, end: date, initial_cash: float, **extra: object) -> pd.DataFrame:
    """无数据时返回带固定列的空表（不返回 None），attrs 仍填充基础信息。"""
    empty = pd.DataFrame(columns=_COLUMNS)
    empty.attrs.update(
        symbol=code,
        start_date=start,
        end_date=end,
        initial_cash=float(initial_cash),
        final_equity=float(initial_cash),
        total_return=0.0,
        trades=pd.DataFrame(columns=_TRADE_COLUMNS),
        unfilled_last_signal=False,
        skipped_bars=0,
        unaffordable_buys=0,
        **extra,
    )
    return empty


def run_backtest(
    symbol: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    strategy_func: Callable[[pd.DataFrame], int],
    initial_cash: float = 100_000,
    *,
    lot_size: int = DEFAULT_LOT_SIZE,
    max_position_fraction: float = 1.0,
    matcher: Matcher | None = None,
) -> pd.DataFrame:
    """跑一次单标的日线回测。

    Args:
        symbol: 标的代码，如 ``"000001.SZ"``（大小写与首尾空格会被归一）。
        start_date: 回测起始日（含）。净值序列不会早于此日。
        end_date: 回测结束日（含）。信号可在此日产生，成交落在其后的交易日。
        strategy_func: 只接受**一个**参数——截至当日（含）的历史 DataFrame，返回
            ``1`` 买入 / ``-1`` 卖出 / ``0`` 持有。入参列为
            ``ts_code, trade_date, open, high, low, close, volume, amount``
            （规范层，volume 单位股）。
        initial_cash: 初始资金（元），默认 10 万。
        lot_size: 最小交易单位（股），默认 100。
        max_position_fraction: 单次买入最多动用现金的比例，默认 1.0。
        matcher: 撮合器；省略则用 ``Matcher()``（滑点万 5、佣金万 2、最低 5 元）。

    关于预热：策略入参是「截至当日的**全部可得历史**」，取数上限按区间自然日数估算，
    而自然日数 ≥ 交易日数，因此起点之前的历史会被自然带上，等价于免费预热。
    代价是无法精确控制「只喂最近 N 根」，但对日线策略而言多喂几根历史无害。

    Returns:
        DataFrame，列固定为 ``trade_date``、``equity``、``cash``、``position``，
        按日期升序，覆盖 ``[start_date, end_date]`` 内实际有行情的交易日（停牌日
        无行、不做前值填充）。``trade_date`` 为 ``Timestamp``。

        额外信息在 ``result.attrs``：

        - ``trades``：成交明细（含 ``signal_date`` 与 ``fill_date``，两者相差 1 个
          交易日，是核对 T+1 的直接证据；末笔可能落在 ``end_date`` 之后）；
        - ``final_equity`` / ``total_return``：期末权益与区间收益率；
        - ``unfilled_last_signal``：末笔信号是否因为没有下一根 bar 而作废；
        - ``skipped_bars``：被跳过的脏数据行数；
        - ``unaffordable_buys``：因现金不足 1 手而未下单的次数。

    Raises:
        ValueError: 日期非法、``start_date > end_date``、``initial_cash <= 0``、
            ``lot_size`` 非正整数、``max_position_fraction`` 不在 ``(0, 1]``、
            ``warmup_bars`` 为负、或策略返回非 ``1/-1/0``。
        TypeError: ``strategy_func`` 不可调用，或返回值不是整数。
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol 不能为空")
    if not callable(strategy_func):
        raise TypeError(f"strategy_func 必须可调用，收到 {type(strategy_func).__name__}")

    start = _to_date(start_date, "start_date")
    end = _to_date(end_date, "end_date")
    if start > end:
        raise ValueError(f"start_date({start}) 晚于 end_date({end})")

    if isinstance(initial_cash, bool) or not isinstance(initial_cash, (int, float)):
        raise TypeError(f"initial_cash 必须是数值，收到 {type(initial_cash).__name__}")
    if not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError(f"initial_cash 必须为正数，收到：{initial_cash!r}")

    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
        raise ValueError(f"lot_size 必须是正整数，收到：{lot_size!r}")
    if not 0 < max_position_fraction <= 1:
        raise ValueError(f"max_position_fraction 应在 (0, 1] 内，收到：{max_position_fraction!r}")

    code = symbol.strip().upper()
    matcher = matcher or Matcher()

    # ---- 取数：向后多留一段只为拿到 end_date 之后紧邻的一根 bar（见模块 docstring）
    fetch_end = end + timedelta(days=EXTRA_CALENDAR_DAYS)
    span_days = (fetch_end - start).days + 1
    bars = get_bars_before(code, fetch_end, lookback=span_days)

    if bars.empty:
        return _empty_result(code, start, end, initial_cash, bars_available=0)

    bars = bars.reset_index(drop=True)
    bar_dates = [ts.date() for ts in bars["trade_date"]]
    positions_in_range = [i for i, d in enumerate(bar_dates) if start <= d <= end]
    if not positions_in_range:
        return _empty_result(code, start, end, initial_cash, bars_available=len(bars))

    portfolio = Portfolio(initial_cash)
    records: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    stats = {"skipped_bars": 0, "unaffordable_buys": 0}
    pending: SignalEvent | None = None
    unfilled_last_signal = False

    def execute(signal: SignalEvent, bar: BarEvent) -> None:
        """按方向定量、撮合、记账、留痕。"""
        qty = 0.0
        if signal.direction == 1:
            qty = _buy_quantity(portfolio.cash, bar, matcher, lot_size, max_position_fraction)
            if qty <= 0:
                stats["unaffordable_buys"] += 1
        elif portfolio.position > 0:
            qty = portfolio.position

        if qty <= 0:
            return
        fill = matcher.match(signal, bar, quantity=qty)
        if fill is None:
            return
        portfolio.on_fill(fill)
        trades.append(
            {
                "signal_date": signal.signal_date,
                "fill_date": fill.fill_date,
                "direction": fill.direction,
                "quantity": fill.quantity,
                "fill_price": fill.fill_price,
                "fee": fill.fee,
                "slippage": fill.slippage,
                "cash_after": portfolio.cash,
                "position_after": portfolio.position,
            }
        )

    last_pos = positions_in_range[-1]

    for pos in positions_in_range:
        row = bars.iloc[pos]
        today = bar_dates[pos]
        bar = _make_bar(code, row)
        if bar is None:
            stats["skipped_bars"] += 1
            continue

        # (1) 昨日信号在今日开盘成交 —— T+1 的真实落地
        if pending is not None:
            execute(pending, bar)
            pending = None

        # (2) 今日收盘后决策：入参严格是「截至今日」的历史，不含未来
        signal_value = _read_signal(strategy_func(bars.iloc[: pos + 1]))
        if signal_value != _HOLD:
            pending = SignalEvent(code, today, signal_value)

        # (3) 今日收盘盯市，记净值
        records.append(
            {
                "trade_date": row["trade_date"],
                "equity": portfolio.total_equity(bar.close),
                "cash": portfolio.cash,
                "position": portfolio.position,
            }
        )

    # ---- 区间末笔信号：用「区间外紧邻的一根」撮合（这正是多读一天的目的）
    if pending is not None:
        if last_pos + 1 < len(bars):
            extra_bar = _make_bar(code, bars.iloc[last_pos + 1])
            if extra_bar is None:
                unfilled_last_signal = True
            else:
                execute(pending, extra_bar)
        else:
            unfilled_last_signal = True
        pending = None

    result = pd.DataFrame(records, columns=_COLUMNS)
    final_equity = float(result["equity"].iloc[-1])
    result.attrs.update(
        symbol=code,
        start_date=start,
        end_date=end,
        initial_cash=float(initial_cash),
        final_equity=final_equity,
        total_return=final_equity / float(initial_cash) - 1,
        trades=pd.DataFrame(trades, columns=_TRADE_COLUMNS),
        unfilled_last_signal=unfilled_last_signal,
        bars_available=int(len(bars)),
        last_processed_date=bar_dates[last_pos],
        **stats,
    )
    return result
