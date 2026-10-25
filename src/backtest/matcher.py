"""撮合器：把 SignalEvent + T+1 日 Bar 变成 FillEvent。

**为什么成交价必须取 ``next_bar.open``**

策略在 T 日生成信号时，T 日的 ``close`` 已经确定，但它**不能**用来成交——同一时点
既当决策依据又当成交价，是回测里最典型的未来函数。本模块把成交价绑死在 T+1 日
开盘价上：

- 信号日 T 的收盘 → 决策依据；
- T+1 日开盘 → 成交价（真实可执行的第一个价格）。

因此 ``signal`` 只贡献**方向**，不贡献价格。若将来要做「当日收盘成交」，那属于另一个
撮合模型（信号时点需提前到当日开盘前），不要在这里改 ``next_bar`` 的取法。

**数量从哪来**

``SignalEvent`` 只有 ``direction`` 和可选的 ``target_position``（占净值比例），没有
股数；``FillEvent`` 又必须有 ``quantity``。所以 :meth:`Matcher.match` 增加了一个可选
``quantity`` 参数：

- 回测引擎/Sizer 应当显式传入股数（推荐，也是唯一用于实盘决策的用法）；
- 省略时退化为 :data:`DEFAULT_QUANTITY`（1 手 = 100 股），**仅为让本模块可独立验证**。
  真实回测若一直走默认值，会产生「每笔都只交易 1 手」的隐形错误，因此日志/报表里
  必须能看出数量来自哪里。

**费用口径**

只计**佣金**：``max(成交金额 × 费率, 最低佣金)``，默认万 2、最低 5 元，按分四舍五入。
**未计印花税（卖出 0.05%）与过户费**——漏掉印花税会让回测收益每次卖出虚高约 5bp。
这是刻意留给费用模型的扩展点：做成参数后在 :meth:`_calc_fee` 一处加齐，不影响
撮合价格逻辑。

**未建模的部分（MVP 边界）**

- 涨跌停无法成交：需要昨收才能算涨跌停价，本模块只拿到一根 bar，信息不足；
- 停牌/一字板用 ``open <= 0`` 或 ``volume <= 0`` 近似判断；
- 部分成交与成交量占比上限（如不超过当日成交量的 10%）；
- T+1 可卖数量校验（属 Broker/风控，不属撮合）。
"""

from __future__ import annotations

import math

from backtest.events import BarEvent, FillEvent, SignalEvent

__all__ = ["Matcher", "DEFAULT_QUANTITY"]

#: 省略 ``quantity`` 时的兜底数量：1 手（100 股）。
#: 仅为独立验证与单测服务，**不应**作为真实回测的默认下单量。
DEFAULT_QUANTITY: float = 100.0

#: 浮点容差：把 10.004999999 这类误差推回 10.005 再进位，
#: 避免「本该 10.01 却算成 10.00」的二进制浮点偏心。
_EPS = 1e-9


def _round_half_up(value: float, digits: int) -> float:
    """四舍五入（不用内置 ``round`` 的「五入取偶」）。

    ``round(2.675, 2)`` 得到 2.67 而非 2.68，是二进制浮点与银行家舍入共同作用的结果。
    券商按四舍五入结算，这里显式实现，避免逐笔累积出方向性偏差。
    """
    factor = 10**digits
    return math.floor(value * factor + 0.5 + _EPS) / factor


class Matcher:
    """T+1 开盘成交的简化撮合器（单标的、市价单）。

    Args:
        slippage_rate: 滑点率，默认 ``0.0005``（万 5）。买入向上加、卖出向下降，
            滑点恒对己方不利——刻意的保守假设，避免回测靠「有利滑点」盈利。
        commission_rate: 佣金费率，默认 ``0.0002``（万 2）。
        min_commission: 单笔最低佣金（元），默认 ``5.0``（A 股券商普遍下限）。
        tick_size: 价格最小变动单位（元）。``0``（默认）不做 tick 取整，保持
            ``fill_price == open × (1 ± slippage_rate)`` 的精确关系；设为 ``0.01``
            则按 A 股最小报价单位取整，更贴近真实成交价，但与公式有至多 0.005 元偏差。

    Example:
        >>> from datetime import date
        >>> from backtest.events import BarEvent, SignalEvent
        >>> m = Matcher()
        >>> sig = SignalEvent("600519.SH", date(2025, 1, 2), 1)          # T 日：只给方向
        >>> bar = BarEvent("600519.SH", date(2025, 1, 3), 100.0, 101.0, 99.0, 100.5, 1_000_000)
        >>> f = m.match(sig, bar, quantity=1000)                        # T+1 开盘 100.0
        >>> round(f.fill_price, 4), round(f.slippage, 4), round(f.fee, 2)
        (100.05, 0.05, 20.01)
        >>> round(f.cash_delta, 2)                                       # 买入：付出 100070.01
        -100070.01
    """

    __slots__ = ("_slippage_rate", "_commission_rate", "_min_commission", "_tick_size")

    def __init__(
        self,
        *,
        slippage_rate: float = 0.0005,
        commission_rate: float = 0.0002,
        min_commission: float = 5.0,
        tick_size: float = 0.0,
    ) -> None:
        for name, value in (
            ("slippage_rate", slippage_rate),
            ("commission_rate", commission_rate),
            ("min_commission", min_commission),
            ("tick_size", tick_size),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是数值，收到 {type(value).__name__}: {value!r}")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数，收到：{value!r}")
        # 滑点率 > 100% 会把卖出价推到 0 以下，属于明显配置错误而非极端行情
        if slippage_rate >= 1.0:
            raise ValueError(f"slippage_rate 应小于 1（100%），收到：{slippage_rate!r}")

        self._slippage_rate = float(slippage_rate)
        self._commission_rate = float(commission_rate)
        self._min_commission = float(min_commission)
        self._tick_size = float(tick_size)

    # ------------------------------------------------------------ 只读参数

    @property
    def slippage_rate(self) -> float:
        """滑点率（买加卖减，恒对己方不利）。"""
        return self._slippage_rate

    @property
    def commission_rate(self) -> float:
        """佣金费率。"""
        return self._commission_rate

    @property
    def min_commission(self) -> float:
        """单笔最低佣金（元）。"""
        return self._min_commission

    @property
    def tick_size(self) -> float:
        """价格最小变动单位（0 表示不取整）。"""
        return self._tick_size

    # ---------------------------------------------------------------- 撮合

    def match(
        self,
        signal: SignalEvent,
        next_bar: BarEvent | None,
        quantity: float | None = None,
    ) -> FillEvent | None:
        """把信号撮合成一笔成交。

        Args:
            signal: T 日生成的信号。**只取方向**，成交价来自 ``next_bar``。
            next_bar: T+1 日的 bar。``None`` 表示数据已结束，该信号作废。
            quantity: 成交股数。省略时用 :data:`DEFAULT_QUANTITY`（1 手），仅便于
                独立验证；真实回测应由 Sizer 显式传入。

        Returns:
            成交回报；以下情况返回 ``None``（不成交，而不是抛异常）：

            - ``next_bar is None``：区间结束，最后一根 bar 之后的信号无法成交；
            - ``signal.direction == 0``：持有，无交易意图；
            - ``next_bar.open <= 0``：停牌或无有效开盘价；
            - ``next_bar.volume <= 0``：当日无成交（停牌/一字板近似判断）。

        Raises:
            ValueError: 标的代码不一致；``next_bar.trade_date <= signal.signal_date``；
                或 ``quantity <= 0``。

        为什么「日期不递增」要报错而不是返回 ``None``：静默跳过只会让回测少一笔
        成交，表面看是收益略低，实际掩盖了引擎把信号日 bar 当成次日 bar 的时序
        bug——那正是未来函数的入口，必须炸出来。
        """
        if next_bar is None:
            return None
        if not isinstance(signal, SignalEvent):
            raise TypeError(f"signal 需要 SignalEvent，收到 {type(signal).__name__}")
        if not isinstance(next_bar, BarEvent):
            raise TypeError(f"next_bar 需要 BarEvent 或 None，收到 {type(next_bar).__name__}")

        if signal.symbol != next_bar.symbol:
            raise ValueError(
                f"信号与行情标的不一致：signal={signal.symbol} next_bar={next_bar.symbol}"
            )
        if next_bar.trade_date <= signal.signal_date:
            raise ValueError(
                f"T+1 撮合要求成交日严格晚于信号日，收到 signal_date="
                f"{signal.signal_date}、next_bar.trade_date={next_bar.trade_date}；"
                f"通常意味着引擎把信号日当天的 bar 当成次日 bar 传入（未来函数）"
            )

        # 持有信号：没有交易意图，不产生成交
        if signal.direction == 0:
            return None

        # 停牌 / 无有效价格：不成交。先判行情有效性，再算钱
        if next_bar.open <= 0 or next_bar.volume <= 0:
            return None

        qty = DEFAULT_QUANTITY if quantity is None else quantity
        if isinstance(qty, bool) or not isinstance(qty, (int, float)):
            raise TypeError(f"quantity 必须是数值，收到 {type(qty).__name__}: {qty!r}")
        if not math.isfinite(float(qty)) or qty <= 0:
            raise ValueError(f"quantity 必须为正数，收到：{qty!r}")

        fill_price = self._fill_price(next_bar.open, signal.direction)
        notional = fill_price * float(qty)
        fee = self._calc_fee(notional)

        return FillEvent(
            symbol=signal.symbol,
            fill_date=next_bar.trade_date,
            fill_price=fill_price,
            quantity=float(qty),
            direction=signal.direction,
            fee=fee,
            slippage=abs(fill_price - next_bar.open),
        )

    # -------------------------------------------------------------- 内部计算

    def _fill_price(self, open_price: float, direction: int) -> float:
        """成交价 = 开盘价 × (1 ± 滑点率)，买加卖减。

        买入加滑点（买贵）、卖出减滑点（卖便宜），两个方向都对己方不利。
        """
        price = open_price * (1 + direction * self._slippage_rate)
        if self._tick_size > 0:
            price = _round_half_up(price / self._tick_size, 0) * self._tick_size
            price = _round_half_up(price, 10)  # 消掉 tick 乘法的浮点尾巴
        return price

    def _calc_fee(self, notional: float) -> float:
        """佣金 = max(成交金额 × 费率, 最低佣金)，按分四舍五入。

        最低佣金必须在**取整后**再比：若先比再取整，2.001→5.0 没问题，但
        4.996→5.00 与 5.004→5.00 的边界会因顺序不同而抖动。
        """
        fee = _round_half_up(notional * self._commission_rate, 2)
        return max(fee, self._min_commission)

    def estimate_fee(self, notional: float) -> float:
        """按给定成交金额估算佣金（元），供 Sizer 预留资金时使用。"""
        if isinstance(notional, bool) or not isinstance(notional, (int, float)):
            raise TypeError(f"notional 必须是数值，收到 {type(notional).__name__}: {notional!r}")
        if not math.isfinite(float(notional)) or notional < 0:
            raise ValueError(f"notional 必须是非负有限数，收到：{notional!r}")
        return self._calc_fee(float(notional))
