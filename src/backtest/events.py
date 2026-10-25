"""回测引擎的事件模型。

一条完整的事件流是单向的::

    BarEvent ──► Strategy ──► SignalEvent ──► Sizer ──► OrderEvent ──► Broker ──► FillEvent

**不变式（改代码前先读）**

1. 事件一经创建即**不可变**（``frozen=True``）。回测里最怕的 bug 是下游悄悄改写了
   上游事件（比如撮合时把 ``fill_price`` 改回理论价），不可变能让这类错误变成
   明确的 ``FrozenInstanceError`` 而不是算错。
2. 日期字段一律规范化为 :class:`datetime.date`。``datetime`` / ``pandas.Timestamp``
   会被截断到日（``Timestamp`` 是 ``datetime`` 的子类，因此无需依赖 pandas）。
   原因：``date(2025, 1, 2) == pd.Timestamp("2025-01-02")`` 的结果是 **False**，
   两种类型混用会让对账、分组、去重静默失效。
3. 事件只描述「发生了什么」，不包含撮合规则、仓位上限等策略或风控逻辑。

**单位约定（务必与 ``docs/data-dictionary.md`` §1.4 对齐）**

========================  ====================================================
价格（open/high/low/close/fill_price）  元/股（不复权，与落盘数据一致）
``volume`` / ``quantity``                股（不是手）
``fee``                                  元，**合计**费用（佣金＋印花税＋过户费）
``slippage``                             元/股，每股滑点，正数表示对己方不利
========================  ====================================================

``slippage`` 与 ``fill_price`` 刻意**分成两个字段**：前者是「为什么成交价比理论价差」，
后者是「实际成交在什么价」。合并成一个「成交价」会让归因分析（手续费侵蚀 vs 滑点
侵蚀 vs 策略亏损）无法拆分，也没法在模拟盘里单独标定滑点参数。

**as-of 语义**：BarEvent 只携带 ``trade_date``，不携带 ``available_at``。日线 bar 在
当日 15:00 才完整（数据字典 §4.2），这一约束由取数接口
``data.accessor.get_bars_before`` 负责，事件本身不重复表达——但引擎在把 BarEvent
投递给策略时必须遵守「同一 ``trade_date`` 的 bar 不得用于当日盘中决策」。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

__all__ = [
    "BarEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "EVENT_ORDER",
    "SUPPORTED_ORDER_TYPES",
]


#: 同一交易日内的确定性处理顺序：先喂行情，再出信号，再下单，最后成交。
#: 事件队列排序时用它做次级键，保证「两次运行结果一致」（PRD 验收项）。
EVENT_ORDER: dict[str, int] = {
    "BarEvent": 0,
    "SignalEvent": 1,
    "OrderEvent": 2,
    "FillEvent": 3,
}

#: MVP 只支持市价单。限价单需要额外携带 ``price`` 字段并配套撮合规则，属于推迟项，
#: 因此这里显式拒绝而不是接受后静默按市价处理。
SUPPORTED_ORDER_TYPES: frozenset[str] = frozenset({"market"})


def _as_date(value: object, field_name: str) -> date:
    """把 date / datetime / pandas.Timestamp 统一成 ``datetime.date``。

    只做日期截断，不做时区转换——本项目所有时间戳都是 ``Asia/Shanghai`` 的 naive
    时间（数据字典 §5），带 tzinfo 的输入在这里会被拒绝，避免 UTC 混入后整体偏移 8 小时。
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise ValueError(
                f"{field_name} 不允许带时区（收到 tzinfo={value.tzinfo}）；"
                f"本项目统一使用 Asia/Shanghai 的 naive 时间"
            )
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(
        f"{field_name} 必须是 date 或 datetime，收到 {type(value).__name__}: {value!r}"
    )


def _require_positive(value: float, field_name: str) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{field_name} 必须为正数，收到：{value!r}")


def _require_non_negative(value: float, field_name: str) -> None:
    if value is None or value < 0:
        raise ValueError(f"{field_name} 不能为负，收到：{value!r}")


@dataclass(frozen=True, slots=True)
class BarEvent:
    """一根已完成的行情 bar（策略的输入）。

    只承载已收盘的 bar：构造它就意味着「这根数据可以用了」，因此引擎不得把当日
    尚未收完的 bar 投递给盘中决策（见模块 docstring 的 as-of 说明）。

    Attributes:
        symbol: 标的代码，如 ``600519.SH``。
        trade_date: 交易日。
        open: 开盘价（元/股，不复权）。
        high: 最高价（元/股，不复权）。
        low: 最低价（元/股，不复权）。
        close: 收盘价（元/股，不复权）。
        volume: 成交量（股）。
    """

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if not self.symbol.isupper():
            raise ValueError(f"symbol 必须为大写（数据字典 §1），收到：{self.symbol!r}")
        object.__setattr__(self, "trade_date", _as_date(self.trade_date, "trade_date"))

        for name in ("open", "high", "low", "close"):
            _require_positive(getattr(self, name), name)
        _require_non_negative(self.volume, "volume")

        # OHLC 自洽性：脏数据在这里就炸，而不是等到算出离谱收益再回头查
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(
                f"OHLC 不自洽：open={self.open} high={self.high} "
                f"low={self.low} close={self.close}"
            )

    @property
    def sort_key(self) -> tuple[date, int, str]:
        """事件队列排序键：先日期，再事件类型，最后代码。"""
        return (self.trade_date, EVENT_ORDER["BarEvent"], self.symbol)


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """策略输出的交易意图。

    Attributes:
        symbol: 标的代码。
        signal_date: 信号生成日。**这是对账的关键时间戳**——它回答了「策略在哪天
            看到的什么、据此决定了什么」，与 :attr:`OrderEvent.order_date`、
            :attr:`FillEvent.fill_date` 串起来才能还原「信号 → 下单 → 成交」的耗时与
            偏差。缺了它，只能知道「某天买了」，无法判断是信号延迟还是撮合延迟。
        direction: ``1`` 买入 / ``-1`` 卖出 / ``0`` 持有（不动）。
        target_position: 目标仓位，表示占组合净值的比例（如 ``0.1`` 表示 10%），
            可选。参考开源实现的用法：``direction`` 表达「想干什么」，``target_position``
            表达「想做到多大」，便于按组合权重调仓而非按股数下单。
            为 ``None`` 时由仓位管理层（Sizer）根据 ``direction`` 自行决定数量。
    """

    symbol: str
    signal_date: date
    direction: int
    target_position: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if not self.symbol.isupper():
            raise ValueError(f"symbol 必须为大写（数据字典 §1），收到：{self.symbol!r}")
        object.__setattr__(self, "signal_date", _as_date(self.signal_date, "signal_date"))

        if self.direction not in (1, -1, 0):
            raise ValueError(
                f"direction 只能是 1(买) / -1(卖) / 0(持有)，收到：{self.direction!r}"
            )
        if isinstance(self.direction, bool):
            raise TypeError("direction 必须是 int，不要用 bool（True 会被当成 1）")

        if self.target_position is not None:
            if not -1.0 <= float(self.target_position) <= 1.0:
                raise ValueError(
                    f"target_position 应在 [-1, 1] 内（负值表示做空，A 股 MVP 不支持），"
                    f"收到：{self.target_position!r}"
                )
            if self.direction == 0 and self.target_position not in (0, None):
                raise ValueError(
                    f"direction=0（持有）却给出 target_position={self.target_position}，语义冲突"
                )

    @property
    def is_actionable(self) -> bool:
        """是否为需要落到订单的信号（``0`` 表示只持有，无需下单）。"""
        return self.direction != 0

    @property
    def sort_key(self) -> tuple[date, int, str]:
        return (self.signal_date, EVENT_ORDER["SignalEvent"], self.symbol)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """发给交易网关的委托单（模拟盘与实盘共用同一套字段）。

    Attributes:
        symbol: 标的代码。
        order_date: 下单日（成交可能发生在之后的交易日，T+1 制度下必然如此）。
        quantity: 委托数量，单位**股**，恒为正数；买卖方向由 :attr:`direction` 表达。
            不用正负号表示方向，是为了避免「数量为负表示卖出」这种约定与持仓计算里的
            负号相乘后变成正数，从而静默反向。
        direction: ``1`` 买入 / ``-1`` 卖出。**不允许 0**：不需要动作就不应该下单。
        order_type: 委托类型，MVP 仅支持 ``"market"``（见 :data:`SUPPORTED_ORDER_TYPES`）。
    """

    symbol: str
    order_date: date
    quantity: float
    direction: int
    order_type: str = "market"

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if not self.symbol.isupper():
            raise ValueError(f"symbol 必须为大写（数据字典 §1），收到：{self.symbol!r}")
        object.__setattr__(self, "order_date", _as_date(self.order_date, "order_date"))

        if self.direction not in (1, -1):
            raise ValueError(
                f"OrderEvent.direction 只能是 1(买) / -1(卖)，收到：{self.direction!r}；"
                f"无需动作时不应该生成订单"
            )
        _require_positive(self.quantity, "quantity")

        if self.order_type not in SUPPORTED_ORDER_TYPES:
            raise ValueError(
                f"暂不支持 order_type={self.order_type!r}；"
                f"MVP 只支持 {sorted(SUPPORTED_ORDER_TYPES)}，"
                f"限价单需要额外携带 price 字段，属推迟项"
            )

    @property
    def signed_quantity(self) -> float:
        """带方向的股数：买入为正、卖出为负，供持仓累计使用。"""
        return self.quantity * self.direction

    @property
    def sort_key(self) -> tuple[date, int, str]:
        return (self.order_date, EVENT_ORDER["OrderEvent"], self.symbol)


@dataclass(frozen=True, slots=True)
class FillEvent:
    """成交回报。

    ``fill_price`` 与 ``slippage`` 分开保存，是要能分别归因：

    - ``fill_price``：**实际**成交价（含滑点后的价格）；
    - ``slippage``：相对参考价（通常是理论价或信号日收盘价）的**每股**滑点，
      正数表示对己方不利（买贵 / 卖便宜）；
    - ``fee``：本次成交的总费用。

    三者关系（参考价已知时）：``fill_price ≈ 参考价 + direction * slippage``。
    等式不强制校验，因为参考价的选择属于撮合模型的内部约定，事件层不该替它做主。

    Attributes:
        symbol: 标的代码。
        fill_date: 成交日（T+1 制度下通常晚于 ``order_date``）。
        fill_price: 实际成交价，元/股。
        quantity: 成交数量，股，恒为正。
        direction: ``1`` 买入 / ``-1`` 卖出。
        fee: 本次成交的总费用，元（佣金＋印花税＋过户费，不含滑点）。
        slippage: 每股滑点，元/股；正数表示不利，可为负（有利滑点）。
    """

    symbol: str
    fill_date: date
    fill_price: float
    quantity: float
    direction: int
    fee: float
    slippage: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if not self.symbol.isupper():
            raise ValueError(f"symbol 必须为大写（数据字典 §1），收到：{self.symbol!r}")
        object.__setattr__(self, "fill_date", _as_date(self.fill_date, "fill_date"))

        if self.direction not in (1, -1):
            raise ValueError(
                f"FillEvent.direction 只能是 1(买) / -1(卖)，收到：{self.direction!r}"
            )
        _require_positive(self.fill_price, "fill_price")
        _require_positive(self.quantity, "quantity")
        _require_non_negative(self.fee, "fee")

        if not isinstance(self.slippage, (int, float)) or isinstance(self.slippage, bool):
            raise TypeError(f"slippage 必须是数值，收到：{self.slippage!r}")

    @property
    def notional(self) -> float:
        """成交金额（元），不含费用。"""
        return self.fill_price * self.quantity

    @property
    def cash_delta(self) -> float:
        """对现金的影响（元）：买入为负（付出），卖出为正（收到），已扣费用。"""
        return -self.direction * self.notional - self.fee

    @property
    def sort_key(self) -> tuple[date, int, str]:
        return (self.fill_date, EVENT_ORDER["FillEvent"], self.symbol)
