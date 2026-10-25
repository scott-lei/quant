"""账户与组合计算：单标的、只做多、现金账户。

**职责边界**

本类只做「成交 → 现金/持仓/成本」的记账，不判断该不该成交。撮合（价格、滑点、
费用）、风控（仓位上限、现金是否够）分别属于 Broker 与 Risk 层。但有一条硬约束
写在记账层：**持仓不能为负、现金默认不能为负**——这两条一旦被静默放过，回测就会
在不知不觉中变成「做空 + 融资」，而那正是 PRD 明确排除的范围。

**成本口径**

``avg_cost`` 是**每股买入均价，只含成交价、不含费用**：

- 以 10 元买 1000 股、fee=5，则 ``avg_cost == 10.0``，不是 10.005；
- 费用直接从现金扣，体现在 ``total_equity`` 的减少上，不摊进成本价。

两种口径都能自洽，但**不能混用**。这里选「费用不计入成本」的理由：费用是已发生
的现金流，摊进成本价会让 ``avg_cost × position`` 不等于任何一笔真实成交金额，
对账时反而更难解释。累计费用单独记在 :attr:`total_fee`，需要时可用于归因。

**已实现盈亏**

本类不计算已实现盈亏（卖出时的盈亏需要额外的 ``realized_pnl`` 账本与逐笔匹配
规则），属于后续步骤。现在能算的是：``cash``、``position``、``avg_cost``、
``market_value``、``total_equity``——这五项已足够还原任意时点的账户状态。

**精度**

金额与股数都用 ``float``，累计多次成交会有浮点误差。持仓归零判定用
:data:`POSITION_EPS` 容差，避免 ``1e-13`` 这类残渣把账户卡在「未平仓」状态。
不对金额做四舍五入：A 股最小价格变动 0.01 元、费用最小 0.01 元，四舍五入应由
Broker 在生成 FillEvent 前完成，记账层再舍入会与成交回报对不上。
"""

from __future__ import annotations

import math

from backtest.events import FillEvent

__all__ = ["Portfolio", "POSITION_EPS"]

#: 持仓归零容差（股）。浮点累加后 ``position`` 可能是 1e-13 而非 0，
#: 用它判定「已平仓」，避免把残渣当成持仓。
POSITION_EPS: float = 1e-9


def _require_number(value: object, field_name: str) -> float:
    """校验是有限实数（拒绝 bool / str / NaN / inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数值，收到 {type(value).__name__}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限数值，收到：{value!r}")
    return result


class Portfolio:
    """单标的现金账户，只支持多头。

    属性均为只读：``cash`` / ``position`` / ``avg_cost`` 只能通过 :meth:`on_fill`
    改变。这样做是为了让「持仓变负」「现金被凭空改写」这类错误在写入处就暴露，
    而不是等回测跑完再回头查一笔对不上的钱。

    Args:
        initial_cash: 初始现金（元），必须为正。
        allow_negative_cash: 是否允许现金为负。默认 ``False``——本项目不做融资
            融券，现金不足应当直接拒绝成交。仅在做「不留现金约束」的研究性回测时
            才置 ``True``，且结果不能用于实盘决策。

    Example:
        >>> from datetime import date
        >>> from backtest.events import FillEvent
        >>> p = Portfolio(100_000)
        >>> f = FillEvent("600519.SH", date(2025, 1, 2), 10.0, 1000, 1, fee=5.0, slippage=0.0)
        >>> p.on_fill(f)
        >>> round(p.cash, 2), p.position, p.avg_cost
        (89995.0, 1000.0, 10.0)
    """

    __slots__ = ("_initial_cash", "_cash", "_position", "_avg_cost", "_total_fee", "_allow_negative_cash")

    def __init__(self, initial_cash: float, *, allow_negative_cash: bool = False) -> None:
        cash = _require_number(initial_cash, "initial_cash")
        if cash <= 0:
            raise ValueError(f"initial_cash 必须为正数，收到：{initial_cash!r}")

        self._initial_cash = cash
        self._cash = cash
        self._position: float = 0.0
        self._avg_cost: float = 0.0
        self._total_fee: float = 0.0
        self._allow_negative_cash = bool(allow_negative_cash)

    # ------------------------------------------------------------------ 状态

    @property
    def initial_cash(self) -> float:
        """初始现金（元），只读，用于算累计收益率的分母。"""
        return self._initial_cash

    @property
    def cash(self) -> float:
        """可用现金（元）。买入后已扣除成交金额与费用。"""
        return self._cash

    @property
    def position(self) -> float:
        """持仓股数，恒 ``>= 0``（不支持做空；最小交易单位由 Broker 保证）。"""
        return self._position

    @property
    def avg_cost(self) -> float:
        """每股买入均价（元/股，不含费用）。空仓时为 ``0.0``。"""
        return self._avg_cost

    @property
    def total_fee(self) -> float:
        """累计已支付费用（元），用于「手续费侵蚀」归因。"""
        return self._total_fee

    @property
    def is_flat(self) -> bool:
        """是否空仓（用容差判定，见 :data:`POSITION_EPS`）。"""
        return self._position <= POSITION_EPS

    # ------------------------------------------------------------------ 记账

    def on_fill(self, fill: FillEvent) -> None:
        """把一笔成交计入账户。

        买入：``cash -= fill_price * quantity + fee``，持仓增加，``avg_cost`` 按
        成交价加权更新。

        卖出：``cash += fill_price * quantity - fee``，持仓减少，``avg_cost``
        **保持不变**（卖出不改变剩余股份的买入成本）；清仓时归零。

        注意 ``fill_price`` 已含滑点（见 ``events`` 模块），本方法不再处理滑点，
        否则会对同一笔滑点重复计价。

        Args:
            fill: 成交回报。

        Raises:
            TypeError: 传入的不是 :class:`FillEvent`。
            ValueError: 卖出数量超过持仓、买入后现金为负（未开启
                ``allow_negative_cash``）、或成交数量非正。
        """
        if not isinstance(fill, FillEvent):
            raise TypeError(f"on_fill 需要 FillEvent，收到 {type(fill).__name__}")

        quantity = _require_number(fill.quantity, "fill.quantity")
        price = _require_number(fill.fill_price, "fill.fill_price")
        fee = _require_number(fill.fee, "fill.fee")
        if quantity <= 0:
            raise ValueError(f"成交数量必须为正，收到：{fill.quantity!r}")
        if price <= 0:
            raise ValueError(f"成交价必须为正，收到：{fill.fill_price!r}")

        if fill.direction == 1:
            self._apply_buy(price, quantity, fee)
        elif fill.direction == -1:
            self._apply_sell(price, quantity, fee)
        else:  # pragma: no cover - FillEvent 已在构造时拦截
            raise ValueError(f"FillEvent.direction 只能是 1/-1，收到：{fill.direction!r}")

    def _apply_buy(self, price: float, quantity: float, fee: float) -> None:
        cash_after = self._cash - price * quantity - fee
        if cash_after < 0 and not self._allow_negative_cash:
            raise ValueError(
                f"现金不足：需 {price * quantity + fee:.2f} 元"
                f"（含费用 {fee:.2f}），仅有 {self._cash:.2f} 元。"
                f"本系统不做融资；如确需无现金约束，请显式传入 allow_negative_cash=True"
            )

        # 加权平均成本：只用成交价，fee 不进成本（见模块 docstring）
        new_position = self._position + quantity
        self._avg_cost = (self._avg_cost * self._position + price * quantity) / new_position
        self._position = new_position
        self._cash = cash_after
        self._total_fee += fee

    def _apply_sell(self, price: float, quantity: float, fee: float) -> None:
        if quantity - self._position > POSITION_EPS:
            raise ValueError(
                f"卖出数量 {quantity:g} 超过持仓 {self._position:g} 股；"
                f"本系统不支持做空（T+1 可用数量由 Broker 负责校验，此处只拦「卖超过持仓」）"
            )

        self._position -= quantity
        if self._position <= POSITION_EPS:
            # 浮点残渣归零，同时清掉成本价，避免空仓时 avg_cost 还是旧值
            self._position = 0.0
            self._avg_cost = 0.0

        self._cash += price * quantity - fee
        self._total_fee += fee

    # -------------------------------------------------------------- 估值口径

    def market_value(self, current_price: float) -> float:
        """持仓市值（元）＝ ``position × current_price``。空仓时恒为 0。"""
        price = _require_number(current_price, "current_price")
        if price <= 0:
            raise ValueError(f"current_price 必须为正数，收到：{current_price!r}")
        if self._position == 0.0:
            return 0.0
        return self._position * price

    def total_equity(self, current_price: float) -> float:
        """账户总权益（元）＝ ``cash + market_value``，即按市价盯市后的净值。"""
        return self._cash + self.market_value(current_price)

    def unrealized_pnl(self, current_price: float) -> float:
        """浮动盈亏（元）＝ ``(current_price - avg_cost) × position``，不含费用。"""
        if self._position == 0.0:
            return 0.0
        price = _require_number(current_price, "current_price")
        return (price - self._avg_cost) * self._position

    def snapshot(self, current_price: float) -> dict[str, float]:
        """当前账户状态快照，供逐日记录净值曲线。

        返回键固定，便于直接拼成 DataFrame 行；数值均为 ``float``。
        """
        mv = self.market_value(current_price)
        return {
            "cash": self._cash,
            "position": self._position,
            "avg_cost": self._avg_cost,
            "market_value": mv,
            "total_equity": self._cash + mv,
            "total_fee": self._total_fee,
        }

    # ------------------------------------------------------------------ 其它

    def __repr__(self) -> str:
        return (
            f"Portfolio(cash={self._cash:.2f}, position={self._position:g}, "
            f"avg_cost={self._avg_cost:.4f}, total_fee={self._total_fee:.2f})"
        )
