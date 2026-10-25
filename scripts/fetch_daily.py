#!/usr/bin/env python
"""按日期区间批量抓取 A 股日线行情并落盘为 Parquet 分区。

用法::

    python scripts/fetch_daily.py 20250101 20250110

设计要点：
- 按**工作日**枚举（不判断交易日），非交易日 Tushare 返回空表，自然跳过。
- 已存在的分区直接跳过，因此该脚本可重复执行、可断点续跑。
- 每处理 ``PROGRESS_EVERY`` 个日期打印一次进度；日期之间 ``SLEEP_SECONDS`` 秒
  间隔，防止触发接口限频。
- 字段显式声明，不依赖 Tushare 的默认返回集合（默认集合可能缺少所需列）。

``pre_close`` 虽然在请求字段中，但 ``storage.save_daily`` 只保留契约内的 8 列，
该列会在落盘时被裁掉——保留它只是为了将来需要时不必再改请求。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 允许直接以脚本方式运行（无需 pip install -e .）：把 src 加入模块搜索路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data import storage  # noqa: E402
from data.tushare_client import MissingTokenError, pro_api  # noqa: E402

#: Tushare 请求字段，显式声明
FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,vol,amount"

#: 每处理多少个日期打印一次进度
PROGRESS_EVERY = 20

#: 相邻两次请求之间的间隔（秒）。120 积分档约 50 次/分钟，0.5 秒 = 120 次/分钟，
#: 对按交易日查询的单次调用足够；若遇到限频报错，可调大到 1.2。
SLEEP_SECONDS = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按区间批量抓取 A 股日线行情到 data/raw/daily/trade_date=YYYYMMDD/",
    )
    parser.add_argument("start_date", help="起始日期，格式 YYYYMMDD（含）")
    parser.add_argument("end_date", help="结束日期，格式 YYYYMMDD（含）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已存在的分区，强制重新抓取并覆盖",
    )
    args = parser.parse_args(argv)

    for name in ("start_date", "end_date"):
        raw = getattr(args, name).strip().replace("-", "")
        if len(raw) != 8 or not raw.isdigit():
            parser.error(f"{name} 格式应为 YYYYMMDD，收到：{getattr(args, name)!r}")
        setattr(args, name, raw)

    if args.start_date > args.end_date:
        parser.error(f"start_date ({args.start_date}) 不能晚于 end_date ({args.end_date})")
    return args


def partition_path(trade_date: str) -> Path:
    """返回某交易日对应的 Parquet 落盘路径。"""
    return storage.DAILY_ROOT / f"trade_date={trade_date}" / "data.parquet"


def weekdays(start: str, end: str) -> list[str]:
    """枚举 ``[start, end]`` 闭区间内的所有工作日（周一至周五），不判断交易日。"""
    cur = datetime.strptime(start, "%Y%m%d").date()
    last = datetime.strptime(end, "%Y%m%d").date()
    out: list[str] = []
    while cur <= last:
        if cur.weekday() < 5:  # 0=周一 … 4=周五
            out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dates = weekdays(args.start_date, args.end_date)

    if not dates:
        print(f"{args.start_date} ~ {args.end_date} 之间没有工作日，无事可做。")
        return 0

    print(
        f"区间 {args.start_date} ~ {args.end_date}：{len(dates)} 个工作日，"
        f"落盘目录 {storage.DAILY_ROOT}"
    )
    print(f"进度每 {PROGRESS_EVERY} 个日期打印一次，请求间隔 {SLEEP_SECONDS}s\n")

    try:
        pro = pro_api()
    except MissingTokenError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    n_skip = n_empty = n_saved = n_fail = 0
    rows_total = 0

    for i, d in enumerate(dates, start=1):
        path = partition_path(d)

        requested = False

        if path.exists() and not args.force:
            n_skip += 1
            status = "skip"
        else:
            requested = True
            try:
                df = pro.daily(trade_date=d, fields=FIELDS)
            except Exception as exc:  # 网络抖动、限频等：记录后继续，不中断整批
                n_fail += 1
                status = f"FAIL {type(exc).__name__}: {str(exc)[:60]}"
                df = None

            if df is not None:
                if df.empty:
                    n_empty += 1
                    status = "empty"
                else:
                    storage.save_daily(df, d)
                    n_saved += 1
                    rows_total += len(df)
                    status = f"save {len(df)} 行"

        print(f"[{i:>4}/{len(dates)}] {d} {status}")

        # 每 N 个日期打印一次汇总，长区间跑起来能看出速率
        if i % PROGRESS_EVERY == 0:
            print(
                f"  ---- 进度 {i}/{len(dates)} ({i / len(dates):.0%}) | "
                f"新增 {n_saved} 个分区 {rows_total} 行 | "
                f"跳过 {n_skip} | 空(非交易日) {n_empty} | 失败 {n_fail}"
            )

        # 限频只与「真的发出了请求」有关：跳过的日期没有请求，不需要等待。
        # 否则续跑几千天存量时会白等几十分钟。最后一个日期之后也不等待。
        if requested and i < len(dates):
            time.sleep(SLEEP_SECONDS)

    print(
        f"\n完成：共 {len(dates)} 个工作日 | 新增 {n_saved} 个分区（{rows_total} 行）| "
        f"跳过已存在 {n_skip} | 非交易日 {n_empty} | 失败 {n_fail}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
