# quant-system

A 股个人量化分析与交易系统：单人开发，日线到分钟线频率，用于跑通「回测 → 模拟盘 → 小资金实盘」的验证闭环。

## 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install pytest pandas tushare
```

现在已有正式包目录 `src/`，推荐直接用可编辑安装（同时装齐全部依赖）：

```bash
python -m pip install -e .
```

## 配置 Tushare Token

token 只从环境变量 `TUSHARE_TOKEN` 读取，不写进任何入库文件。

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

然后把 `https://tushare.pro/user/token` 拿到的 token 填进 `.env` 的 `TUSHARE_TOKEN=` 后面。`.env` 已被 `.gitignore` 忽略，不会进 Git。

```python
from data.tushare_client import pro_api

pro = pro_api()                       # 缺 token 会抛 MissingTokenError
df = pro.daily(ts_code="600519.SH", start_date="20260101")
```

## 运行测试

```bash
python -m pytest
```

或运行单个文件：

```bash
python -m pytest tests/test_smoke.py -v
```

## 文档

- `docs/prd.md`：一页纸 PRD
- `docs/adr/`：架构决策记录
