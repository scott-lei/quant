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
python -m pip install pytest pandas
```

依赖同时声明在 `pyproject.toml` 中；后续有了正式包目录后，可改为 `pip install -e .` 安装。

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
