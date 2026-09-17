# GPT 国际象棋（Chess）

研究语言模型国际象棋能力的项目，由围棋项目 `~/practice/go` 移植而来。
架构保持一致：**棋盘规则 → 引擎分析 → 模型选点 → 图形界面**。

## 与围棋版的对应关系

| 围棋版 (`go/`)            | 国际象棋版 (`chess/`)          | 说明                          |
| ------------------------- | ------------------------------ | ----------------------------- |
| `modules/board.py` GoBoard | `modules/board.py` ChessBoard  | 规则由 `python-chess` 提供    |
| `modules/analyzer.py` KataGo | `modules/analyzer.py` ChessAnalyzer | KataGo GTP → Stockfish UCI |
| `modules/player.py`       | `modules/player.py`            | 坐标记法 → SAN 代数记法       |
| `modules/gui.py`          | `modules/gui.py`               | 19 路棋盘 → 8×8 棋盘          |
| `modules/config.py`       | `modules/config.py`            | 环境变量配置                  |

## 依赖

```bash
conda activate practice
pip install python-chess pygame openai
```

- `python-chess`：国际象棋规则引擎（合法着法、将军、终局判定）。
- `Stockfish`（可选）：更强的局面评估。未安装时自动退回内置的子力评估，
  应用仍然可以完整运行。安装方法：

  ```bash
  sudo apt-get install stockfish   # Debian / Ubuntu
  ```

## 运行

```bash
conda activate practice
cd ~/practice/chess
export DEEPSEEK_API_KEY="sk-..."
# 可选：
export STOCKFISH_BIN="/usr/games/stockfish"
python main.py
```

> 注意：本项目目录名与国际象棋库 `chess` 同名，请务必先 `cd ~/practice/chess`
> 再从目录内启动，否则会被 `python-chess` 库顶层的 `chess` 包遮蔽。

## 环境变量

| 变量                  | 默认值                    | 说明                                     |
| --------------------- | ------------------------- | ---------------------------------------- |
| `DEEPSEEK_API_KEY`    | （必填）                  | DeepSeek / OpenAI 兼容 API 密钥          |
| `DEEPSEEK_MODEL`      | `deepseek-flash`          | 模型名，也兼容 `OPENAI_MODEL`            |
| `DEEPSEEK_BASE_URL`   | `https://api.deepseek.com` | OpenAI 兼容接口地址                     |
| `STOCKFISH_BIN`       | 自动查找常见路径          | Stockfish 可执行文件                     |
| `STOCKFISH_DEPTH`     | `15`                      | 默认搜索深度                             |
| `STOCKFISH_THREADS`   | `1`                       | Stockfish 线程数                         |
| `STOCKFISH_HASH`      | `16`                      | Stockfish 哈希表大小（MB）               |

## 玩法

- 人类执白、AI 执黑（可在 `main.py` 中把 `human_color` 改为 `2` 交换）。
- 点击己方棋子选中，再点击目标格走子；右键取消选择。
- 兵升变自动升为后。
- 「分析局面」用 Stockfish 评估当前局面并给出候选着法，同时对比上一手
  GPT 走法并给出质量评级（好棋 / 一般 / 较差 / 很差）。
- 「AI 走棋」手动让 AI 走一手；「回退」悔棋；「编辑棋盘」手动摆子；
  「思考链」打开 DeepSeek 思考内容的流式窗口。

## 目录结构

```
chess/
├── main.py              # 入口：组装依赖并启动 GUI
├── README.md
└── modules/
    ├── __init__.py
    ├── board.py         # ChessBoard：python-chess 封装
    ├── config.py        # 环境变量配置
    ├── analyzer.py      # ChessAnalyzer：Stockfish UCI 封装 + 内置评估
    ├── player.py        # 提示词构建与模型请求
    └── gui.py           # pygame 图形界面
```
