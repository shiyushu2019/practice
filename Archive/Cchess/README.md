# GPT 中国象棋（Xiangqi / Cchess）

研究语言模型中国象棋能力的项目，结构与围棋版 `~/practice/go`、国际象棋版
`~/practice/chess` 保持一致：**棋盘规则 → 局面分析 → 模型选点 → 图形界面**。

## 与其它版本的对应关系

| 围棋版 (`go/`)          | 国际象棋版 (`chess/`)      | 中国象棋版 (`Cchess/`)        |
| ----------------------- | -------------------------- | ----------------------------- |
| `board.py` GoBoard      | `board.py` ChessBoard      | `board.py` CchessBoard（自实现规则） |
| `analyzer.py` KataGo    | `analyzer.py` Stockfish    | `analyzer.py` 内置 alpha-beta 搜索 |
| `player.py` 坐标记法    | `player.py` SAN 记法       | `player.py` 起点+终点记法     |
| `gui.py`                | `gui.py`                   | `gui.py` 9×10 交叉点棋盘      |

## 依赖

```bash
conda activate practice
pip install pygame openai
```

中国象棋规则由本包自行实现（`modules/board.py`），无需额外规则库；分析器
使用内置搜索，无需外部象棋引擎。

## 运行

```bash
conda activate practice
cd ~/practice/Cchess
export DEEPSEEK_API_KEY="sk-..."
python main.py
```

## 环境变量

| 变量                | 默认值                    | 说明                                 |
| ------------------- | ------------------------- | ------------------------------------ |
| `DEEPSEEK_API_KEY`  | （必填）                  | DeepSeek / OpenAI 兼容 API 密钥      |
| `DEEPSEEK_MODEL`    | `deepseek-flash`          | 模型名，也兼容 `OPENAI_MODEL`        |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址                 |
| `CC_SEARCH_DEPTH`   | `2`                       | 内置分析器搜索深度（1/2/3）          |

## 着法记法

- 9 列 × 10 行棋盘，红方先行。
- 列用 `a-i`（从左到右），行用 `0-9`（从上到下，`0`=黑方底线、`9`=红方底线）。
- 着法记作“起点+终点”，例如 `b7e7`（红炮从 b7 平到 e7）、`h9g7`（红马 h9 进 g7）。

棋子记号（大写=红方，小写=黑方）：`K`帅/`k`将、`A`仕/`a`士、`E`相/`e`象、
`H`马/`h`马、`R`车/`r`车、`C`炮/`c`炮、`P`兵/`p`卒。

## 玩法

- 人类执红、AI 执黑（可在 `main.py` 中把 `human_color` 改为 `2` 交换）。
- 点击己方棋子选中，再点击目标点走子；右键取消选择。
- 「分析局面」用内置搜索评估当前局面并给出候选着法，同时对比上一手 GPT
  走法并给出质量评级（好棋 / 一般 / 较差 / 很差）。
- 「AI 走棋」手动让 AI 走一手；「回退」悔棋；「编辑棋盘」手动摆子；
  「思考链」打开 DeepSeek 思考内容的流式窗口。

## 目录结构

```
Cchess/
├── main.py              # 入口：组装依赖并启动 GUI
├── README.md
└── modules/
    ├── __init__.py
    ├── board.py         # CchessBoard：中国象棋规则
    ├── config.py        # 环境变量配置
    ├── analyzer.py      # CchessAnalyzer：内置 alpha-beta 搜索
    ├── player.py        # 提示词构建与模型请求
    └── gui.py           # pygame 图形界面
```
