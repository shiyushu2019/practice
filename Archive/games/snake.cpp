// 贪吃蛇游戏 —— 终端版
// 编译: g++ -std=c++17 -O2 snake.cpp -o snake
// 运行: ./snake
// 操作: 方向键 / WASD 移动, p 暂停, q 退出

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <random>
#include <string>

#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

// ---------------- 终端控制 ----------------

struct RawTerminal {
    termios original;
    bool active = false;

    void enter() {
        tcgetattr(STDIN_FILENO, &original);
        termios raw = original;
        raw.c_lflag &= ~(ICANON | ECHO); // 关闭行缓冲和回显
        raw.c_cc[VMIN] = 0;              // 非阻塞读取
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        active = true;
    }

    ~RawTerminal() {
        if (active) tcsetattr(STDIN_FILENO, TCSAFLUSH, &original);
    }

    RawTerminal() = default;
    RawTerminal(const RawTerminal&) = delete;
    RawTerminal& operator=(const RawTerminal&) = delete;
};

// ---------------- 游戏逻辑 ----------------

enum class Direction { Up, Down, Left, Right };

struct Point {
    int x = 0, y = 0;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
};

constexpr int kInitialLength = 4; // 初始蛇身长度
constexpr double kPi = 3.14159265358979323846;
constexpr double kSuperFoodChance = 0.05; // 超级食物出现概率

class SnakeGame {
public:
    SnakeGame(int w, int h)
        : width_(w), height_(h),
          dir_(Direction::Right),
          pendingDir_(Direction::Right),
          rng_(std::random_device{}()),
          distX_(1, w - 2),
          distY_(1, h - 2) {
        // 初始化蛇身: 头朝右, 身体向左延伸出初始长度
        int headX = w / 2, headY = h / 2;
        for (int i = 0; i < kInitialLength; ++i) {
            snake_.push_back(Point{headX - i, headY});
        }
        placeFood();
    }

    // 每帧推进一步, 返回 false 表示游戏结束
    bool step() {
        dir_ = pendingDir_;
        Point head = snake_.front();
        switch (dir_) {
            case Direction::Up:    head.y -= 1; break;
            case Direction::Down:  head.y += 1; break;
            case Direction::Left:  head.x -= 1; break;
            case Direction::Right: head.x += 1; break;
        }

        // 撞墙
        if (head.x <= 0 || head.x >= width_ - 1 ||
            head.y <= 0 || head.y >= height_ - 1) {
            return false;
        }

        // 撞到自己 (尾部会缩走, 不算)
        bool eating = (head == food_);
        // 本步不吃且没有待增长 -> 尾部会被移除, 可以不参与碰撞
        bool tailWillMove = (!eating && growth_ == 0);
        auto bodyEnd = snake_.end();
        if (tailWillMove) --bodyEnd;
        for (auto it = snake_.begin(); it != bodyEnd; ++it) {
            if (*it == head) return false;
        }

        snake_.push_front(head);
        if (eating) {
            if (foodIsSuper_) {
                score_ += 30;
                growth_ += 3; // 超级食物额外增长
            } else {
                score_ += 10;
                growth_ += 1;
            }
            ++foodsEaten_;
            placeFood();
        }

        if (growth_ > 0) {
            --growth_;          // 增长期间保留尾部
        } else {
            snake_.pop_back();
        }
        return true;
    }

    void changeDirection(Direction d) {
        // 禁止直接反向
        if ((dir_ == Direction::Up    && d == Direction::Down)  ||
            (dir_ == Direction::Down  && d == Direction::Up)    ||
            (dir_ == Direction::Left  && d == Direction::Right) ||
            (dir_ == Direction::Right && d == Direction::Left)) {
            return;
        }
        pendingDir_ = d;
    }

    // 渲染整个画面
    void render() const {
        std::string frame;
        frame.reserve(static_cast<size_t>(width_ + 1) * height_ + 128);

        frame += "\x1b[H";          // 光标移到左上角
        frame += "\x1b[?25l";       // 隐藏光标

        // 顶边
        frame.append(width_, '#');
        frame += "\r\n";

        for (int y = 1; y < height_ - 1; ++y) {
            frame += '#';
            for (int x = 1; x < width_ - 1; ++x) {
                Point p{x, y};
                if (p == food_) {
                    frame += foodIsSuper_ ? '$' : '@';
                } else {
                    bool isSnake = false;
                    for (const auto& s : snake_) {
                        if (s == p) { isSnake = true; break; }
                    }
                    frame += isSnake ? 'O' : ' ';
                }
            }
            frame += '#';
            frame += "\r\n";
        }

        // 底边
        frame.append(width_, '#');
        frame += "\r\n";

        char info[128];
        std::snprintf(info, sizeof(info),
                      "得分:%d 长度:%zu  @+10/+1 $+30/+3  [方向键/WASD]移动 [p]暂停 [q]退出",
                      score_, snake_.size());
        frame += info;
        frame += "\x1b[K"; // 清除行尾残留

        std::fwrite(frame.data(), 1, frame.size(), stdout);
        std::fflush(stdout);
    }

    void renderGameOver() const {
        std::string frame;
        frame += "\x1b[H\x1b[?25l\x1b[2J";
        frame += "========== GAME OVER ==========\r\n";
        char info[96];
        std::snprintf(info, sizeof(info), "最终得分: %d    蛇长度: %zu\r\n", score_, snake_.size());
        frame += info;
        frame += "[r]重新开始  [q]退出\r\n";
        frame += "\x1b[?25h"; // 显示光标
        std::fwrite(frame.data(), 1, frame.size(), stdout);
        std::fflush(stdout);
    }

    int score() const { return score_; }
    int foodsEaten() const { return foodsEaten_; }

private:
    // 放置食物: 小概率为超级食物; 普通食物随得分从中心向外扩散
    void placeFood() {
        foodIsSuper_ = (unitDist_(rng_) < kSuperFoodChance);
        int guard = 0;
        do {
            food_ = foodIsSuper_ ? randomEdgePosition() : randomFoodPosition();
        } while (onSnake(food_) && ++guard < 2000);

        if (onSnake(food_)) { // 兜底: 全图扫描找一个空位
            for (int y = 1; y < height_ - 1; ++y)
                for (int x = 1; x < width_ - 1; ++x)
                    if (!onSnake(Point{x, y})) { food_ = Point{x, y}; return; }
        }
    }

    // 普通食物: 得分越高允许出现的半径越大 (由中心逐渐扩散到全图, 难度循序渐进)
    Point randomFoodPosition() {
        double cx = (width_ - 1) / 2.0;
        double cy = (height_ - 1) / 2.0;
        double progress = std::min(1.0, score_ / 400.0); // 400 分后铺满全图
        double maxR = std::max(cx, cy) + 1.0;
        double r = maxR * (0.20 + 0.80 * progress);
        double angle = angleDist_(rng_);
        double rr = r * std::sqrt(unitDist_(rng_)); // 圆盘内均匀采样
        int x = static_cast<int>(std::lround(cx + std::cos(angle) * rr));
        int y = static_cast<int>(std::lround(cy + std::sin(angle) * rr));
        return Point{std::clamp(x, 1, width_ - 2), std::clamp(y, 1, height_ - 2)};
    }

    // 超级食物: 贴在边界或四角, 位置刁钻
    Point randomEdgePosition() {
        if (unitDist_(rng_) < 0.4) { // 四角
            int x = (unitDist_(rng_) < 0.5) ? 1 : width_ - 2;
            int y = (unitDist_(rng_) < 0.5) ? 1 : height_ - 2;
            return Point{x, y};
        }
        if (unitDist_(rng_) < 0.5) { // 上/下边
            int y = (unitDist_(rng_) < 0.5) ? 1 : height_ - 2;
            return Point{distX_(rng_), y};
        }
        // 左/右边
        int x = (unitDist_(rng_) < 0.5) ? 1 : width_ - 2;
        return Point{x, distY_(rng_)};
    }

    bool onSnake(const Point& p) const {
        for (const auto& s : snake_) {
            if (s == p) return true;
        }
        return false;
    }

    int width_, height_;
    std::deque<Point> snake_;
    Point food_;
    bool foodIsSuper_ = false;
    int growth_ = 0;     // 待增长节数
    int foodsEaten_ = 0; // 已吃食物数量
    Direction dir_, pendingDir_;
    int score_ = 0;

    std::mt19937 rng_;
    std::uniform_int_distribution<int> distX_, distY_;
    std::uniform_real_distribution<double> unitDist_{0.0, 1.0};
    std::uniform_real_distribution<double> angleDist_{0.0, kPi * 2.0};
};

// ---------------- 键盘输入 ----------------

// 读取一个方向指令; 有输入返回 true, 无输入返回 false
bool readDirection(const RawTerminal&, Direction& out, bool& quit, bool& pause) {
    char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n <= 0) return false;

    if (c == '\x1b') { // 可能是方向键序列 ESC [ A/B/C/D
        char seq[2];
        ssize_t m = read(STDIN_FILENO, seq, 2);
        if (m >= 2 && seq[0] == '[') {
            switch (seq[1]) {
                case 'A': out = Direction::Up;    return true;
                case 'B': out = Direction::Down;  return true;
                case 'C': out = Direction::Right; return true;
                case 'D': out = Direction::Left;  return true;
                default:  return false;
            }
        }
        return false;
    }

    switch (c) {
        case 'w': case 'W': out = Direction::Up;    return true;
        case 's': case 'S': out = Direction::Down;  return true;
        case 'a': case 'A': out = Direction::Left;  return true;
        case 'd': case 'D': out = Direction::Right; return true;
        case 'p': case 'P': pause = true;  return false;
        case 'q': case 'Q': quit = true;   return false;
        default: return false;
    }
}

// ---------------- 难度菜单 ----------------

struct Difficulty {
    const char* name;
    const char* desc;
    int baseSpeed; // 初始速度 (毫秒/步)
    int minSpeed;  // 该难度最快速度上限
};

const Difficulty kDifficulties[3] = {
    {"简单", "慢速 · 适合休闲", 240, 120},
    {"中等", "中速 · 经典节奏", 180, 90},
    {"困难", "快速 · 挑战极限", 130, 65},
};

enum {
    KEY_NONE  = 0,
    KEY_UP    = 256,
    KEY_DOWN  = 257,
    KEY_ENTER = 10,
    KEY_QUIT  = 'q',
};

// 读取一个逻辑按键 (返回 KEY_* 或字符本身, 无输入返回 KEY_NONE)
int readKey() {
    char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n <= 0) return KEY_NONE;

    if (c == '\x1b') { // 方向键 ESC [ A/B
        char seq[2];
        ssize_t m = read(STDIN_FILENO, seq, 2);
        if (m >= 2 && seq[0] == '[') {
            if (seq[1] == 'A') return KEY_UP;
            if (seq[1] == 'B') return KEY_DOWN;
        }
        return KEY_NONE;
    }

    if (c == '\n' || c == '\r') return KEY_ENTER;
    if (c == 'q' || c == 'Q') return KEY_QUIT;
    return static_cast<unsigned char>(c);
}

// 显示难度菜单, 返回所选难度索引; 返回 -1 表示退出
int showDifficultyMenu() {
    int sel = 1; // 默认选中"中等"
    while (true) {
        std::string frame;
        frame += "\x1b[H\x1b[?25l";
        frame += "============ 贪吃蛇 ============\r\n\r\n";
        frame += "选择难度 (方向键选择, 回车确认):\r\n\r\n";

        for (int i = 0; i < 3; ++i) {
            frame += (i == sel) ? "  \x1b[7m> " : "    ";
            frame += kDifficulties[i].name;
            frame += "  ";
            frame += kDifficulties[i].desc;
            frame += (i == sel) ? "\x1b[0m" : "";
            frame += "\r\n";
        }

        frame += "\r\n[q] 退出\r\n";
        std::fwrite(frame.data(), 1, frame.size(), stdout);
        std::fflush(stdout);

        int key = readKey();
        if (key == KEY_UP)        sel = (sel + 2) % 3;
        else if (key == KEY_DOWN) sel = (sel + 1) % 3;
        else if (key == KEY_ENTER) return sel;
        else if (key == KEY_QUIT)  return -1;

        usleep(10000); // 防止忙等
    }
}

int main() {
    // 终端字符高约为宽的 2 倍以上, 加大宽高比让画面接近正方形
    const int W = 50, H = 20;

    RawTerminal raw;
    raw.enter();

    std::printf("\x1b[2J");   // 清屏
    std::printf("\x1b[H");    // 光标归位

    int diffIdx = showDifficultyMenu();
    std::printf("\x1b[0m\x1b[2J\x1b[H"); // 清除菜单残留
    if (diffIdx < 0) {
        std::printf("\x1b[?25h");
        std::printf("再见!\r\n");
        return 0;
    }

    const Difficulty& diff = kDifficulties[diffIdx];

    bool running = true;
    while (running) {
        SnakeGame game(W, H);
        bool alive = true;
        bool paused = false;
        int speedMs = diff.baseSpeed; // 初始速度 (毫秒/步)

        while (alive) {
            game.render();

            // 等待一个时间片, 期间持续读键盘
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(STDIN_FILENO, &fds);
            timeval tv{0, speedMs * 1000};
            int r = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);

            bool quit = false;
            if (r > 0) {
                Direction d;
                bool pause = false;
                if (readDirection(raw, d, quit, pause)) {
                    game.changeDirection(d);
                }
                if (pause) paused = !paused;
            }
            if (quit) { running = false; break; }

            if (paused) continue; // 暂停时冻结画面

            if (!game.step()) alive = false;

            // 每吃一个食物就加速一点, 逐渐加快, 但不会超过该难度上限
            speedMs = std::max(diff.minSpeed, diff.baseSpeed - game.foodsEaten() * 3);
        }

        if (!running) break;

        game.renderGameOver();

        // 等待重新开始或退出
        while (running) {
            char c;
            ssize_t n = read(STDIN_FILENO, &c, 1);
            if (n <= 0) continue;
            if (c == 'q' || c == 'Q') { running = false; break; }
            if (c == 'r' || c == 'R') break;
        }
    }

    std::printf("\x1b[?25h\x1b[2J\x1b[H"); // 恢复光标并清屏
    std::printf("感谢游玩, 再见!\r\n");
    return 0;
}
