import tkinter as tk
import json
import random
import os
import time

# ---------------- Hyperparameters ----------------
ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.2
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.99995
SAVE_INTERVAL = 500
Q_FILE = "player_qtable.json"
STATS_FILE = "player_stats.json"

# ---------------- Q-Learning Player ----------------
class QLearningPlayer:
    def __init__(self, symbol, q=None):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"
        self.q = q if q is not None else {}

    def get_state(self, board):
        return "".join(board)

    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == "_"]

    def choose_action(self, board, explore=True):
        state = self.get_state(board)
        actions = self.get_valid_actions(board)
        if not actions:
            return None

        if explore and random.random() < EPSILON:
            return random.choice(actions)

        q_values = [self.q.get(f"{state}:{a}", 0) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, old_state, action, reward, next_state, next_actions):
        key = f"{old_state}:{action}"
        old_q = self.q.get(key, 0)
        next_q = max((self.q.get(f"{next_state}:{a}", 0) for a in next_actions), default=0)
        new_q = old_q + ALPHA * (reward + GAMMA * next_q - old_q)
        self.q[key] = new_q


# ---------------- Helper Functions ----------------
def check_win(board, symbol):
    lines = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    return any(all(board[i] == symbol for i in line) for line in lines)

def is_draw(board):
    return "_" not in board

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            print(f"⚠️  Failed to load {path}, resetting.")
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------- Computer Opponent (Rule-Based) ----------------
def computer_move(board, symbol, player_symbol):
    empty = [i for i, cell in enumerate(board) if cell == "_"]
    if not empty:
        return None

    # Try to win
    for i in empty:
        board[i] = symbol
        if check_win(board, symbol):
            board[i] = "_"
            return i
        board[i] = "_"

    # Block opponent
    for i in empty:
        board[i] = player_symbol
        if check_win(board, player_symbol):
            board[i] = "_"
            return i
        board[i] = "_"

    # Take corners
    for i in [0, 2, 6, 8]:
        if board[i] == "_":
            return i

    # Take center
    if board[4] == "_":
        return 4

    # Take sides
    for i in [1, 3, 5, 7]:
        if board[i] == "_":
            return i

    return random.choice(empty)


# ---------------- Trainer ----------------
class Trainer:
    def __init__(self, gui):
        self.gui = gui
        self.q_table = load_json(Q_FILE, {})
        self.stats = load_json(STATS_FILE, {"games": 0, "wins": 0, "draws": 0})
        self.running = False

    def reset_board(self):
        self.board = ["_"] * 9
        for b in self.gui.buttons:
            b.config(text="", bg="SystemButtonFace")

    def update_gui_stats(self):
        s = self.stats
        self.gui.stats_label.config(
            text=f"Games: {s['games']} | Wins: {s['wins']} | Draws: {s['draws']} | Epsilon: {EPSILON:.3f}"
        )

    def play_one_game(self):
        global EPSILON
        player = QLearningPlayer("X", self.q_table)
        self.reset_board()
        history = []

        while True:
            # --- AI's Turn ---
            state = player.get_state(self.board)
            action = player.choose_action(self.board, explore=True)
            if action is None:
                break

            self.board[action] = player.symbol
            self.gui.update_button(action, player.symbol)
            history.append((state, action))
            self.gui.root.update()

            time.sleep(1.0 / self.gui.speed_var.get())

            if check_win(self.board, player.symbol):
                # AI wins
                for s, a in reversed(history):
                    player.update_q(s, a, 1, player.get_state(self.board), [])
                self.stats["wins"] += 1
                self.stats["games"] += 1
                break

            if is_draw(self.board):
                for s, a in history:
                    player.update_q(s, a, 0.2, player.get_state(self.board), [])
                self.stats["draws"] += 1
                self.stats["games"] += 1
                break

            # --- Computer's Turn ---
            comp_action = computer_move(self.board, "O", player.symbol)
            if comp_action is None:
                break
            self.board[comp_action] = "O"
            self.gui.update_button(comp_action, "O")
            self.gui.root.update()

            time.sleep(1.0 / self.gui.speed_var.get())

            if check_win(self.board, "O"):
                # AI loses
                last_state, last_action = history[-1]
                player.update_q(last_state, last_action, -1, player.get_state(self.board), [])
                self.stats["games"] += 1
                break

            if is_draw(self.board):
                for s, a in history:
                    player.update_q(s, a, 0.2, player.get_state(self.board), [])
                self.stats["draws"] += 1
                self.stats["games"] += 1
                break

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        if self.stats["games"] % SAVE_INTERVAL == 0:
            save_json(Q_FILE, self.q_table)
            save_json(STATS_FILE, self.stats)
        self.update_gui_stats()

    def loop(self):
        if not self.running:
            return
        self.play_one_game()
        self.gui.root.after(10, self.loop)

    def start_training(self):
        if self.running:
            return
        self.running = True
        self.gui.status_label.config(text="Training vs Computer...")
        self.loop()

    def stop_training(self):
        self.running = False
        save_json(Q_FILE, self.q_table)
        save_json(STATS_FILE, self.stats)
        self.gui.status_label.config(text="Paused. Q-table and stats saved.")


# ---------------- GUI ----------------
class TicTacToeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe Q-Learning vs Computer")
        self.root.geometry("420x620")

        self.buttons = []
        for i in range(9):
            b = tk.Button(self.root, text="", width=8, height=4, font=("Arial", 20))
            b.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(b)

        self.stats_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.stats_label.grid(row=3, column=0, columnspan=3, pady=10)

        self.status_label = tk.Label(self.root, text="Press Start to train vs computer.", font=("Arial", 12))
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)

        tk.Label(self.root, text="Speed Control (Games/sec):").grid(row=5, column=0, columnspan=3)
        self.speed_var = tk.DoubleVar(value=10)
        self.speed_slider = tk.Scale(
            self.root, from_=1, to=1000, orient=tk.HORIZONTAL,
            variable=self.speed_var, length=300
        )
        self.speed_slider.grid(row=6, column=0, columnspan=3, pady=5)

        self.start_btn = tk.Button(self.root, text="▶ Start", command=self.start_training)
        self.start_btn.grid(row=7, column=0, columnspan=3, pady=5)

        self.stop_btn = tk.Button(self.root, text="⏸ Stop", command=self.stop_training)
        self.stop_btn.grid(row=8, column=0, columnspan=3, pady=5)

        self.trainer = Trainer(self)
        self.trainer.update_gui_stats()

    def update_button(self, index, symbol):
        color = "blue" if symbol == "X" else "red"
        self.buttons[index].config(text=symbol, fg=color)

    def start_training(self):
        self.trainer.start_training()

    def stop_training(self):
        self.trainer.stop_training()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TicTacToeGUI()
    app.run()
