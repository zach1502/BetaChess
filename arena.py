import os
import torch
import chess
from beta_chess import ChessNet
from MCTS import UCT_search
import encoding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name: str):
    net = ChessNet()
    net.eval()
    net.to(DEVICE)

    model_path = f"model_data/{model_name}"
    if os.path.exists(model_path):
        print(f"Loading model {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        try:
            net.load_state_dict(checkpoint["model_state_dict"])
        except:
            net.load_state_dict(checkpoint)
        return net
    else:
        print(f"Model {model_path} not found")
        return None

def init_model_stats():
    model_stats = {}

    if os.path.exists("model_stats.txt"):
        with open("model_stats.txt", "r") as f:
            for line in f.readlines():
                line = line.split("|")
                model_name = line[0].split(":")[1].strip()
                wins = int(line[1].split(":")[1].strip())
                losses = int(line[2].split(":")[1].strip())
                draws = int(line[3].split(":")[1].strip())
                elo = float(line[4].split(":")[1].strip())
                model_stats[model_name] = {
                    "model_name": model_name,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "elo": elo
                }
        return model_stats

    for file in os.listdir("model_data"):
        model_stats[file] = {
            "model_name": file,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "elo": 1000
        }

    return model_stats

def save_model_stats(model_stats):
    with open("model_stats.txt", "w") as f:
        for model_name, stats in model_stats.items():
            f.write(f"Name: {model_name} | Wins: {stats['wins']} | Losses: {stats['losses']} | Draws: {stats['draws']} | Elo: {stats['elo']}\n")

def update_model_stats(model_stats, model_white, model_black, result):
    # update elo and stats
    if result == "1-0":
        model_stats[model_white]["wins"] += 1
        model_stats[model_black]["losses"] += 1
    elif result == "0-1":
        model_stats[model_white]["losses"] += 1
        model_stats[model_black]["wins"] += 1
    else:
        model_stats[model_white]["draws"] += 1
        model_stats[model_black]["draws"] += 1

    # update elo
    elo_white = model_stats[model_white]["elo"]
    elo_black = model_stats[model_black]["elo"]
    expected_white = 1 / (1 + 10 ** ((elo_black - elo_white) / 400))
    expected_black = 1 / (1 + 10 ** ((elo_white - elo_black) / 400))

    if result == "1-0":
        model_stats[model_white]["elo"] = elo_white + 32 * (1 - expected_white)
        model_stats[model_black]["elo"] = elo_black + 32 * (0 - expected_black)
    elif result == "0-1":
        model_stats[model_white]["elo"] = elo_white + 32 * (0 - expected_white)
        model_stats[model_black]["elo"] = elo_black + 32 * (1 - expected_black)
    else:
        model_stats[model_white]["elo"] = elo_white + 32 * (0.5 - expected_white)
        model_stats[model_black]["elo"] = elo_black + 32 * (0.5 - expected_black)
    
    save_model_stats(model_stats)

    return model_stats


def compete(model_white: ChessNet, model_black: ChessNet):
    board = chess.Board()

    #while not board.is_game_over():
    for _ in range(10):
        if board.turn:
            best_move, _ = UCT_search(board, 512, model_white, root=None)
        else:
            best_move, _ = UCT_search(board, 512, model_black, root=None)

        decoded_move = encoding.decode_action(best_move, board)
        board.push(decoded_move)

        print(board)
        print()

    mock_come = chess.Outcome(termination=chess.Termination.INSUFFICIENT_MATERIAL, winner=None)

    return mock_come #board.outcome()

def arena():
    model_stats = init_model_stats()
    model_names = list(model_stats.keys())
    
    for i, i_file_name in enumerate(model_names):
        for j, j_file_name in enumerate(model_names):
            if i == j:
                continue
                
            print(f"Competing {i} (white) vs {j} (black)...")
            model_white = load_model(i_file_name)
            model_black = load_model(j_file_name)

            if model_white is None or model_black is None:
                continue
                
            print(f"Competing {i_file_name} (white) vs {j_file_name} (black)...")
            result = compete(model_white, model_black)


            model_stats = update_model_stats(model_stats, i_file_name, j_file_name, result.result())

if __name__ == "__main__":
    arena()