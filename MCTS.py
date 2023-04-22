#!/usr/bin/env python
import pickle
import os
import collections
import copy
import datetime
import numpy as np

import chess
import torch
import encoding

# DO NOT CHANGE
MOVE_REPRESENTATION_SIZE = 4672 # 8 * 8 * 73

LEGAL_MOVES_CACHE = {}
MOVES_PAST_30_CACHE = {} # keeps memory usage down

WHITE_WINS = 0
BLACK_WINS = 0
DRAWS = 0

class UCTNode():
    def __init__(self, game, move, parent=None, c_puct=1.0):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([MOVE_REPRESENTATION_SIZE], dtype=np.float32)
        self.child_total_value = np.zeros([MOVE_REPRESENTATION_SIZE], dtype=np.float32)
        self.child_number_visits = np.zeros([MOVE_REPRESENTATION_SIZE], dtype=np.float32)
        self.action_idxes = []
        self.c_puct = c_puct

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return self.c_puct * np.sqrt(self.number_visits) * (abs(self.child_priors) / (1 + self.child_number_visits))

    def select_leaf(self):
        current = self
        while current.is_expanded:
            current.number_visits += 1
            current.total_value -= 1
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            legal_moves_mask = np.zeros([MOVE_REPRESENTATION_SIZE], dtype=bool)

            game_fen = self.game.fen()
            legal_moves = self.get_legal_moves(game_fen)
            for move in legal_moves:
                legal_moves_mask[encoding.encode_action(move, self.game)] = True

            bestmove = np.argmax((self.child_Q() + self.child_U()) * legal_moves_mask)
        return bestmove

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self, c_puct=1.0)
        return self.children[move]

    def get_legal_moves(self, game_fen):
        if game_fen in MOVES_PAST_30_CACHE:
            return MOVES_PAST_30_CACHE[game_fen]

        if game_fen in LEGAL_MOVES_CACHE:
            return LEGAL_MOVES_CACHE[game_fen]

        if len(self.game.move_stack) > 30:
            MOVES_PAST_30_CACHE[game_fen] = list(self.game.legal_moves)
            return MOVES_PAST_30_CACHE[game_fen]

        LEGAL_MOVES_CACHE[game_fen] = list(self.game.legal_moves)
        return LEGAL_MOVES_CACHE[game_fen]

    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32)+0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors

        # check in cache
        game_fen = self.game.fen()
        legal_moves = self.get_legal_moves(game_fen)

        for move in legal_moves: # get all legal moves for current board state s
            action_idxs.append(encoding.encode_action(move, self.game))

        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        if self.parent.parent is None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p

        # Parent-Q initialization
        if not isinstance(self.parent, DummyNode):
            parent_q_value = self.parent.child_Q()[self.move]
            for idx in action_idxs:
                self.child_total_value[idx] = parent_q_value

    def decode_n_move_pieces(self,board,encoded_move):
        move = encoding.decode_action(encoded_move, board)
        board.push(move)
        return board

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            if current.game.turn == chess.BLACK:
                current.total_value += (1*value_estimate)+1
            elif current.game.turn == chess.WHITE:
                current.total_value += (-1*value_estimate)-1

            current = current.parent

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

def UCT_search(game_state, num_reads, net, root=None):
    if root is None:
        root = UCTNode(game_state, move=None, parent=DummyNode(), c_puct=1.0)

    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = encoding.encode_board(leaf.game)
        encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()

        child_priors, value_estimate = net(encoded_s)

        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        if leaf.game.is_checkmate(): # if checkmate
            leaf.backup(value_estimate)
            continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)

    state_fen = game_state.fen()
    if state_fen not in LEGAL_MOVES_CACHE:
        LEGAL_MOVES_CACHE[state_fen] = list(game_state.legal_moves)

    legal_moves = [
        encoding.encode_action(move, game_state) for move in LEGAL_MOVES_CACHE[state_fen]
    ]
    legal_child_visits = root.child_number_visits[legal_moves]
    best_move_idx = np.argmax(legal_child_visits)
    best_move = legal_moves[best_move_idx]
    return best_move, root

def update_root_node(root, best_move):
    if (best_move in root.children) and (np.random.random_sample() < -1):
        new_root = root.children[best_move]
        new_root.parent = DummyNode()
        return new_root
    else:
        # Create a new root node if the child node does not exist
        return UCTNode(root.game, move=None, parent=DummyNode(), c_puct=1.0)

def decode_and_move(board, encoded_move):
    decoded_move = encoding.decode_action(encoded_move, board)
    board.push(decoded_move)
    return board

def get_policy(root):
    policy = np.zeros([MOVE_REPRESENTATION_SIZE], dtype=np.float32)
    for idx in np.where(root.child_number_visits!=0)[0]:
        policy[idx] = root.child_number_visits[idx]/root.child_number_visits.sum()
    return policy

def save_as_pickle(filename, data, iteration):
    folder = f"./datasets/iter{iteration}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    completeName = os.path.join(f"./datasets/iter{iteration}/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def MCTS_self_play(chessnet, num_games, cpu, iteration, num_reads, game_move_limit):
    global BLACK_WINS
    global WHITE_WINS
    global DRAWS

    for idxx in range(0, num_games):
        current_board = chess.Board()
        checkmate = False
        dataset = [] # to get state, policy, value for neural network training
        states = []
        value = 0
        root = None
        while checkmate is False and current_board.fullmove_number <= game_move_limit:

            if current_board.can_claim_threefold_repetition():
                break

            states.append(current_board.fen())
            board_state = copy.deepcopy(encoding.encode_board(current_board))
            best_move, root = UCT_search(current_board, num_reads, chessnet, root)
            current_board = decode_and_move(current_board, best_move) # decode move and move piece(s)
            policy = get_policy(root)

            root = update_root_node(root, best_move)

            dataset.append([board_state, policy])
            print(f"White Wins: {WHITE_WINS}")
            print(f"Draws: {DRAWS}")
            print(f"Black Wins: {BLACK_WINS}")
            print(f"Game: {idxx} CPU: {cpu}")
            print(f"Move Number: {current_board.fullmove_number}")
            print(f"Best move: {encoding.decode_action(best_move, current_board)}")
            print(current_board)
            print()
            if current_board.is_checkmate():
                if current_board.turn == chess.WHITE:
                    BLACK_WINS += 1
                    value = -1
                elif current_board.turn == chess.BLACK:
                    WHITE_WINS += 1
                    value = 1
                checkmate = True
            
            # Reset unlikely moves cache, keeps memory use reasonable (Otherwise 128 games could max out 32gb!)
            MOVES_PAST_30_CACHE.clear()


        if value == 0:
            DRAWS += 1

        dataset_p = []
        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        del dataset
        save_as_pickle("dataset_cpu%i_%i_%s" % (cpu,idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p, iteration)