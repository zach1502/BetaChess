#!/usr/bin/env python
import sys
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

from beta_chess import ChessNet, train, create_beta_net
from MCTS import MCTS_self_play


# recommend powers of 2
# 6 processes, 30 games each = recc 32 GB RAM

# Training parameters
NUM_PROCESSES_MCTS = 8
NUM_PROCESSES_TRAIN = 2
NUM_GAMES_TO_SELF_PLAY = 96
NUM_ITERATIONS = 40
NUM_EPOCHS = 2048

# MCTS parameters
NUM_READS = 512 # ideally should be a lot higher, but takes too long
GAME_MOVE_LIMIT = 120

def get_best_available_device():    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

DEVICE = get_best_available_device()

def run_MCTS(iteration):
    # Runs MCTS
    net_to_play=f"current_net_trained8_iter{iteration}.pth.tar"
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    sys.stdout.write("#############################\n")
    sys.stdout.write("#      MCTS USING CUDA      #\n")
    sys.stdout.write("#############################\n")

    net.to(DEVICE)
    net.share_memory()
    net.eval()

    current_net_filename = os.path.join("./model_data/",\
                                    net_to_play)
    checkpoint = checkpoint = torch.load(current_net_filename, map_location=DEVICE)

    net.load_state_dict(checkpoint['model_state_dict'])


    processes1 = []
    for i in range(NUM_PROCESSES_MCTS):
        p1 = mp.Process(target=MCTS_self_play, args=(net, NUM_GAMES_TO_SELF_PLAY, i, iteration, NUM_READS, GAME_MOVE_LIMIT))
        p1.start()
        processes1.append(p1)
    for p1 in processes1:
        p1.join()


def run_net_training(iteration):
    # Runs Net training
    net_to_train = f"current_net_trained8_iter{iteration}.pth.tar"
    save_as = f"current_net_trained8_iter{iteration+1}.pth.tar"
    # gather data
    data_path = f"./datasets/iter{iteration}/"
    datasets = []

    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    datasets = np.array(datasets)

    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    print("#############################")
    print("#    TRAINING USING CUDA    #")
    print("#############################")
    net.cuda()
    net.share_memory()
    net.train()

    current_net_filename = os.path.join("./model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['model_state_dict'])

    processes2 = []
    for i in range(NUM_PROCESSES_TRAIN):
        p2 = mp.Process(target=train,args=(net, datasets, 0, NUM_EPOCHS, i, iteration))
        p2.start()
        processes2.append(p2)
    for p2 in processes2:
        p2.join()
    # save results
    torch.save({'model_state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                    save_as))


if __name__=="__main__":
    torch.backends.cudnn.benchmark = True

    if not os.path.exists("./datasets/"):
        os.mkdir("./datasets/")
    if not os.path.exists("./model_data/"):
        os.mkdir("./model_data/")
        create_beta_net()


    for i in range(1, NUM_ITERATIONS+1):
        # run_MCTS(i)
        run_net_training(i)
