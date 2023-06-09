#!/usr/bin/env python
import sys
import json
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

from beta_chess import ChessNet, train, create_beta_net
from MCTS import MCTS_self_play

NUM_PROCESSES_MCTS = 4
NUM_ITERATIONS = 400

def load_settings():
    global NUM_PROCESSES_MCTS, NUM_ITERATIONS

    with open("settings.json") as f:
        settings = json.load(f)

    NUM_PROCESSES_MCTS = settings["general"]["num_processes"]
    NUM_ITERATIONS = settings["general"]["num_iterations"]

    print("Loaded settings:")
    print(f"NUM_PROCESSES_MCTS: {NUM_PROCESSES_MCTS}")
    print(f"NUM_ITERATIONS: {NUM_ITERATIONS}")
    

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
        p1 = mp.Process(target=MCTS_self_play, args=(net, i, iteration))
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

    train(net, datasets, iteration)

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

    load_settings()

    for i in range(0, NUM_ITERATIONS+1):
        run_MCTS(i)
        run_net_training(i)
