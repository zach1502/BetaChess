#!/usr/bin/env python
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
matplotlib.use("Agg")

NUM_RESIDUAL_BLOCKS = 4
BATCH_SIZE = 32
PATIENCE = 128
PRINT_INTERVAL = 32
LEARNING_RATE = 0.03
NUM_EPOCHS = 2048

def load_settings():
    global NUM_RESIDUAL_BLOCKS, BATCH_SIZE, PATIENCE, PRINT_INTERVAL, LEARNING_RATE, NUM_EPOCHS

    with open('settings.json') as f:
        data = json.load(f)

    NUM_RESIDUAL_BLOCKS = data['training']['num_residual_blocks']
    BATCH_SIZE = data['training']['batch_size']
    PATIENCE = data['training']['patience']
    PRINT_INTERVAL = data['training']['print_interval']
    LEARNING_RATE = data['training']['learning_rate']
    NUM_EPOCHS = data['training']['num_epochs']

    print("Loaded settings from settings.json")
    print("NUM_RESIDUAL_BLOCKS:", NUM_RESIDUAL_BLOCKS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("PATIENCE:", PATIENCE)
    print("PRINT_INTERVAL:", PRINT_INTERVAL)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("NUM_EPOCHS:", NUM_EPOCHS)

def get_best_available_device():    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

DEVICE = get_best_available_device()

class BoardData(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].transpose(2, 0, 1), self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_size = 8 * 8 * 73
        self.conv1 = nn.Conv2d(20, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 20, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 8 * 8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 8 * 8 * 128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(NUM_RESIDUAL_BLOCKS)])
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in self.res_blocks:
            s = block(s)
        s = self.outblock(s)
        return s

class ChessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error

def load_some_previous_games(iteration):
    loaded_games = []
    total_loaded_files = 0
    for prev_iter in range(iteration, -1, -1):
        prev_iter_folder = f"datasets/iter{prev_iter}"
        all_files = [os.path.join(prev_iter_folder, file) for file in os.listdir(prev_iter_folder)]
        
        num_files_to_load = int(len(all_files) - len(all_files) * (iteration-prev_iter) * 0.25)
        total_loaded_files += num_files_to_load
        print(f"loading {num_files_to_load} files from iter{prev_iter}")

        if num_files_to_load < 1:
            break
        files_to_load = random.sample(all_files, num_files_to_load)

        for file in files_to_load:
            pickle_in = open(file, "rb")
            loaded_games.append(pickle.load(pickle_in))

    print(f"loaded {total_loaded_files} files")

    return np.concatenate(loaded_games, axis=0)

def train(net, dataset, iteration=0):
    load_settings()

    net.to(DEVICE)
    net.train()
    criterion = ChessLoss()
    criterion.to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)
    scaler = GradScaler()

    previous_games = load_some_previous_games(iteration)
    dataset = np.concatenate([dataset, previous_games], axis=0)

    split_idx = int(len(dataset) * 0.8)
    train_set = BoardData(dataset[:split_idx])
    val_set = BoardData(dataset[split_idx:])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    best_val_loss = float('inf')
    counter = 0

    train_losses = []
    val_losses = []

    print(f"Process ID: {os.getpid()} Training...")
    for epoch in range(NUM_EPOCHS):
        # Training loop
        net.train()
        total_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            state, policy, value = state.to(DEVICE).float(), policy.to(DEVICE).float(), value.to(DEVICE).float()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                policy_pred, value_pred = net(state)
                loss = criterion(value_pred[:, 0], value, policy_pred, policy)

            scaler.scale(loss).backward()
            total_train_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if (i + 1) % PRINT_INTERVAL == 0:
                print(f"Process ID: {os.getpid()} [Epoch: {epoch + 1}, {(i + 1) * BATCH_SIZE}/{len(train_set)} points] average loss per sample: {total_train_loss / ((i + 1) * BATCH_SIZE):.6f}")
                print(f"Policy: {policy[0].argmax().item()}, {policy_pred[0].argmax().item()}")
                print(f"Value: {value[0].item()}, {value_pred[0, 0].item()}")

        # Validation loop
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                state, policy, value = data
                state, policy, value = state.to(DEVICE).float(), policy.to(DEVICE).float(), value.to(DEVICE).float()

                policy_pred, value_pred = net(state)
                loss = criterion(value_pred[:, 0], value, policy_pred, policy)
                total_val_loss += loss.item()

                if (i + 1) % PRINT_INTERVAL == 0:
                    print(f"Process ID: {os.getpid()} [Epoch: {epoch + 1}, {(i + 1) * BATCH_SIZE}/{len(val_set)} points] average loss per sample: {total_val_loss / ((i + 1) * BATCH_SIZE):.6f}")
                    print(f"Policy: {policy[0].argmax().item()}, {policy_pred[0].argmax().item()}")
                    print(f"Value: {value[0].item()}, {value_pred[0, 0].item()}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            print("Early stopping counter reset")
            torch.save(net.state_dict(), f"model_data/current_net_trained8_iter{iteration+1}.pth.tar")
        else:
            counter += 1
            print(f'Early Stopping counter: {counter} out of {PATIENCE}')
            print(f"Best val loss: {best_val_loss}")
            print(f"Avg val loss: {avg_val_loss}")
            if counter >= PATIENCE:
                print("Early stopping")
                break

        scheduler.step()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epoch_range = range(1, len(train_losses) + 1)

    ax1.plot(epoch_range, train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss vs Epoch")
    ax1.legend()

    ax2.plot(epoch_range, val_losses, label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss vs Epoch")
    ax2.legend()

    # save plot
    plt.savefig(f"loss_vs_epoch_iter{iteration}.png")


def create_beta_net():
    load_settings()
    net = ChessNet()

    # get current file path
    current_path = os.path.dirname(os.path.realpath(__file__))

    # save model
    torch.save(
        {'model_state_dict': net.state_dict(),}, 
        os.path.join(current_path, "model_data", "current_net_trained8_iter0.pth.tar")
    )
