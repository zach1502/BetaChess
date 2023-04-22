#!/usr/bin/env python
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
matplotlib.use("Agg")

NUM_RESIDUAL_BLOCKS = 19
BATCH_SIZE = 16
PATIENCE = 20
PRINT_INTERVAL = 16

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
        for block in range(NUM_RESIDUAL_BLOCKS):
            setattr(self, f"res_{block}", ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(NUM_RESIDUAL_BLOCKS):
            s = getattr(self, f"res_{block}")(s)
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

def train(net, dataset, epoch_start=0, epoch_stop=20, cpu=0, iteration=0):
    torch.manual_seed(cpu)
    net.train()
    criterion = ChessLoss()
    criterion.cuda()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)
    scaler = GradScaler()

    split_idx = int(len(dataset) * 0.8)
    train_set = BoardData(dataset[:split_idx])
    val_set = BoardData(dataset[split_idx:])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    best_val_loss = float('inf')
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epoch_start, epoch_stop):
        # Training loop
        net.train()
        total_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()

            with torch.cuda.amp.autocast():
                policy_pred, value_pred = net(state)
                loss = criterion(value_pred[:, 0], value, policy_pred, policy)

            scaler.scale(loss).backward()
            total_train_loss += scaler.scale(loss).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if (i + 1) % PRINT_INTERVAL == 0:
                print(f"Process ID: {os.getpid()} [Epoch: {epoch + 1}, {(i + 1) * BATCH_SIZE}/{len(train_set)} points] average loss per batch: {total_train_loss / (i + 1):.3f}")
                print(f"Policy: {policy[0].argmax().item()}, {policy_pred[0].argmax().item()}")
                print(f"Value: {value[0].item()}, {value_pred[0, 0].item()}")

        # Validation loop
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                state, policy, value = data
                state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()

                policy_pred, value_pred = net(state)
                loss = criterion(value_pred[:, 0], value, policy_pred, policy)
                total_val_loss += loss.item()

                if (i + 1) % PRINT_INTERVAL == 0:
                    print(f"Process ID: {os.getpid()} [Epoch: {epoch + 1}, {(i + 1) * BATCH_SIZE}/{len(val_set)} points] average loss per batch: {total_val_loss / (i + 1):.3f}")
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
            torch.save(net.state_dict(), f"model_data/current_net_trained8_iter{iteration+1}.pth.tar")
        else:
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {PATIENCE}')
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
    plt.savefig(f"loss_vs_epoch_cpu{cpu}_iter{iteration}.png")


def create_beta_net():
    net = ChessNet()

    # save model
    torch.save({
            'model_state_dict': net.state_dict(),
            }, "current_net_trained8_iter0.pth.tar")