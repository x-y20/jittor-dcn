import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

# ==== PyTorch====
import torch
import torch.nn as tnn
import torch.optim as toptim

try:
    from mmcv.ops import DeformConv2d as TorchDeformConv2d
except ImportError:
    print("mmcv not installed, skipping PyTorch training.")
    TorchDeformConv2d = None


class TorchDeformableCNN(tnn.Module):
    def __init__(self):
        super().__init__()
        self.offset = tnn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.deform = TorchDeformConv2d(1, 8, kernel_size=3, padding=1)
        self.relu = tnn.ReLU()
        self.pool = tnn.MaxPool2d(2)
        self.fc = tnn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        offset = self.offset(x)
        x = self.relu(self.deform(x, offset))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def torch_train(X, y):
    model = TorchDeformableCNN()
    optimizer = toptim.SGD(model.parameters(), lr=0.01)
    criterion = tnn.CrossEntropyLoss()
    batch_size = 10
    losses = []
    metrics = []

    log_path = "compare_with_pytorch/pytorch_log.txt"
    model_path = "compare_with_pytorch/pytorch_model.pth"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        pass  

    for epoch in range(10):
        start_time = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0
        grad_norm_sum = 0.0

        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size]).float()
            yb = torch.tensor(y[i:i+batch_size], dtype=torch.long)
            batch_size_current = xb.size(0)
            total += batch_size_current

            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad.norm().item() **2)
            grad_norm_sum += (grad_norm** 0.5)

            optimizer.step()
            epoch_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == yb).sum().item()

        avg_loss = epoch_loss / (len(X) // batch_size)
        accuracy = correct / total
        avg_grad_norm = grad_norm_sum / (len(X) // batch_size)
        epoch_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 **2

        log_str = f"[PyTorch] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Grad Norm: {avg_grad_norm:.4f}, Time: {epoch_time:.2f}s, Memory: {memory_usage:.1f}MB"
        print(log_str)  

        with open(log_path, "a") as f:
            f.write(log_str + "\n")

        losses.append(avg_loss)
        metrics.append((avg_loss, accuracy, avg_grad_norm, epoch_time, memory_usage))

    torch.save(model.state_dict(), model_path)

    return losses, metrics


# ==== Jittor ====
import jittor as jt
from jittor import nn
from deform_conv import DeformConv2d

jt.flags.use_cuda = 0

class JittorDeformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.deform = DeformConv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.Pool(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def execute(self, x):
        x = self.relu(self.deform(x))
        x = self.pool(x)
        x = x.reshape([x.shape[0], -1])
        return self.fc(x)


def jittor_train(X, y):
    model = JittorDeformNet()
    optimizer = nn.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    batch_size = 10
    losses = []
    metrics = []

    log_path = "compare_with_pytorch/jittor_log.txt"
    model_path = "compare_with_pytorch/jittor_model.pth"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        pass  

    for epoch in range(10):
        start_time = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0
        grad_norm_sum = 0.0

        for i in range(0, len(X), batch_size):
            xb = jt.array(X[i:i + batch_size])
            yb = jt.array(y[i:i + batch_size])
            batch_size_current = xb.shape[0]
            total += batch_size_current

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.backward(loss)

            grad_norm = 0.0
            for p in model.parameters():
                grad = p.opt_grad(optimizer)
                if grad is not None:
                    norm = grad.norm()
                    if norm.size != 1:
                        norm = norm.sum()
                    grad_norm += (norm.item() **2)
            grad_norm_sum += (grad_norm** 0.5)

            optimizer.step()
            epoch_loss += loss.item()

            pred_np = pred.numpy()
            predicted = np.argmax(pred_np, axis=1)
            correct += (predicted == y[i:i + batch_size]).sum().item()

        avg_loss = epoch_loss / (len(X) // batch_size)
        accuracy = correct / total
        avg_grad_norm = grad_norm_sum / (len(X) // batch_size)
        epoch_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 **2

        log_str = f"[Jittor] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Grad Norm: {avg_grad_norm:.4f}, Time: {epoch_time:.2f}s, Memory: {memory_usage:.1f}MB"
        print(log_str)  #

        with open(log_path, "a") as f:
            f.write(log_str + "\n")

        losses.append(avg_loss)
        metrics.append((avg_loss, accuracy, avg_grad_norm, epoch_time, memory_usage))

    jt.save(model.state_dict(), model_path)

    return losses, metrics


def plot_losses(jittor_losses, torch_losses, save_path="compare_with_pytorch/loss_curve_compare.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    if jittor_losses:
        plt.plot(jittor_losses, label="Jittor DCN", color='red', marker='o')
    if torch_losses:
        plt.plot(torch_losses, label="PyTorch DCN", color='green', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Deformable Conv: Jittor vs PyTorch")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"[Info] Saved loss comparison to {save_path}")



def main():
    np.random.seed(42)
    X = np.random.rand(100, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, size=(100,))

    jittor_losses, jittor_metircs = jittor_train(X, y)

    if TorchDeformConv2d:
        torch_losses, torch_metrics = torch_train(X, y)
    else:
        torch_losses = []
        print("[Warning] mmcv not installed, skipping PyTorch loss curve.")

    plot_losses(jittor_losses, torch_losses)


if __name__ == "__main__":
    main()
