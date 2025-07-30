# jittor-dcn

环境配置（环境配置部分省略了诸如VMware的安装、Ubuntu的安装、conda的安装等）：

jittor-dcn的实现基于python3.8（最好是3.7-3.10，否则可能无法兼容jittor），以下为本项目所需的依赖库：
pip install numpy matplotlib

pip install jittor

pip install torch torchvision

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html

由于jittor在windows操作系统下兼容性不好，所以本项目在VMware pro的ubuntu 24.10虚拟机上开发。

在终端输入：python -c "import jittor"

不报错代表jittor配置正确

常见问题（运行下面的代码即可）：
1.Jittor 启动过程中，无法成功识别 cc路径或者存在g++版本问题，如“AssertionError: assert jit_utils.cc”
sudo bash -c 'echo "deb http://archive.ubuntu.com/ubuntu focal main universe" >> /etc/apt/sources.list'
sudo apt update
sudo apt install g++-10
export cc_path=/usr/bin/g++-10
python3 -c "import jittor"

2.libstdc++版本问题，如“ImportError: libstdc++.so.6: version `GLIBCXX_3.4.30' ”
conda install -n jittor-env -c conda-forge libstdcxx-ng
export cc_path=/usr/bin/g++-10
python3 -c "import jittor"



数据准备脚本：
import os
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms

def prepare_mnist_subset(output_dir="data/sample_dataset", n=100):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    images = []
    labels = []

    for i in range(n):
        img, label = mnist[i]
        images.append(np.array(img).astype(np.float32))
        labels.append(label)

    X = np.stack(images, axis=0)  # [N, 1, 28, 28]
    y = np.array(labels)          # [N]

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"Saved {n} samples to {output_dir}")

if __name__ == '__main__':
    prepare_mnist_subset()


训练脚本：
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


测试脚本：
import os
import numpy as np
import matplotlib.pyplot as plt
import time

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
       output = self.fc(x)
       if isinstance(output, tuple):
           return output[0]
       return output


def extract_tensor(data):
   while isinstance(data, tuple):
       data = data[0]
   return data


def load_model(model, model_path, framework="jittor"):
   if not os.path.exists(model_path):
       raise FileNotFoundError(f"Model file not exists: {model_path}")

   try:
       if framework == "jittor":
           model.load_state_dict(jt.load(model_path))
       elif framework == "pytorch":
           model.load_state_dict(torch.load(model_path, map_location="cpu"))
       print(f"Successfully loaded {framework} model: {model_path}")
       return model
   except Exception as e:
       raise RuntimeError(f"Model loading failed: {str(e)}")


def test(framework="jittor", model_path=None):
   data_dir = "data/sample_dataset"
   if not os.path.exists(data_dir):
       os.makedirs(data_dir, exist_ok=True)
       np.save(os.path.join(data_dir, "X.npy"), np.random.randn(100, 1, 28, 28).astype(np.float32))
       np.save(os.path.join(data_dir, "y.npy"), np.random.randint(0, 10, size=100))

   X_test = np.load(os.path.join(data_dir, "X.npy"))[:20]
   y_test = np.load(os.path.join(data_dir, "y.npy"))[:20]
   print(f"Test data loaded: {len(X_test)} samples")

   if framework == "jittor":
       model = JittorDeformNet()
       default_path = "compare_with_pytorch/jittor_model.pth"
   elif framework == "pytorch" and TorchDeformConv2d is not None:
       model = TorchDeformableCNN()
       default_path = "compare_with_pytorch/pytorch_model.pth"
   else:
       raise ValueError(f"Unsupported framework: {framework}")

   model_path = model_path or default_path
   model = load_model(model, model_path, framework)
   model.eval()

   start_time = time.time()
   correct = 0
   total = len(X_test)

   try:
       if framework == "jittor":
           with jt.no_grad():
               inputs = jt.array(X_test)
               outputs = model(inputs)
               outputs = extract_tensor(outputs)
               pred_classes = outputs.argmax(dim=1)
               pred_classes = extract_tensor(pred_classes)
               pred_np = pred_classes.numpy() if hasattr(pred_classes, 'numpy') else np.array(pred_classes)

       elif framework == "pytorch":
           with torch.no_grad():
               inputs = torch.tensor(X_test).float()
               outputs = model(inputs)
               pred_classes = outputs.argmax(dim=1)
               pred_np = pred_classes.numpy()

       correct = (pred_np == y_test).sum()
       accuracy = correct / total
       infer_time = time.time() - start_time

       print(f"\n===== {framework} Test Results =====")
       print(f"Test samples: {total}")
       print(f"Correct predictions: {correct}")
       print(f"Test accuracy: {accuracy:.2%}")
       print(f"Inference time: {infer_time:.4f}s")
       print(f"Average time per sample: {infer_time / total:.4f}s")
       return accuracy

   except Exception as e:
       print(f"Test failed: {str(e)}")
       return 0.0


if __name__ == "__main__":
   test(framework="jittor")
   if TorchDeformConv2d is not None:
       test(framework="pytorch")

   X_test = np.load("data/sample_dataset/X.npy")[:5]
   y_test = np.load("data/sample_dataset/y.npy")[:5]

   jt_model = JittorDeformNet()
   jt_model = load_model(jt_model, "compare_with_pytorch/jittor_model.pth", "jittor")
   jt_model.eval()

   with jt.no_grad():
       jt_inputs = jt.array(X_test)
       jt_outputs = jt_model(jt_inputs)
       jt_outputs = extract_tensor(jt_outputs)
       jt_preds_tensor = jt_outputs.argmax(dim=1)
       jt_preds_tensor = extract_tensor(jt_preds_tensor)
       jt_preds = jt_preds_tensor.numpy() if hasattr(jt_preds_tensor, 'numpy') else np.array(jt_preds_tensor)

   plt.figure(figsize=(10, 4))
   for i in range(5):
       plt.subplot(1, 5, i + 1)
       plt.imshow(X_test[i, 0], cmap="gray")
       plt.title(f"True: {y_test[i]}\nPred: {jt_preds[i]}")
       plt.axis("off")
   plt.tight_layout()
   plt.savefig("test_predictions.png")
   print("\nPrediction visualization saved to: test_predictions.png")


实验与性能log：
[Jittor] Epoch 1, Loss: 2.6616, Accuracy: 0.0600, Grad Norm: 28.1755, Time: 0.69s, Memory: 338.6MB
[Jittor] Epoch 2, Loss: 2.5128, Accuracy: 0.0900, Grad Norm: 23.2172, Time: 0.08s, Memory: 338.6MB
[Jittor] Epoch 3, Loss: 2.4398, Accuracy: 0.1200, Grad Norm: 20.4726, Time: 0.08s, Memory: 338.6MB
[Jittor] Epoch 4, Loss: 2.3932, Accuracy: 0.1200, Grad Norm: 18.6160, Time: 0.10s, Memory: 338.6MB
[Jittor] Epoch 5, Loss: 2.3594, Accuracy: 0.1600, Grad Norm: 17.2321, Time: 0.09s, Memory: 338.7MB
[Jittor] Epoch 6, Loss: 2.3330, Accuracy: 0.1600, Grad Norm: 16.1765, Time: 0.07s, Memory: 338.7MB
[Jittor] Epoch 7, Loss: 2.3112, Accuracy: 0.1700, Grad Norm: 15.3542, Time: 0.09s, Memory: 338.7MB
[Jittor] Epoch 8, Loss: 2.2926, Accuracy: 0.1700, Grad Norm: 14.7245, Time: 0.10s, Memory: 338.7MB
[Jittor] Epoch 9, Loss: 2.2762, Accuracy: 0.1900, Grad Norm: 14.2483, Time: 0.07s, Memory: 338.7MB
[Jittor] Epoch 10, Loss: 2.2612, Accuracy: 0.2000, Grad Norm: 13.8976, Time: 0.10s, Memory: 338.7MB

[PyTorch] Epoch 1, Loss: 2.3680, Accuracy: 0.1100, Grad Norm: 4.4646, Time: 0.57s, Memory: 358.9MB
[PyTorch] Epoch 2, Loss: 2.3273, Accuracy: 0.1200, Grad Norm: 4.3160, Time: 0.27s, Memory: 358.9MB
[PyTorch] Epoch 3, Loss: 2.3002, Accuracy: 0.1300, Grad Norm: 4.2278, Time: 0.27s, Memory: 358.9MB
[PyTorch] Epoch 4, Loss: 2.2739, Accuracy: 0.1500, Grad Norm: 4.1466, Time: 0.31s, Memory: 358.9MB
[PyTorch] Epoch 5, Loss: 2.2480, Accuracy: 0.1800, Grad Norm: 4.0750, Time: 0.28s, Memory: 358.9MB
[PyTorch] Epoch 6, Loss: 2.2223, Accuracy: 0.1900, Grad Norm: 4.0121, Time: 0.28s, Memory: 358.9MB
[PyTorch] Epoch 7, Loss: 2.1967, Accuracy: 0.2100, Grad Norm: 3.9573, Time: 0.27s, Memory: 359.0MB
[PyTorch] Epoch 8, Loss: 2.1711, Accuracy: 0.2300, Grad Norm: 3.9099, Time: 0.29s, Memory: 359.0MB
[PyTorch] Epoch 9, Loss: 2.1453, Accuracy: 0.2400, Grad Norm: 3.8699, Time: 0.27s, Memory: 359.0MB
[PyTorch] Epoch 10, Loss: 2.1193, Accuracy: 0.3400, Grad Norm: 3.8364, Time: 0.27s, Memory: 359.0MB


jittor与pytorch对齐训练曲线：
<img width="563" height="421" alt="image" src="https://github.com/user-attachments/assets/5a1fb4a3-0881-4161-808d-3e6814095128" />
