# jittor-dcn

## 使用教程

将本项目的脚本文件置于同一目录下，先运行prepare_data.py生成数据，然后依次train.py，test.py

## 环境配置

环境配置部分省略了诸如VMware的安装、Ubuntu的安装、conda的安装等。

jittor-dcn的实现基于python3.8（最好是3.7-3.10，否则可能无法兼容jittor），以下为本项目所需的依赖库：

```bash
pip install numpy matplotlib
pip install jittor
pip install torch torchvision
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
```

由于jittor在windows操作系统下兼容性不好，所以本项目在VMware pro的ubuntu 24.10虚拟机上开发。

在终端输入：
```bash
python -c "import jittor"
```
不报错代表jittor配置正确

### 常见问题

运行下面的代码即可：

1. Jittor 启动过程中，无法成功识别 cc路径或者存在g++版本问题，如"AssertionError: assert jit_utils.cc"
```bash
sudo bash -c 'echo "deb http://archive.ubuntu.com/ubuntu focal main universe" >> /etc/apt/sources.list'
sudo apt update
sudo apt install g++-10
export cc_path=/usr/bin/g++-10
python3 -c "import jittor"
```

2. libstdc++版本问题，如"ImportError: libstdc++.so.6: version `GLIBCXX_3.4.30' "
```bash
conda install -n jittor-env -c conda-forge libstdcxx-ng
export cc_path=/usr/bin/g++-10
python3 -c "import jittor"
```

## 数据准备脚本

```python
import os
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import random


def create_detection_image(mnist_dataset, img_size=128, num_objects=1):
    canvas = np.zeros((1, img_size, img_size), dtype=np.float32)
    boxes = []
    labels = []

    for _ in range(num_objects):
        idx = random.randint(0, len(mnist_dataset) - 1)
        digit_img, digit_label = mnist_dataset[idx]
        digit_img = np.array(digit_img).squeeze()

        max_x = img_size - 28
        max_y = img_size - 28
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        canvas[0, y:y + 28, x:x + 28] = np.maximum(canvas[0, y:y + 28, x:x + 28], digit_img)

        x1 = x / img_size
        y1 = y / img_size
        x2 = (x + 28) / img_size
        y2 = (y + 28) / img_size
        boxes.append([x1, y1, x2, y2])
        labels.append(digit_label)

    return canvas, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def prepare_mnistdet(output_dir="data/mnistdet", n_train=500, n_test=100):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    train_images = []
    train_boxes = []
    train_labels = []

    for i in range(n_train):
        img, boxes, labels = create_detection_image(mnist, num_objects=1)
        train_images.append(img)
        train_boxes.append(boxes)
        train_labels.append(labels)

    np.save(os.path.join(output_dir, "train_images.npy"), np.array(train_images))
    np.save(os.path.join(output_dir, "train_boxes.npy"), np.array(train_boxes, dtype=object), allow_pickle=True)
    np.save(os.path.join(output_dir, "train_labels.npy"), np.array(train_labels, dtype=object), allow_pickle=True)

    test_images = []
    test_boxes = []
    test_labels = []

    for i in range(n_test):
        img, boxes, labels = create_detection_image(mnist, num_objects=1)
        test_images.append(img)
        test_boxes.append(boxes)
        test_labels.append(labels)

    np.save(os.path.join(output_dir, "test_images.npy"), np.array(test_images))
    np.save(os.path.join(output_dir, "test_boxes.npy"), np.array(test_boxes, dtype=object), allow_pickle=True)
    np.save(os.path.join(output_dir, "test_labels.npy"), np.array(test_labels, dtype=object), allow_pickle=True)

    print(f"Created MNISTDet dataset with {n_train} training and {n_test} test samples (single object per image)")


if __name__ == '__main__':
    prepare_mnistdet()
```

## 训练脚本

```python
import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as tnn
import torch.optim as toptim
import torch.nn.functional as F


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def compute_map(all_preds, all_gts, iou_threshold=0.5):
    classes = set([gt[0] for gt in all_gts] + [pred[0] for pred in all_preds])
    aps = []

    for cls in classes:
        preds = [p for p in all_preds if p[0] == cls]
        gts = [g for g in all_gts if g[0] == cls]
        n_gts = len(gts)
        if n_gts == 0:
            continue

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
        tp = np.zeros(len(preds_sorted))
        fp = np.ones(len(preds_sorted))
        gt_matched = [False] * n_gts

        for i, (_, pred_bbox, _) in enumerate(preds_sorted):
            max_iou = 0
            match_idx = -1
            for j, (_, gt_bbox) in enumerate(gts):
                if not gt_matched[j]:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        match_idx = j
            if max_iou >= iou_threshold and match_idx != -1:
                tp[i] = 1
                fp[i] = 0
                gt_matched[match_idx] = True

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall = cum_tp / n_gts
        ap = 0
        prev_recall = 0
        for p, r in zip(precision, recall):
            ap += p * (r - prev_recall)
            prev_recall = r
        aps.append(ap)

    return np.mean(aps) if aps else 0.0

class TorchEDNetDetection(tnn.Module):
    def __init__(self, num_classes=10, groups=2):
        super().__init__()
        self.conv1 = tnn.Conv2d(1, 16, 3, 1, 1, groups=1)
        self.bn1 = tnn.BatchNorm2d(16)
        self.relu = tnn.ReLU(inplace=True)

        self.conv2 = tnn.Conv2d(16, 32, 3, 2, 1, groups=groups)
        self.bn2 = tnn.BatchNorm2d(32)

        self.conv3 = tnn.Conv2d(32, 64, 3, 2, 1, groups=groups)
        self.bn3 = tnn.BatchNorm2d(64)

        self.conv4 = tnn.Conv2d(64, 128, 3, 2, 1, groups=groups)
        self.bn4 = tnn.BatchNorm2d(128)

        self.conv5 = tnn.Conv2d(128, 256, 3, 2, 1, groups=groups)
        self.bn5 = tnn.BatchNorm2d(256)

        self.gap = tnn.AdaptiveAvgPool2d(1)
        self.fc_cls = tnn.Linear(256, num_classes)
        self.fc_bbox = tnn.Linear(256, 4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls = self.fc_cls(x)
        bbox = torch.sigmoid(self.fc_bbox(x))
        return cls, bbox


def torch_train_detection(data_dir):
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_boxes = np.load(os.path.join(data_dir, "train_boxes.npy"), allow_pickle=True)
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"), allow_pickle=True)

    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    test_boxes = np.load(os.path.join(data_dir, "test_boxes.npy"), allow_pickle=True)
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"), allow_pickle=True)

    model = TorchEDNetDetection(groups=2)
    optimizer = toptim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    cls_criterion = tnn.CrossEntropyLoss()
    bbox_criterion = tnn.SmoothL1Loss()

    def torch_smooth_l1_loss(pred, target, beta=1.0):
        diff = torch.abs(pred - target)
        mask = diff < beta
        loss = torch.where(mask, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        return loss.mean()

    batch_size = 10
    losses = []

    log_path = "compare_with_pytorch/pytorch_detection_log.txt"
    model_path = "compare_with_pytorch/pytorch_detection_model.pth"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        pass

    for epoch in range(10):
        start_time = time.time()
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_total_loss = 0.0
        correct = 0
        total = 0

        indices = np.random.permutation(len(train_images))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            batch_cls_targets = []
            batch_bbox_targets = []

            for idx in batch_indices:
                batch_images.append(train_images[idx])
                if len(train_labels[idx]) > 0:
                    batch_cls_targets.append(train_labels[idx][0])
                    batch_bbox_targets.append(list(train_boxes[idx][0].astype(float)))
                else:
                    batch_cls_targets.append(0)
                    batch_bbox_targets.append([0.25, 0.25, 0.75, 0.75])

            bbox_np = np.array(batch_bbox_targets, dtype=np.float32)
            bbox_targets = torch.tensor(bbox_np).float()

            xb = torch.tensor(np.array(batch_images)).float()
            cls_targets = torch.tensor(batch_cls_targets, dtype=torch.long)

            optimizer.zero_grad()
            cls_out, bbox_out = model(xb)

            cls_loss = cls_criterion(cls_out, cls_targets)
            bbox_loss = torch_smooth_l1_loss(bbox_out, bbox_targets)
            total_loss = cls_loss + 5.0 * bbox_loss

            total_loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_bbox_loss += bbox_loss.item()
            epoch_total_loss += total_loss.item()

            _, predicted = torch.max(cls_out.data, 1)
            correct += (predicted == cls_targets).sum().item()
            total += len(cls_targets)

        model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for i in range(len(test_images)):
                img = torch.tensor(test_images[i:i + 1]).float()
                cls_out, bbox_out = model(img)
                pred_cls = torch.argmax(cls_out, dim=1).item()
                pred_bbox = bbox_out[0].numpy()
                score = F.softmax(cls_out, dim=1)[0, pred_cls].item()

                all_preds.append((pred_cls, pred_bbox, score))
                if len(test_labels[i]) > 0:
                    all_gts.append((test_labels[i][0], test_boxes[i][0]))

        mAP = compute_map(all_preds, all_gts)
        model.train()

        n_batches = (len(indices) + batch_size - 1) // batch_size
        avg_cls_loss = epoch_cls_loss / n_batches
        avg_bbox_loss = epoch_bbox_loss / n_batches
        avg_total_loss = epoch_total_loss / n_batches
        epoch_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 ** 2

        log_str = f"[PyTorch] Epoch {epoch + 1}, Total Loss: {avg_total_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, BBox Loss: {avg_bbox_loss:.4f}, mAP: {mAP:.4f}, Time: {epoch_time:.2f}s, Memory: {memory_usage:.1f}MB"
        print(log_str)

        with open(log_path, "a") as f:
            f.write(log_str + "\n")

        losses.append(avg_total_loss)

    torch.save(model.state_dict(), model_path)
    return losses


import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0


class JittorEDNetDetection(nn.Module):
    def __init__(self, num_classes=10, groups=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, groups=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1, groups=groups)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, groups=groups)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1, groups=groups)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1, groups=groups)
        self.bn5 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = nn.Linear(256, num_classes)
        self.fc_bbox = nn.Linear(256, 4)

    def execute(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = x.reshape([x.shape[0], -1])
        cls = self.fc_cls(x)
        bbox = jt.sigmoid(self.fc_bbox(x))
        return cls, bbox


def jittor_train_detection(data_dir):
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_boxes = np.load(os.path.join(data_dir, "train_boxes.npy"), allow_pickle=True)
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"), allow_pickle=True)

    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    test_boxes = np.load(os.path.join(data_dir, "test_boxes.npy"), allow_pickle=True)
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"), allow_pickle=True)

    model = JittorEDNetDetection(groups=2)
    optimizer = nn.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    cls_criterion = nn.CrossEntropyLoss()

    def jittor_smooth_l1_loss(pred, target, beta=1.0):
        diff = jt.abs(pred - target)
        mask = diff < beta
        loss = jt.where(mask, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        return loss.mean()

    def bbox_criterion(pred, target):
        return jittor_smooth_l1_loss(pred, target, beta=1.0)

    batch_size = 10
    losses = []

    log_path = "compare_with_pytorch/jittor_detection_log.txt"
    model_path = "compare_with_pytorch/jittor_detection_model.pth"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        pass

    for epoch in range(10):
        start_time = time.time()
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_total_loss = 0.0
        correct = 0
        total = 0

        indices = np.random.permutation(len(train_images))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            batch_cls_targets = []
            batch_bbox_targets = []

            for idx in batch_indices:
                batch_images.append(train_images[idx])
                if len(train_labels[idx]) > 0:
                    batch_cls_targets.append(train_labels[idx][0])
                    batch_bbox_targets.append(list(train_boxes[idx][0]))
                else:
                    batch_cls_targets.append(0)
                    batch_bbox_targets.append([0.25, 0.25, 0.75, 0.75])

            bbox_np = np.array(batch_bbox_targets, dtype=np.float32)
            bbox_targets = jt.array(bbox_np)

            xb = jt.array(np.array(batch_images))
            cls_targets = jt.array(batch_cls_targets)

            optimizer.zero_grad()
            cls_out, bbox_out = model(xb)

            cls_loss = cls_criterion(cls_out, cls_targets)
            bbox_loss = bbox_criterion(bbox_out, bbox_targets)
            total_loss = cls_loss + 5.0 * bbox_loss

            optimizer.backward(total_loss)
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_bbox_loss += bbox_loss.item()
            epoch_total_loss += total_loss.item()

            pred_classes = cls_out.argmax(dim=1)
            if isinstance(pred_classes, tuple):
                pred_classes = pred_classes[0]
            correct += (pred_classes.numpy() == np.array(batch_cls_targets)).sum()
            total += len(batch_cls_targets)

        model.eval()
        all_preds = []
        all_gts = []
        with jt.no_grad():
            for i in range(len(test_images)):
                img = jt.array(test_images[i:i + 1])
                cls_out, bbox_out = model(img)
                pred_cls = cls_out.argmax(dim=1)[0].item()
                pred_bbox = bbox_out[0].numpy()
                score = jt.nn.softmax(cls_out, dim=1)[0, pred_cls].item()

                all_preds.append((pred_cls, pred_bbox, score))
                if len(test_labels[i]) > 0:
                    all_gts.append((test_labels[i][0], test_boxes[i][0]))

        mAP = compute_map(all_preds, all_gts)
        model.train()

        n_batches = (len(indices) + batch_size - 1) // batch_size
        avg_cls_loss = epoch_cls_loss / n_batches
        avg_bbox_loss = epoch_bbox_loss / n_batches
        avg_total_loss = epoch_total_loss / n_batches
        accuracy = correct / total
        epoch_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 ** 2

        log_str = f"[Jittor] Epoch {epoch + 1}, Total Loss: {avg_total_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, BBox Loss: {avg_bbox_loss:.4f}, mAP: {mAP:.4f}, Time: {epoch_time:.2f}s, Memory: {memory_usage:.1f}MB"
        print(log_str)

        with open(log_path, "a") as f:
            f.write(log_str + "\n")

        losses.append(avg_total_loss)

    jt.save(model.state_dict(), model_path)
    return losses


def plot_losses(jittor_losses, torch_losses, save_path="compare_with_pytorch/detection_loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    if jittor_losses:
        plt.plot(jittor_losses, label="Jittor Detection", color='red', marker='o')
    if torch_losses:
        plt.plot(torch_losses, label="PyTorch Detection", color='green', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Object Detection Training: Jittor vs PyTorch")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"[Info] Saved loss comparison to {save_path}")


def main():
    data_dir = "data/mnistdet"
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}. Please run prepare_data.py first.")
        return

    jittor_losses = jittor_train_detection(data_dir)
    torch_losses = torch_train_detection(data_dir)

    plot_losses(jittor_losses, torch_losses)


if __name__ == "__main__":
    main()
```

## 测试脚本

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import torch
import torch.nn as tnn

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0

def load_model(model, model_path, framework):
    if framework == "jittor":
        model.load(model_path)
    elif framework == "pytorch":
        model.load_state_dict(torch.load(model_path))
    return model

class TorchEDNetDetection(tnn.Module):
    def __init__(self, num_classes=10, groups=2):
        super().__init__()
        self.conv1 = tnn.Conv2d(1, 16, 3, 1, 1, groups=1)
        self.bn1 = tnn.BatchNorm2d(16)
        self.relu = tnn.ReLU(inplace=True)

        self.conv2 = tnn.Conv2d(16, 32, 3, 2, 1, groups=groups)
        self.bn2 = tnn.BatchNorm2d(32)

        self.conv3 = tnn.Conv2d(32, 64, 3, 2, 1, groups=groups)
        self.bn3 = tnn.BatchNorm2d(64)

        self.conv4 = tnn.Conv2d(64, 128, 3, 2, 1, groups=groups)
        self.bn4 = tnn.BatchNorm2d(128)

        self.conv5 = tnn.Conv2d(128, 256, 3, 2, 1, groups=groups)
        self.bn5 = tnn.BatchNorm2d(256)

        self.gap = tnn.AdaptiveAvgPool2d(1)
        self.fc_cls = tnn.Linear(256, num_classes)
        self.fc_bbox = tnn.Linear(256, 4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls = self.fc_cls(x)
        bbox = torch.sigmoid(self.fc_bbox(x))
        return cls, bbox


class JittorEDNetDetection(nn.Module):
    def __init__(self, num_classes=10, groups=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, groups=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1, groups=groups)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, groups=groups)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1, groups=groups)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1, groups=groups)
        self.bn5 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = nn.Linear(256, num_classes)
        self.fc_bbox = nn.Linear(256, 4)

    def execute(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = x.reshape([x.shape[0], -1])
        cls = self.fc_cls(x)
        bbox = jt.sigmoid(self.fc_bbox(x))
        return cls, bbox


def visualize_detection(image, true_boxes, true_labels, pred_box, pred_label, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image[0], cmap='gray')
    ax1.set_title('Ground Truth')
    for box, label in zip(true_boxes, true_labels):
        x1, y1, x2, y2 = box * 128
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 5, str(label), color='green', fontsize=12)

    ax2.imshow(image[0], cmap='gray')
    ax2.set_title('Prediction')
    x1, y1, x2, y2 = pred_box * 128
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    ax2.text(x1, y1 - 5, str(pred_label), color='red', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def test_detection(framework="jittor"):
    data_dir = "data/mnistdet"
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}. Please run prepare_data.py first.")
        return

    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    test_boxes = np.load(os.path.join(data_dir, "test_boxes.npy"), allow_pickle=True)
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"), allow_pickle=True)

    n_samples = min(5, len(test_images))
    random_indices = np.random.choice(len(test_images), size=n_samples, replace=False)

    if framework == "jittor":
        model = JittorEDNetDetection()
        model_path = "compare_with_pytorch/jittor_detection_model.pth"
        if os.path.exists(model_path):
            model = load_model(model, model_path, "jittor")
            print(f"Loaded trained model from {model_path}")
        else:
            print(f"Warning: No trained model found at {model_path}. Using random initialization.")
            for p in model.parameters():
                jt.init.gauss_(p, 0, 0.02)
    else:
        model = TorchEDNetDetection()
        model_path = "compare_with_pytorch/pytorch_detection_model.pth"
        if os.path.exists(model_path):
            model = load_model(model, model_path, "pytorch")
            print(f"Loaded trained model from {model_path}")
        else:
            print(f"Warning: No trained model found at {model_path}. Using random initialization.")
            for p in model.parameters():
                tnn.init.normal_(p, 0, 0.02)

    model.eval()

    for i, idx in enumerate(random_indices):
        image = test_images[idx:idx + 1]
        true_boxes = test_boxes[idx]
        true_labels = test_labels[idx]

        if framework == "jittor":
            with jt.no_grad():
                inputs = jt.array(image)
                cls_out, bbox_out = model(inputs)
                pred_cls_tensor = cls_out.argmax(dim=1)
                if isinstance(pred_cls_tensor, tuple):
                    pred_cls_tensor = pred_cls_tensor[0]
                pred_cls = pred_cls_tensor.numpy()[0]
                pred_bbox = bbox_out.numpy()[0]
        else:
            with torch.no_grad():
                inputs = torch.tensor(image).float()
                cls_out, bbox_out = model(inputs)
                pred_cls = cls_out.argmax(dim=1).numpy()[0]
                pred_bbox = bbox_out.numpy()[0]

        visualize_detection(
            image[0],
            true_boxes,
            true_labels,
            pred_bbox,
            pred_cls,
            f"detection_result_{i}.png"
        )

        print(f"Sample {i}: Predicted class={pred_cls}, bbox={pred_bbox}")
        print(f"          True classes={true_labels}, bboxes={true_boxes[0]}")


if __name__ == "__main__":
    test_detection(framework="jittor")
```

## 实验与性能log

### Jittor训练日志
```
[Jittor] Epoch 1, Total Loss: 2.4170, Cls Loss: 2.2828, BBox Loss: 0.0268, mAP: 0.0000, Time: 39.18s, Memory: 363.8MB
[Jittor] Epoch 2, Total Loss: 2.2762, Cls Loss: 2.1453, BBox Loss: 0.0262, mAP: 0.0000, Time: 4.97s, Memory: 364.2MB
[Jittor] Epoch 3, Total Loss: 2.0799, Cls Loss: 1.9497, BBox Loss: 0.0260, mAP: 0.0132, Time: 4.48s, Memory: 365.2MB
[Jittor] Epoch 4, Total Loss: 1.8026, Cls Loss: 1.6750, BBox Loss: 0.0255, mAP: 0.0000, Time: 4.53s, Memory: 365.2MB
[Jittor] Epoch 5, Total Loss: 1.4974, Cls Loss: 1.3696, BBox Loss: 0.0256, mAP: 0.0261, Time: 4.42s, Memory: 365.2MB
[Jittor] Epoch 6, Total Loss: 1.1181, Cls Loss: 0.9929, BBox Loss: 0.0250, mAP: 0.0000, Time: 4.30s, Memory: 365.2MB
[Jittor] Epoch 7, Total Loss: 0.7810, Cls Loss: 0.6567, BBox Loss: 0.0249, mAP: 0.0261, Time: 4.59s, Memory: 365.2MB
[Jittor] Epoch 8, Total Loss: 0.5087, Cls Loss: 0.3847, BBox Loss: 0.0248, mAP: 0.0050, Time: 4.28s, Memory: 365.2MB
[Jittor] Epoch 9, Total Loss: 0.3349, Cls Loss: 0.2144, BBox Loss: 0.0241, mAP: 0.0100, Time: 4.26s, Memory: 365.2MB
[Jittor] Epoch 10, Total Loss: 0.2502, Cls Loss: 0.1338, BBox Loss: 0.0233, mAP: 0.0200, Time: 4.26s, Memory: 365.2MB
```

### PyTorch训练日志
```
[PyTorch] Epoch 1, Total Loss: 2.4310, Cls Loss: 2.2946, BBox Loss: 0.0273, mAP: 0.0000, Time: 3.94s, Memory: 492.6MB
[PyTorch] Epoch 2, Total Loss: 2.2869, Cls Loss: 2.1542, BBox Loss: 0.0265, mAP: 0.0000, Time: 3.46s, Memory: 492.6MB
[PyTorch] Epoch 3, Total Loss: 2.0868, Cls Loss: 1.9563, BBox Loss: 0.0261, mAP: 0.0000, Time: 3.53s, Memory: 492.6MB
[PyTorch] Epoch 4, Total Loss: 1.8548, Cls Loss: 1.7254, BBox Loss: 0.0259, mAP: 0.0000, Time: 3.51s, Memory: 492.6MB
[PyTorch] Epoch 5, Total Loss: 1.5870, Cls Loss: 1.4575, BBox Loss: 0.0259, mAP: 0.0000, Time: 3.47s, Memory: 492.6MB
[PyTorch] Epoch 6, Total Loss: 1.2328, Cls Loss: 1.1049, BBox Loss: 0.0256, mAP: 0.0000, Time: 3.54s, Memory: 492.6MB
[PyTorch] Epoch 7, Total Loss: 0.9362, Cls Loss: 0.8097, BBox Loss: 0.0253, mAP: 0.0100, Time: 3.58s, Memory: 492.6MB
[PyTorch] Epoch 8, Total Loss: 0.6362, Cls Loss: 0.5112, BBox Loss: 0.0250, mAP: 0.0000, Time: 3.79s, Memory: 492.6MB
[PyTorch] Epoch 9, Total Loss: 0.4222, Cls Loss: 0.2974, BBox Loss: 0.0250, mAP: 0.0000, Time: 3.54s, Memory: 487.6MB
[PyTorch] Epoch 10, Total Loss: 0.3105, Cls Loss: 0.1902, BBox Loss: 0.0241, mAP: 0.0050, Time: 3.48s, Memory: 492.6MB
```

## Jittor与PyTorch对齐训练曲线

<img width="562" height="421" alt="image" src="https://github.com/user-attachments/assets/50dac31f-db46-49f6-8527-70ac40b0f4b1" />


## 可变形卷积的可视化展示

<img width="457" height="487" alt="image" src="https://github.com/user-attachments/assets/1948ee7e-d76c-47de-874b-3c17d35948a3" />

