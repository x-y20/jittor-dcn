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
