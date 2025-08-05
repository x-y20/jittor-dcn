import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import torch
import torch.nn as tnn

import jittor as jt
from jittor import nn
from deform_conv import DeformConv2d
from train import TorchDeformConv2d

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

        self.conv2 = TorchDeformConv2d(16, 32, 3, 2, 1)
        self.bn2 = tnn.BatchNorm2d(32)

        self.conv3 = TorchDeformConv2d(32, 64, 3, 2, 1)
        self.bn3 = tnn.BatchNorm2d(64)

        self.conv4 = TorchDeformConv2d(64, 128, 3, 2, 1)
        self.bn4 = tnn.BatchNorm2d(128)

        self.conv5 = TorchDeformConv2d(128, 256, 3, 2, 1)
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

        self.conv2 = DeformConv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = DeformConv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = DeformConv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = DeformConv2d(128, 256, 3, 2, 1)
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
