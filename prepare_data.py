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
