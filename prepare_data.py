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
