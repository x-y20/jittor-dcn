import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from jittor import nn
from deform_conv import DeformConv2d

jt.flags.use_cuda = 0

class DeformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.deform_layer = DeformConv2d(1, 8, 3, padding=1)
    def execute(self, x):
        offset = self.deform_layer.offset_conv(x)  # [B, 2N, H, W]
        return offset

def visualize_offset_map(image, offset, title="", step=2, alpha=0.3, scale=0.5):
    """
    image: numpy array, shape [H, W]
    offset: numpy array, shape [H, W, N, 2]
    """
    H, W = image.shape
    N = offset.shape[2]
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    for i in range(0, H, step):
        for j in range(0, W, step):
            for n in range(N):
                dx, dy = offset[i, j, n]
                plt.arrow(j, i, dx * scale, dy * scale,
                          head_width=0.3, head_length=0.3,
                          color='red', alpha=alpha)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    X = np.load("data/sample_dataset/X.npy")[:10]
    imgs = X[:, 0]

    model = DeformNet()
    model.eval()

    with jt.no_grad():
        offsets = model(jt.array(X.astype(np.float32)))
        offsets = offsets.numpy()

    for idx in range(10):
        img = imgs[idx]
        off = offsets[idx]  # [2N, H, W]

        N = off.shape[0] // 2
        off = off.reshape(2, N, off.shape[1], off.shape[2])
        off = off.transpose(2, 3, 1, 0)  # [H, W, N, 2]

        visualize_offset_map(img, off, title=f"Sample {idx+1}", step=2, alpha=0.3)

        input("Press Enter to show next...")

if __name__ == "__main__":
    main()
