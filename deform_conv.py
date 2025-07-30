import jittor as jt
from jittor import nn
import math


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        N = self.kernel_size[0] * self.kernel_size[1]

        self.offset_conv = nn.Conv(
            in_channels, 2 * N,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding
        )

        std = math.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = jt.init.gauss([
            out_channels, in_channels, *self.kernel_size
        ], mean=0.0, std=std)

        self.bias = jt.init.constant(shape=[out_channels], value=0.0) if bias else None

        self.offset_conv.weight = jt.zeros_like(self.offset_conv.weight)
        self.offset_conv.bias = jt.zeros_like(self.offset_conv.bias)


    def grid_sample_wrapper(self, x, coords):
        B, C, H, W = x.shape
        N = coords.shape[3]

        norm_x = coords[..., 0] / (W - 1) * 2 - 1
        norm_y = coords[..., 1] / (H - 1) * 2 - 1
        grid = jt.stack([norm_x, norm_y], dim=-1)  # [B, H, W, N, 2]

        grid = grid.permute(0, 3, 1, 2, 4).reshape(B * N, H, W, 2)  # [B*N, H, W, 2]
        x_repeat = x.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, C, H, W)

        sampled = nn.grid_sample(x_repeat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.reshape(B, N, C, H, W).permute(0, 2, 3, 4, 1)  # [B, C, H, W, N]
        return sampled

    def execute(self, x):
        offset = self.offset_conv(x)
        B, _, H, W = offset.shape
        N = self.kernel_size[0] * self.kernel_size[1]

        offset = offset.view(B, 2, N, H, W).permute(0, 3, 4, 2, 1)

        yv = jt.arange(H).view(1, H, 1, 1).repeat(B, 1, W, N)
        xv = jt.arange(W).view(1, 1, W, 1).repeat(B, H, 1, N)
        grid = jt.stack([xv, yv], dim=-1)

        sampling_locs = grid + offset 
        sampled = self.grid_sample_wrapper(x, sampling_locs)
        sampled = sampled.permute(0, 2, 3, 4, 1).reshape(B, H, W, -1)

        weight_mat = self.weight.reshape(self.out_channels, -1)
        sampled_flat = sampled.reshape(B * H * W, -1)
        wm = weight_mat.transpose(1, 0)

        out_flat = jt.matmul(sampled_flat, wm)
        out = out_flat.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
