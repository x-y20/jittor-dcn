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
        grid = jt.stack([norm_x, norm_y], dim=-1)  

        grid = grid.permute(0, 3, 1, 2, 4).reshape(B * N, H, W, 2)
        x_repeat = x.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, C, H, W)

        sampled = nn.grid_sample(x_repeat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.reshape(B, N, C, H, W).permute(0, 2, 3, 4, 1)
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

class DeformRoIPool(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=1):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def execute(self, features, rois, offsets):
        B, C, H, W = features.shape
        num_rois = rois.shape[0]
        pooled_h, pooled_w = self.output_size

        batch_indices = rois[:, 0].long()
        roi_coords = rois[:, 1:5] * self.spatial_scale
        x1, y1, x2, y2 = roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2], roi_coords[:, 3]
        roi_w = jt.maximum(x2 - x1, 1e-6)
        roi_h = jt.maximum(y2 - y1, 1e-6)

        ph = jt.arange(pooled_h, dtype=jt.float32)
        pw = jt.arange(pooled_w, dtype=jt.float32)
        ph_grid, pw_grid = jt.meshgrid(ph, pw)
        ph_grid = ph_grid.reshape(-1)
        pw_grid = pw_grid.reshape(-1)

        bin_w = roi_w[:, None] / pooled_w
        bin_h = roi_h[:, None] / pooled_h

        bin_cx = x1[:, None] + (pw_grid + 0.5) * bin_w
        bin_cy = y1[:, None] + (ph_grid + 0.5) * bin_h

        offset_x = offsets[:, ph_grid * pooled_w + pw_grid, 0] * roi_w[:, None]
        offset_y = offsets[:, ph_grid * pooled_w + pw_grid, 1] * roi_h[:, None]

        cx = bin_cx + offset_x
        cy = bin_cy + offset_y

        x0 = jt.floor(cx).long()
        x1 = x0 + 1
        y0 = jt.floor(cy).long()
        y1 = y0 + 1

        x0 = jt.clamp(x0, 0, W - 1)
        x1 = jt.clamp(x1, 0, W - 1)
        y0 = jt.clamp(y0, 0, H - 1)
        y1 = jt.clamp(y1, 0, H - 1)

        dx = cx - x0.float()
        dy = cy - y0.float()

        w00 = (1 - dx) * (1 - dy)
        w01 = (1 - dx) * dy
        w10 = dx * (1 - dy)
        w11 = dx * dy

        batch_idx = batch_indices[:, None].repeat(1, pooled_h * pooled_w)
        feats = features[batch_idx.reshape(-1), :, y0.reshape(-1), x0.reshape(-1)]
        feats = feats.reshape(num_rois, pooled_h * pooled_w, C).permute(0, 2, 1)
        val00 = (feats * w00[:, None, :]).sum(dim=2)

        feats = features[batch_idx.reshape(-1), :, y1.reshape(-1), x0.reshape(-1)]
        feats = feats.reshape(num_rois, pooled_h * pooled_w, C).permute(0, 2, 1)
        val01 = (feats * w01[:, None, :]).sum(dim=2)

        feats = features[batch_idx.reshape(-1), :, y0.reshape(-1), x1.reshape(-1)]
        feats = feats.reshape(num_rois, pooled_h * pooled_w, C).permute(0, 2, 1)
        val10 = (feats * w10[:, None, :]).sum(dim=2)

        feats = features[batch_idx.reshape(-1), :, y1.reshape(-1), x1.reshape(-1)]
        feats = feats.reshape(num_rois, pooled_h * pooled_w, C).permute(0, 2, 1)
        val11 = (feats * w11[:, None, :]).sum(dim=2)

        output = val00 + val01 + val10 + val11
        return output.reshape(num_rois, C, pooled_h, pooled_w)


class DeformPSRoIPool(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=1, no_trans=False, group_size=1, part_size=None,
                 trans_std=0.1):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = part_size if part_size else self.output_size
        self.trans_std = trans_std

    def execute(self, features, rois, offsets):
        B, C, H, W = features.shape
        num_rois = rois.shape[0]
        pooled_h, pooled_w = self.output_size
        part_h, part_w = self.part_size
        C_out = C // (pooled_h * pooled_w)

        batch_indices = rois[:, 0].long()
        roi_coords = rois[:, 1:5] * self.spatial_scale
        x1, y1, x2, y2 = roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2], roi_coords[:, 3]
        roi_w = jt.maximum(x2 - x1, 1e-6)
        roi_h = jt.maximum(y2 - y1, 1e-6)

        ph = jt.arange(pooled_h, dtype=jt.float32)
        pw = jt.arange(pooled_w, dtype=jt.float32)
        ph_grid, pw_grid = jt.meshgrid(ph, pw)
        ph_flat = ph_grid.reshape(-1)
        pw_flat = pw_grid.reshape(-1)
        part_idx = ph_flat * pooled_w + pw_flat

        bin_w = roi_w[:, None] / part_w
        bin_h = roi_h[:, None] / part_h

        bin_cx = x1[:, None] + (pw_flat + 0.5) * bin_w
        bin_cy = y1[:, None] + (ph_flat + 0.5) * bin_h

        if not self.no_trans:
            trans_x = offsets[:, part_idx * 2] * roi_w[:, None] * self.trans_std
            trans_y = offsets[:, part_idx * 2 + 1] * roi_h[:, None] * self.trans_std
            cx = bin_cx + trans_x
            cy = bin_cy + trans_y
        else:
            cx = bin_cx
            cy = bin_cy

        x0 = jt.floor(cx).long()
        x1 = x0 + 1
        y0 = jt.floor(cy).long()
        y1 = y0 + 1

        x0 = jt.clamp(x0, 0, W - 1)
        x1 = jt.clamp(x1, 0, W - 1)
        y0 = jt.clamp(y0, 0, H - 1)
        y1 = jt.clamp(y1, 0, H - 1)

        dx = cx - x0.float()
        dy = cy - y0.float()

        w00 = (1 - dx) * (1 - dy)
        w01 = (1 - dx) * dy
        w10 = dx * (1 - dy)
        w11 = dx * dy

        c_out = jt.arange(C_out)[:, None]
        channel_idx = c_out * pooled_h * pooled_w + part_idx
        channel_idx = channel_idx.repeat(num_rois, 1, 1)

        batch_idx = batch_indices[:, None, None].repeat(1, C_out, pooled_h * pooled_w)
        y0_ = y0[:, None, :].repeat(1, C_out, 1)
        x0_ = x0[:, None, :].repeat(1, C_out, 1)
        y1_ = y1[:, None, :].repeat(1, C_out, 1)
        x1_ = x1[:, None, :].repeat(1, C_out, 1)

        val00 = features[batch_idx, channel_idx, y0_, x0_] * w00[:, None, :]
        val01 = features[batch_idx, channel_idx, y1_, x0_] * w01[:, None, :]
        val10 = features[batch_idx, channel_idx, y0_, x1_] * w10[:, None, :]
        val11 = features[batch_idx, channel_idx, y1_, x1_] * w11[:, None, :]

        output = (val00 + val01 + val10 + val11).sum(dim=2)
        return output.reshape(num_rois, C_out, pooled_h, pooled_w)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
