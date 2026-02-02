import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(BaseModule):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        c = self.conv(x)
        c = self.bn(c)
        c = self.act(c)
        return c

class FractionalGaborFilter(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super().__init__()
        self.real_weights = nn.ParameterList()
        # self.imag_weights = nn.ParameterList() # Currently unused in original logic

        for angle in angles:
            for scale in scales:
                real_weight = self.generate_fractional_gabor(
                    in_channels, out_channels, kernel_size, order, angle, scale
                )
                self.real_weights.append(nn.Parameter(real_weight))

    def generate_fractional_gabor(self, in_channels, out_channels, size, order, angle, scale):
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        angle_rad = angle * np.pi / 180.0 
        x_theta = x * np.cos(angle_rad) + y * np.sin(angle_rad)
        y_theta = -x * np.sin(angle_rad) + y * np.cos(angle_rad)

        real_part = np.exp(-((x_theta**2 + (y_theta / scale) ** 2) ** order)) * np.cos(
            2 * np.pi * x_theta / scale
        )
        
        real_weight = torch.tensor(real_part, dtype=torch.float32).view(1, 1, size[0], size[1])
        real_weight = real_weight.repeat(out_channels, 1, 1, 1) # [out, 1, H, W] - Group conv style
        return real_weight

    def forward(self, x):
        real_weights = [weight for weight in self.real_weights]
        # Simply summing weights and multiplying by x? 
        # Note: Original code logic was: sum(weight * x). 
        # This implies weight and x broadcast. Assuming x is a parameter in GaborSingle.
        real_result = sum(weight * x for weight in real_weights)
        return real_result

class GaborSingle(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super().__init__()
        self.gabor = FractionalGaborFilter(
            in_channels, out_channels, kernel_size, order, angles, scales
        )
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Generate dynamic weights
        out_weight = self.gabor(self.t)
        # Standard convolution with generated weights
        out = F.conv2d(x, out_weight, stride=1, padding=(out_weight.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        # Note: Padding + MaxPool with stride 1 effectively keeps resolution same or shifts it.
        # Assuming intention is to keep resolution same for feature extraction within block.
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out

class GaborFPU(BaseModule):
    def __init__(self, in_channels, out_channels, order=0.25, angles=[0, 45, 90, 135], scales=[1, 2, 3, 4]):
        super().__init__()
        self.gabor = GaborSingle(
            in_channels // 4, out_channels // 4, (3, 3), order, angles, scales
        )
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat(
            [self.gabor(x1), self.gabor(x2), self.gabor(x3), self.gabor(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out

class FrFTFilter(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.register_buffer(
            "weight",
            self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order),
        )

    def generate_FrFT_filter(self, in_channels, out_channels, kernel, f, p):
        N = out_channels
        d_x = kernel[0]
        d_y = kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        [X, Y] = np.meshgrid(x, y)

        real_FrFT_filterX = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filterY = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(N):
            real_FrFT_filterX[:, :, i] = np.cos(
                -f * (X) / math.sin(p) + (f * f + X * X) / (2 * math.tan(p))
            )
            real_FrFT_filterY[:, :, i] = np.cos(
                -f * (Y) / math.sin(p) + (f * f + Y * Y) / (2 * math.tan(p))
            )
            real_FrFT_filter[:, :, i] = (
                real_FrFT_filterY[:, :, i] * real_FrFT_filterX[:, :, i]
            )
        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(N):
            g_f[:, :, :, i] = np.repeat(
                real_FrFT_filter[:, :, i : i + 1], in_channels, axis=2
            )
        g_f = np.array(g_f)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))

        return torch.tensor(g_f_real).type(torch.FloatTensor)

    def forward(self, x):
        # Applying weight mask to param x
        x = x * self.weight
        return x

class FrFTSingle(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fft(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out

class FourierFPU(BaseModule):
    def __init__(self, in_channels, out_channels, order=0.25):
        super().__init__()
        # Ensure division stability
        mid_in = in_channels // 4
        mid_out = out_channels // 4
        
        self.fft1 = FrFTSingle(mid_in, mid_out, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(mid_in, mid_out, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(mid_in, mid_out, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(mid_in, mid_out, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat(
            [self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out

class SPU(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = Conv(in_channels // 2, in_channels // 2, 3, g=in_channels // 2)
        self.c2 = Conv(in_channels // 2, in_channels // 2, 5, g=in_channels // 2)
        self.c3 = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2 + x1)
        x_out = torch.cat([x1, x2], dim=1)
        x_out = self.c3(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out

class SFS_Conv(BaseModule):
    """SFS Convolution Module used as a building block."""
    def __init__(self, in_channels, out_channels, order=0.25, filter="FrGT"):
        super().__init__()
        self.PWC0 = Conv(in_channels, in_channels // 2, 1)
        self.PWC1 = Conv(in_channels, in_channels // 2, 1)
        self.SPU = SPU(in_channels // 2, out_channels)

        assert filter in (
            "FrFT",
            "FrGT",
        ), "The filter type must be either Fractional Fourier Transform(FrFT) or Fractional Gabor Transform(FrGT)."
        if filter == "FrFT":
            self.FPU = FourierFPU(in_channels // 2, out_channels, order)
        elif filter == "FrGT":
            self.FPU = GaborFPU(in_channels // 2, out_channels, order)

        self.PWC_o = Conv(out_channels, out_channels, 1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_spa = self.SPU(self.PWC0(x))
        x_fre = self.FPU(self.PWC1(x))
        out = torch.cat([x_spa, x_fre], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return self.PWC_o(out1 + out2)

@MODELS.register_module()
class SFSCNet(BaseModule):
    """
    SFS-CNet Backbone for MMDetection.
    Constructs a pyramid structure using SFS_Conv blocks.
    
    Args:
        base_channels (int): Base channels for the stem.
        arch_settings (list[int]): Number of SFS blocks in each stage. Default: [2, 2, 2, 2]
        filter_type (str): 'FrFT' or 'FrGT'.
        out_indices (Sequence[int]): Output from which stages.
    """
    def __init__(self, 
                 base_channels=32, 
                 arch_settings=[2, 2, 2, 2], 
                 filter_type="FrGT",
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        
        # 1. Stem Layer (Input -> C1)
        self.stem = Conv(3, base_channels, 3, s=1) # Assuming input size doesn't reduce dramatically initially or uses config input size
        
        self.stages = nn.ModuleList()
        in_c = base_channels
        
        # 2. Build Stages
        for i, num_blocks in enumerate(arch_settings):
            stage = nn.ModuleList()
            out_c = base_channels * (2 ** i)
            
            # Downsample layer (except for the first stage if we want to keep high res, 
            # typically Stage 1 is stride 2 or 4 relative to image)
            # Here we assume a standard structure: Downsample -> Process
            if i > 0:
                downsample = Conv(in_c, out_c, 3, s=2) 
            else:
                downsample = Conv(in_c, out_c, 3, s=2) # First stage also downsamples to save memory
            
            stage.append(downsample)
            
            # Blocks
            for _ in range(num_blocks):
                stage.append(SFS_Conv(out_c, out_c, filter=filter_type))
            
            self.stages.append(stage)
            in_c = out_c

    def forward(self, x):
        outs = []
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            for layer in stage:
                x = layer(x)
            
            if i in self.out_indices:
                outs.append(x)
                
        return tuple(outs)

if __name__ == "__main__":
    # Test code
    batch_size = 2
    height = 256
    width = 256
    
    # Test Model registration
    model = SFSCNet(base_channels=32)
    x = torch.randn(batch_size, 3, height, width)
    
    # Forward
    outputs = model(x)
    print("Model created.")
    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")