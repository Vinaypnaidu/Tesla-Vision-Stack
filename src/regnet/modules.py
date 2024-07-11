import torch.nn as nn
import torch.nn.functional as F

class XBlock(nn.Module):
    """
    XBlock from the Regnet paper. Refer figure 4.
    If se_ratio is provided, it uses the Squeeze-and-Excitation technique 
    and becomes a YBlock (RegnetY design space) which performs slightly 
    better.

    Parameters:
    -----------    
    - w_in (int): Input block width.
    - w_out (int): Output block width.
    - stride (int): Stride for 3x3 conv and skip connection.
    - bot_mul (float): 1 / bottleneck ratio (0, 1].
    - group_w (int): Group width.
    - se_ratio (float): Squeeze-and-Excitation ratio (0, 1], None for XBlock.
    - x (torch.Tensor): Input volume.

    Returns:
    -----------    
    - torch.Tensor: Output features.
    """
    def __init__(self, w_in, w_out, stride=1, bot_mul=1, group_w=1, se_ratio=None) -> None:
        super(XBlock, self).__init__()

        w_b = max(1, int(round(w_out * bot_mul))) 
        groups = w_b // group_w

        self.block1 = nn.Sequential(
            nn.Conv2d(w_in, w_b, kernel_size=1, bias=False),
            nn.BatchNorm2d(w_b), 
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(w_b),
            nn.ReLU()
        )

        if se_ratio is not None:
            w_se = max(1, int(round(w_in * se_ratio))) 
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(w_b, w_se, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(w_se, w_b, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.se_block = None

        self.block3 = nn.Sequential(
            nn.Conv2d(w_b, w_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(w_out)
        )

        if stride != 1 or w_in != w_out:
            self.skip = nn.Sequential(
                nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(w_out)
            )
        else:
            self.skip = None

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        if self.se_block:
            out = out * self.se_block(out)
        out = self.block3(out)
        if self.skip:
            skip_out = self.skip(x)
        else:
            skip_out = x
        out = F.relu(out + skip_out)
        return out

class Stage(nn.Module):
    """
    A Stage consists of several XBlocks. 

    Parameters:
    -----------    
    - num_blocks (int): Number of XBlocks to be added.
    - w_in (int): Input block width.
    - w_out (int): Output block width.
    - stride (int): Stride for 3x3 conv and skip connection.
    - bot_mul (float): 1 / bottleneck ratio (0, 1].
    - group_w (int): Group width.
    - se_ratio (float): Squeeze-and-Excitation ratio (0, 1], None for XBlock.

    Returns:
    -----------    
    - torch.Tensor: Output features.
    """
    def __init__(self, num_blocks, w_in, w_out, stride, bot_mul, group_w, se_ratio=None) -> None:
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", XBlock(w_in, w_out, stride, bot_mul, group_w, se_ratio))
        for i in range(1, num_blocks):
            self.blocks.add_module(f"block_{i}", XBlock(w_out, w_out, 1, bot_mul, group_w, se_ratio))

    def forward(self, x):
        out = self.blocks(x)
        return out
    
class Stem(nn.Module):
    """
    Stem from figure 3 in the RegNet paper: 3x3, BN, ReLU. 
    """
    def __init__(self, w_in=3, w_out=32, kernel_size=3, stride=2, padding=1) -> None:
        super(Stem, self).__init__()
        
        self.stem_block = nn.Sequential(
            nn.Conv2d(w_in, w_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(w_out),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.stem_block(x)
        return out