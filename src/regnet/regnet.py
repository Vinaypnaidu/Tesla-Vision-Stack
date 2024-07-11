import torch
import torch.nn as nn
from .utils import quantize_widths, adjust_widths_groups
from .modules import Stage, Stem, XBlock

class RegNet(nn.Module):
    """
    A RegNet model (Radosavovic et al. 2020) consisting of several stages.
    Note: This is a slightly modified implementation that returns 5 feature volumes.

    Parameters:
    -----------    
    config: Configuration dict. (check config.yaml)
    x (torch.Tensor): Input images.

    Returns:
    --------
    stem_outputs (list of torch.Tensor): Regnet features from the stem (len 2).
    body_outputs (list of torch.Tensor): Regnet features from the body (len 5).
    """
    def __init__(self, config) -> None:
        super(RegNet, self).__init__()
        w_0 = config["W0"]
        w_a = config["WA"]
        w_m = config["WM"]
        d = config["DEPTH"]
        group_w = config["GROUP_W"]
        stride = config["STRIDE"]
        bot_mul = config["BOT_MUL"]
        q = config["Q"] 
        se_ratio = config["SE_RATIO"]
        stem_in = config["STEM_IN"]
        stem_out = config["STEM_OUT"]
        stem_stride = config["STEM_STRIDE"]

        ws, ds = quantize_widths(w_0, w_a, w_m, d, q)
        ws, gs = adjust_widths_groups(ws, group_w, bot_mul)

        self.stem = Stem(w_in=stem_in, w_out=stem_out, stride=stem_stride)
        # downsample blocks not part of the RegNet paper
        self.downsample1 = nn.Sequential(
            XBlock(stem_out, stem_out, 1, bot_mul, group_w, se_ratio),
            XBlock(stem_out, stem_out, 1, bot_mul, group_w, se_ratio),
            XBlock(stem_out, stem_out, stride, bot_mul, group_w, se_ratio)
        )
        final_w = ws[-1]
        self.downsample2 = nn.Sequential(
            XBlock(final_w, final_w, stride, bot_mul, group_w, se_ratio),
            XBlock(final_w, final_w, 1, bot_mul, group_w, se_ratio),
            XBlock(final_w, final_w, 1, bot_mul, group_w, se_ratio)
        )

        w_in = stem_out
        self.body = nn.ModuleList()
        for (w_out, num_blocks, group_w) in zip(ws, ds, gs):
            self.body.append(Stage(num_blocks, w_in, w_out, stride, bot_mul, group_w, se_ratio))
            w_in = w_out

    def forward(self, x):
        stem_outputs = []
        out = self.stem(x)
        stem_outputs.append(out)
        out = self.downsample1(out)
        stem_outputs.append(out)
        body_outputs = []
        for stage in self.body:
            out = stage(out)
            body_outputs.append(out)
        final_stage_out = self.downsample2(body_outputs[-1])
        body_outputs.append(final_stage_out)
        return stem_outputs, body_outputs