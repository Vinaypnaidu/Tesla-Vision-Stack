import numpy as np

def quantize_widths(w_0, w_a, w_m, d, q=8):
    """
    Implements the quantization technique explained in section 3.3.
    """
    depths = np.arange(d)
    u = w_0 + w_a * depths
    s = np.round(np.log(u / w_0) / np.log(w_m))
    w = w_0 * np.power(w_m, s)
    w = np.round(w / q).astype(int) * q # the paper samples widths divisible by 8
    ws, ds = np.unique(w, return_counts=True)
    return ws, ds

def adjust_widths_groups(ws, group_w, bot_mul):
    """
    Adjusts the compatibility of widths, bottlenecks, and groups.
    """
    ws_bot = [int(max(1, w * bot_mul)) for w in ws]
    gs = [min(w_bot, group_w) for w_bot in ws_bot]
    ws_bot = [int(round(w / g)) * g for (w, g) in zip(ws_bot, gs)]
    ws = [int(w / bot_mul) for w in ws_bot]
    return ws, gs