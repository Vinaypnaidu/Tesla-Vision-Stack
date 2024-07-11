import torch
import torch.nn as nn
import numpy as np

class BBoxTransform(nn.Module):
    """
    Transforms bounding box deltas to bounding box coordinates.

    Parameters:
    -----------    
    - boxes (torch.Tensor): Tensor of shape (batch_size, num_anchors, 4) containing the anchor boxes.
    - deltas (torch.Tensor): Tensor of shape (batch_size, num_anchors, 4) containing the predicted deltas.

    Returns:
    -----------    
    - pred_boxes (torch.Tensor): Tensor of shape (batch_size, num_anchors, 4) containing the predicted bounding boxes.
    """
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        self.mean = torch.tensor([0, 0, 0, 0], dtype=torch.float32) if mean is None else mean
        self.std = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32) if std is None else std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def forward(self, anchor_boxes, deltas):
        anchor_boxes = anchor_boxes.to(self.device)
        deltas = deltas.to(self.device)
        anchor_widths = anchor_boxes[:, :, 2] - anchor_boxes[:, :, 0]
        anchor_heights = anchor_boxes[:, :, 3] - anchor_boxes[:, :, 1]
        anchor_ctr_x = anchor_boxes[:, :, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor_boxes[:, :, 1] + 0.5 * anchor_heights

        # denormalize deltas
        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        # reverse of the training process, check `FocalLoss`
        pred_ctr_x = anchor_ctr_x + dx * anchor_widths
        pred_ctr_y = anchor_ctr_y + dy * anchor_heights
        pred_w = torch.exp(dw) * anchor_widths
        pred_h = torch.exp(dh) * anchor_heights

        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=2)
        return pred_boxes
    
class ClipBoxes(nn.Module):
    """
    Clips bounding boxes to lie within the dimensions of the image.

    Parameters:
    -----------    
    - boxes (torch.Tensor): Tensor of shape (batch_size, num_anchors, 4) containing the predicted bounding boxes.
    - img (torch.Tensor): Tensor of shape (batch_size, 3, height, width) containing the input images.

    Returns:
    -----------    
    - clipped_boxes (torch.Tensor): Tensor of shape (batch_size, num_anchors, 4) containing the clipped bounding boxes.
    """
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        N, C, H, W = img.shape
        clipped_boxes = torch.zeros_like(boxes)
        clipped_boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        clipped_boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        clipped_boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=W)
        clipped_boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=H)
        return clipped_boxes
    
class Anchors(nn.Module):
    """
    Generates anchorboxes. 

    Parameters:
    -----------    
    - pyramid_levels (list): A list of ints indicating feature levels from the backbone.
    - strides (list): A list of ints containing strides.
    - sizes (list): A list of ints containing base anchorbox sizes.
    - ratios (list): A list of floats containing aspect ratios.
    - scales (list): A list of floats indicating object scales.

    Returns:
    -----------  
    - all_anchors (np.array): A numpy array of shape (1, num_anchors, 4) containing anchorboxes.
    """
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels if pyramid_levels is not None else [3, 4, 5, 6, 7]
        self.strides = strides if strides is not None else [2 ** x for x in self.pyramid_levels]
        self.sizes = sizes if sizes is not None else [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = ratios if ratios is not None else np.array([0.5, 1, 2])
        self.scales = scales if scales is not None else np.array([2 ** -1, 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shapes = [(np.array(image_shape) + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        
        all_anchors = []
        for idx, p in enumerate(self.pyramid_levels):
            anchors = self.generate_anchors(base_size=self.sizes[idx])
            shifted_anchors = self.shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors.append(shifted_anchors)

        all_anchors = np.concatenate(all_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)

        anchors_tensor = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.cuda.is_available():
            anchors_tensor = anchors_tensor.cuda()
        return anchors_tensor

    def generate_anchors(self, base_size=16):
        ratios = self.ratios
        scales = self.scales

        num_anchors = len(ratios) * len(scales)
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def shift(self, shape, stride, anchors):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_x_flat = shift_x.ravel()
        shift_y_flat = shift_y.ravel()
        shifts = np.vstack((shift_x_flat, shift_y_flat, shift_x_flat, shift_y_flat)).transpose()

        num_anchors = anchors.shape[0]
        num_shifts = shifts.shape[0]
        anchors = anchors.reshape((1, num_anchors, 4))
        shifts = shifts.reshape((1, num_shifts, 4)).transpose((1, 0, 2))
        all_anchors = (anchors + shifts).reshape((num_shifts * num_anchors, 4))
        return all_anchors