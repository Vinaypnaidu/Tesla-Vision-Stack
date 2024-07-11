import torch
import torch.nn as nn
from .utils import BBoxTransform, ClipBoxes, Anchors
from .loss import FocalLoss
from torchvision.ops.boxes import nms 

class ObjectDetectionHead(nn.Module):
    """
    Object detection head of the HydraNet. 

    Parameters:
    -----------
    in_channels (int): Number of output channels in the fused feaures from the BiFPN.
    num_anchors (int): Number of anchor boxes per cell in the feature maps.
    num_layers (int): Number of convolutional layers in classifier and regressor.
    num_classes (int): Number of classes to be detected.

    features (list of torch.Tensor): Fused features from the BiFPN.
    images (torch.Tensor): A batch of images.
    annotations (torch.Tensor): Ground truth boxes.

    Returns:
    --------
    list of torch.Tensor
        If annotations are provided, returns the classification and regression losses.
        If annotations are not provided, returns the NMS scores, NMS classes, and transformed bounding boxes.
    """
    def __init__(self, config):
        super(ObjectDetectionHead, self).__init__()
        in_channels = config["NUM_CHANNELS_BIFPN"]
        num_anchors = config["NUM_ANCHORS"] 
        num_layers = config["NUM_LAYERS_DETECTION"]
        num_classes = config["NUM_CLASSES"]
        self.classificationHead = Classifier(in_channels, num_anchors, num_layers, num_classes)
        self.regressionHead = Regressor(in_channels, num_anchors, num_layers)
        self.bboxTransform = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()
        self.anchors = Anchors()

    def forward(self, features, images):
        regression = torch.cat([self.regressionHead(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationHead(feature) for feature in features], dim=1)
        anchorboxes = self.anchors(images)
        return [classification, regression, anchorboxes]
    
    def process_outputs(self, classification, regression, anchorboxes, images):
        bboxes = self.bboxTransform(anchorboxes, regression)
        bboxes = self.clipBoxes(bboxes, images)

        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > 0.05)[0, :, 0]
        if scores_over_thresh.sum() == 0:
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        # perform inference for the first image in the batch
        classification = classification[0, scores_over_thresh, :]
        bboxes = bboxes[0, scores_over_thresh, :]
        scores = scores[0, scores_over_thresh, :]
        nms_idx = nms(bboxes, scores.squeeze(), 0.5)
        nms_scores, nms_classes = classification[nms_idx, :].max(dim=1)
        return [nms_scores, nms_classes, bboxes[nms_idx, :]]

class Regressor(nn.Module):
    """
    Regression head for object detection.

    Parameters:
    -----------
    in_channels (int): Number of output channels in the fused feaures from the BiFPN.
    num_anchors (int): Number of anchor boxes per cell in the feature maps.
    num_layers (int): Number of convolutional layers.
    inputs (torch.Tensor): BiFPN feature map.

    Returns:
    --------
    output (torch.Tensor): Tensor of shape (N, H * W * num_anchors, 4) containing deltas.
    """
    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs) # (N, C, H, W)
        inputs = self.header(inputs) # (N, num_anchors * 4, H, W)
        output = inputs.permute(0, 2, 3, 1) # (N, H, W, num_anchors * 4)
        return output.contiguous().view(output.shape[0], -1, 4) # (N, H * W * num_anchors, 4)

class Classifier(nn.Module):
    """
    Classification head for object detection.

    Parameters:
    -----------
    in_channels (int): Number of output channels in the fused feaures from the BiFPN.
    num_anchors (int): Number of anchor boxes per cell in the feature maps.
    num_layers (int): Number of convolutional layers.
    num_classes (int): Number of classes in the dataset.
    inputs (torch.Tensor): BiFPN feature map.

    Returns:
    --------
    output (torch.Tensor): Tensor of shape (N, H * W * num_anchors, num_classes) containing classification results.
    """
    def __init__(self, in_channels, num_anchors, num_layers, num_classes):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs) # (N, C, H, W)
        inputs = self.header(inputs) # (N, num_anchors * num_classes, H, W)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1) # (N, H, W, num_anchors * num_classes)
        output = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], inputs.shape[2], self.num_anchors,
                                          self.num_classes) # (N, H, W, num_anchors, num_classes)
        return output.contiguous().view(output.shape[0], -1, self.num_classes) # (N, H * W * num_anchors, num_classes)