import torch
import torch.nn as nn

def calc_iou(anchors, gt_boxes):
    """
    Calculate the Intersection over Union (IoU) between anchor boxes and ground truth boxes.

    Parameters:
    -----------
    - anchors (torch.Tensor): Tensor of shape (N, 4) where N is the number of anchor boxes.
    - gt_boxes (torch.Tensor): Tensor of shape (M, 4) where M is the number of ground truth boxes.

    Returns:
    -----------
    - iou (torch.Tensor): Tensor of shape (N, M) containing the IoU values.
    """

    inter_x0 = torch.max(torch.unsqueeze(anchors[:, 0], dim=1), gt_boxes[:, 0]) # (N, M)
    inter_y0 = torch.max(torch.unsqueeze(anchors[:, 1], dim=1), gt_boxes[:, 1]) # (N, M)
    inter_x1 = torch.min(torch.unsqueeze(anchors[:, 2], dim=1), gt_boxes[:, 2]) # (N, M)
    inter_y1 = torch.min(torch.unsqueeze(anchors[:, 3], dim=1), gt_boxes[:, 3]) # (N, M)

    inter_width = torch.clamp(inter_x1 - inter_x0, min=0) # (N, M)
    inter_height = torch.clamp(inter_y1 - inter_y0, min=0) # (N, M)
    intersection = inter_width * inter_height # (N, M)

    anchor_areas = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]) # (N,)
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) # (M,)
    union = torch.unsqueeze(anchor_areas, dim=1) + gt_areas - intersection # (N, M)
    union = torch.clamp(union, min=1e-8) # Avoid division by zero

    iou = intersection / union # (N, M)
    return iou

class FocalLoss(nn.Module):
    """
    Implements Focal Loss for object detection, from the RetinaNet paper.

    Parameters:
    -----------
    classifications (torch.Tensor): (batch_size, num_anchors, num_classes)
    regressions (torch.Tensor): (batch_size, num_anchors, 4)
    anchors (torch.Tensor): (1, num_anchors, 4)
    annotations (torch.Tensor): (batch_size, num_annotations, 5)

    Returns:
    --------
    Tuple of classification and regression losses: (classification_loss_avg, regression_loss_avg)
    """
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchors = anchors.to(classifications.device)
        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_midpoint_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_midpoint_y = anchor[:, 1] + 0.5 * anchor_heights

        for i in range(batch_size):
            classification = classifications[i, :, :] # (num_anchors, num_classes)
            regression = regressions[i, :, :] # (num_anchors, 4)
            annotation = annotations[i, :, :] 
            annotation = annotation[annotation[:, 4] != -1] # (num_annotations, 5)

            # no objects present
            if annotation.shape[0] == 0: 
                regression_losses.append(torch.tensor(0).float())
                classification_losses.append(torch.tensor(0).float())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            ious = calc_iou(anchors[0, :, :], annotation[:, :4]) # (num_anchors, num_annotations)
            best_ious, best_iou_indices = torch.max(ious, dim=1) # (num_anchors, )

            # prepare classification targets
            targets = -torch.ones_like(classification) # (num_anchors, num_classes)
            targets[torch.lt(best_ious, 0.4), :] = 0 # (num_anchors, num_classes)
            positive_indices = torch.ge(best_ious, 0.5) # (num_anchors, )
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = annotation[best_iou_indices, :] # (num_anchors, 5)
            targets[positive_indices, :] = 0 # (num_anchors, num_classes)
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1 # (num_anchors, num_classes)

            # compute classification loss
            alpha_factor = torch.ones_like(targets) * alpha # (num_anchors, num_classes)
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor) # (num_anchors, num_classes)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification) # (num_anchors, num_classes)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma) # (num_anchors, num_classes)
            bce = -(targets * torch.log(classification) + (1. - targets) * torch.log(1. - classification)) # (num_anchors, num_classes)
            classification_loss = focal_weight * bce # (num_anchors, num_classes)
            zeros = torch.zeros_like(classification_loss) # (num_anchors, num_classes)
            classification_loss = torch.where(torch.ne(targets, -1.), classification_loss, zeros) # (num_anchors, num_classes)
            classification_losses.append(classification_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :] # (num_positive_anchors, 5)

                # prepare regression targets
                anchor_widths_pi = anchor_widths[positive_indices] # (num_positive_anchors, )
                anchor_heights_pi = anchor_heights[positive_indices] # (num_positive_anchors, )
                anchor_midpoint_x_pi = anchor_midpoint_x[positive_indices] # (num_positive_anchors, )
                anchor_midpoint_y_pi = anchor_midpoint_y[positive_indices] # (num_positive_anchors, )

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0] # (num_positive_anchors, )
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1] # (num_positive_anchors, )
                gt_midpoint_x = assigned_annotations[:, 0] + 0.5 * gt_widths # (num_positive_anchors, )
                gt_midpoint_y = assigned_annotations[:, 1] + 0.5 * gt_heights # (num_positive_anchors, )

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_midpoint_x - anchor_midpoint_x_pi) / anchor_widths_pi # (num_positive_anchors, )
                targets_dy = (gt_midpoint_y - anchor_midpoint_y_pi) / anchor_heights_pi # (num_positive_anchors, )
                targets_dw = torch.log(gt_widths / anchor_widths_pi) # (num_positive_anchors, )
                targets_dh = torch.log(gt_heights / anchor_heights_pi) # (num_positive_anchors, )
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh)).t() # (num_positive_anchors, 4)
                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(targets.device)
                targets = targets / norm # (num_positive_anchors, 4)

                # compute regression loss
                diff = torch.abs(targets - regression[positive_indices, :]) # (num_positive_anchors, 4)
                smooth_l1_loss = torch.where(
                    diff <= 1.0 / 9.0,
                    0.5 * 9.0 * diff**2,
                    diff - 0.5 / 9.0
                )
                regression_losses.append(smooth_l1_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float())

        classification_losses = [loss.to(classifications.device) for loss in classification_losses]
        regression_losses = [loss.to(classifications.device) for loss in regression_losses]
        classification_loss_avg = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        regression_loss_avg = torch.stack(regression_losses).mean(dim=0, keepdim=True)
        return classification_loss_avg, regression_loss_avg