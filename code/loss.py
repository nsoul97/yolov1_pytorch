import copy

import torch as th
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: th.Tensor) -> th.Tensor:
    """
    Calculate the bounding box corners' coordinates from the YOLO-formatted bounding box coordinates.

    :param bboxes_coords: A Tensor that contains the (x_center, y_center, width, height) center-formatted bounding
    boxes coordinates.
    :return: A Tensor with the same shape as the bboxes_coords Tensor that contains the (xmin, ymin, xmax, ymax)
             corners-formatted bounding box corners' coordinates.
    """

    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2

    bb_corners = th.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: th.Tensor, bboxes2_coords: th.Tensor) -> th.Tensor:
    """
    Calculate the intersection over union for the given sets of bounding boxes. The coordinates of each boudning box
    are represented in the corners-format (xmin, ymin, xmax, ymax).

    :param bboxes1_coords: The first set of bounding boxes.
    :param bboxes2_coords: The second set of bounding boxes.
    :return: A Tensor that contains IOU scores for the corresponding bounding boxes of the first and of the second set.
    """
    xmin = th.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = th.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax = th.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = th.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # clamp(min=0) for the special case: intersection=0
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    union = area_bb1 + area_bb2 - intersection

    # add 1e-6 to avoid division by 0
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):
    """
    An implementation of the YOLO Loss function. YOLO loss consists of 3 components:
    - the localization loss
    - the objectness loss
    - the classification loss
    """

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        """
        Initialize the YOLO Loss module.

        The L_coord hyperparameter is used, as otherwise the localization and the classification errors would be equally
        weighted, which is not ideal for maximizing the average precision. In this way, the loss from the bounding box
        predictions is increased.

        The majority of the grid cells in every image do not contain any object.  This pushes the “confidence” scores of
        those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to
        model instability, causing training to diverge early on. To this end, the  L_noobj hyperparameter is used to
        decrease the objectness loss when an object does not exist in the grid cell.

        :param S: The number of grid cells per row/column. Each image is divided into an (S x S) grid
        :param C: The number of classes
        :param B: The number of the bounding boxes that YOLO predicts per grid cell
        :param D: The input dimension of the (D x D) RGB images
        :param L_coord: The L_coord hyperparameter
        :param L_noobj: The L_noobj hyperparameter
        """
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.D = D
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        # The indices for each of the B bounding boxes of the algorithm
        self.register_buffer('pred_bb_ind', th.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        """
        The YOLO loss is the sum of the localization error, the objectness error and classification error. These errors
        are sum-squared because in this way they are easier to optimize.

        In order to identify which of the B output bounding boxes is responsible for predicting the object, we have to
        calculate the IOUs of the B boxes with the ground truth box. Since the center coordinates and the dimensions of
        the bounding box are differently normalized, calculating the corner coordinates without reversing the
        normalization of the values would result to incorrect IOU scores. Therefore, we first calculate the corner
        coordinates of the bounding boxes in the original (D x D) image.

        The box with the highest IOU is selected as responsible for the prediction. If all the B predictions do not
        intersect with the ground truth box, the box with the lowest root-mean-square error is selected, as long as the
        error is less than 20. Otherwise, one of the B bounding boxes is selected randomly.

        Instead of the width and the height of the bounding boxes, the model outputs their corresponding square roots.
        In this way, in the localization error, we do not apply a square root operation, whose gradient at 0 is +inf
        and could be responsible for the divergence of our model.

        For the objectness loss, if an object exists in a grid cell and the bounding box is responsible for the
        prediction, the confidence should be equal to the IOU of the prediction and the ground truth box. Otherwise, the
        confidence should be 0.

        :param y_pred: The (N, S, S, C+B*5) predictions of the YOLO model for a mini-batch of N images
        :param y_gt: The (N, S, S, C+5) ground truth labels of the corresponding images
        :return: The YOLO loss for the given predictions and ground truth labels.
        """
        n = y_pred.shape[0]
        exists_obj_i = y_gt[..., 0:1]
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        gt_bboxes_scaled_coords[..., :2] /= self.S
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        pred_bboxes_scaled_coords[..., :2] /= self.S
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:]
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)

        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)
        max_iou_score, max_iou_index = th.max(iou_scores, dim=-1)

        rmse_scores = th.sqrt(th.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = th.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0

        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]
        is_best_box = one_hot(best_index, self.B)

        exists_obj_ij = exists_obj_i * is_best_box
        exists_noobj_ij = 1 - exists_obj_ij

        # Localization Loss
        localization_center_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2))

        localization_dims_loss = self.L_coord * th.sum(exists_obj_ij[..., None] * (
                (th.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2))

        localization_loss = localization_center_loss + localization_dims_loss

        # Objectness Loss
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]

        objectness_obj_loss = th.sum(exists_obj_ij * (iou_scores - pred_bbox_cscores) ** 2)
        objectness_noobj_loss = self.L_noobj * th.sum(exists_noobj_ij * pred_bbox_cscores ** 2)

        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # Classification Loss
        pred_bboxes_class = y_pred[..., :self.C]
        gt_bboxes_class = y_gt[..., 1:self.C + 1]

        classification_loss = th.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # Average YOLO Loss per instance
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss
