import torch
from torch import nn
import torch.nn.functional as F
''' =================================================================================
    Loss functions for training semantic segmentation models
    Including: f1-loss, IoU-loss, dice-loss, focal-loss, tversky-loss and combined loss functions
================================================================================= '''

def f1_loss(y_true, y_pred):
    """
    F1-Loss:
    F1-score = 2*TP / 2*TP+FP+FN
    F1-Loss = 1 - F1-score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} and y_pred has shape {y_pred.shape}")
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    TP = torch.sum(y_true * y_pred)
    FP = torch.sum((1 - y_true) * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    epsilon = 1e-6 # smoothing term, to prevent the denominator of the F1-score from being zero
    F1 = (TP + epsilon) / (TP + (FP + FN) / 2. + epsilon)
    return 1 - F1 # return F1-Loss directly

def iou_loss(y_true, y_pred):
    """
    IoU Loss: |A∩B| / |A∪B| (|A∪B| = |A|+|B|-|A∩B|)
    IoU = Intersection / Union
    Loss_IoU = 1 - IoU
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} and y_pred has shape {y_pred.shape}")
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    epsilon = 1e-6 # smoothing term, to prevent the denominator of the F1-score from being zero
    IoU = (intersection + epsilon) / (union + epsilon)
    return 1 - IoU  # return IoU-Loss directly

def dice_loss(y_true, y_pred):
    """
    Dice Loss: 2|A∩B| / |A|+|B|
    Dice = (2 * Intersection) / (Union + Intersection)
    Loss_Dice = 1 - Dice
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} and y_pred has shape {y_pred.shape}")
    y_true = y_true.reshape(-1).float()
    y_pred = y_pred.reshape(-1).float()

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    epsilon = 1e-6 # smoothing term, to prevent the denominator of the F1-score from being zero
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice  # Dice Loss

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss:
    Loss_Focal = -alpha * (1 - pt)^gamma * log(pt)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} and y_pred has shape {y_pred.shape}")
    y_true = y_true.reshape(-1).float()
    y_pred = y_pred.reshape(-1).float()

    epsilon = 1e-6 # smoothing term, to prevent the denominator of the F1-score from being zero
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)  # limit predicted value to (0,1)

    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # calculate pt
    focal = -alpha * (1 - pt) ** gamma * torch.log(pt)
    return focal.mean()  # Focal Loss

def tversky_loss(y_true, y_pred, alpha=0.2, beta=0.6):
    """
    Tversky Loss:
    Tversky = Intersection / (Intersection + alpha * FP + beta * FN)
    Loss_Tversky = 1 - Tversky
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} and y_pred has shape {y_pred.shape}")
    y_true = y_true.reshape(-1).float()
    y_pred = y_pred.reshape(-1).float()

    TP = torch.sum(y_true * y_pred)  # True Positives
    FP = torch.sum((1 - y_true) * y_pred)  # False Positives
    FN = torch.sum(y_true * (1 - y_pred))  # False Negatives

    epsilon = 1e-6 # smoothing term, to prevent the denominator of the F1-score from being zero
    tversky = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
    return 1 - tversky  # Tversky Loss

def cross_entropy_loss(y_true, y_pred, weight=None):
    """
    Cross Entropy Loss:
    For semantic segmentation, y_pred should be the logits with shape [N, C, ...],
    and y_true should be the ground truth labels with shape [N, ...] (of type long)
    with class indices in the range [0, C-1].
    when use cross entropy loss, we need to remove the last sigmoid-layer in deeplabv3plus
    """
    # y_pred should have one more dimension than y_true
    if y_pred.dim() != y_true.dim() + 1:
        raise ValueError(
            f"Shape mismatch: y_pred should have one more dimension than y_true, "
            f"but got y_pred.dim()={y_pred.dim()} and y_true.dim()={y_true.dim()}"
        )
    # Ensure y_true is of type long (required by F.cross_entropy)
    return F.cross_entropy(y_pred, y_true.long(), weight=weight)


def LossFunction(y_true, y_pred):
    device = y_pred.device
    y_true = y_true.to(device)  # ensure y_true and y_pred are in the same device


    loss_token = 10
    if loss_token == 1: # F1-Loss
        total_loss = f1_loss(y_true, y_pred)
    if loss_token == 2: # IoU-Loss
        total_loss = iou_loss(y_true, y_pred)
    if loss_token == 3: # Dice-Loss
        total_loss = dice_loss(y_true, y_pred)
    if loss_token == 4: # tversky, improved Dice-Loss
        total_loss = focal_loss(y_true, y_pred)
    if loss_token == 5: # tversky, improved Dice-Loss
        total_loss = tversky_loss(y_true, y_pred)
    if loss_token == 6: # tversky, improved Dice-Loss
        total_loss = cross_entropy_loss(y_true, y_pred)

    if loss_token == 10:  # Dice+Focal fusion loss function (best loss function for CoralLab)
        # total_loss = 0.5*focal_loss(y_true, y_pred) + 0.5*dice_loss(y_true, y_pred) # 0.5*focal + 0.5*dice
        total_loss = 1.0 * focal_loss(y_true, y_pred) + 0.25 * dice_loss(y_true, y_pred)  # 1*focal + 0.25*dice
    if loss_token == 11:  # F1-Score
        total_loss = 0.5*f1_loss(y_true, y_pred) + 0.5*dice_loss(y_true, y_pred)  # F1-score + Dice
    if loss_token == 12:  # Dice+Focal fusion loss function (best loss function for CoralLab)
        total_loss = 1 * focal_loss(y_true, y_pred) + 0.25 * iou_loss(y_true, y_pred)  # 0.5*focal + 0.5*dice

    return total_loss
