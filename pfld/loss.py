import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, euler_angle_gt, angle,
                landmarks):
        weight_angle = torch.sqrt(torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1))
        l2_distant = torch.sqrt(torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1))/384
        return torch.mean(weight_angle *
                          l2_distant), torch.mean(l2_distant)


def smoothL1(y_true, y_pred, beta=1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae > beta, mae - 0.5 * beta,
                                 0.5 * mae**2 / beta),
                     axis=-1)
    return torch.mean(loss)


def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK=106):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2)

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x,
                         w * torch.log(1.0 + absolute_x / epsilon),
                         absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    return loss