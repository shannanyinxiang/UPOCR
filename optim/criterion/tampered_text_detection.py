import torch.nn as nn 
import torch.nn.functional as F

class TTDLoss(nn.Module):
    def __init__(self, args):
        super(TTDLoss, self).__init__()
    
    def forward(self, pred, target, ignore_value, multiscale_weights=[5, 6, 10]):
        l1_losses = []
        for pred_, weight in zip(pred, multiscale_weights):
            h, w = pred_.shape[2:]
            target_ = F.interpolate(target, (h, w), mode='nearest')
            l1_loss = self.cal_l1_loss(pred_, target_, ignore_value)
            l1_losses.append(l1_loss * weight)
        l1_loss = sum(l1_losses)

        loss = {}
        loss['l1_loss'] = l1_loss
        return loss
    
    def cal_l1_loss(self, pred, target, ignore_value):
        l1_loss_mat = F.smooth_l1_loss(pred, target, reduction='none')
        l1_loss_mat = l1_loss_mat[target != ignore_value / 255]
        l1_loss = l1_loss_mat.mean()
            
        return l1_loss