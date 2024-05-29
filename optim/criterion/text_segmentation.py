import torch
import torch.nn as nn 
import torch.nn.functional as F

class TextSegLoss(nn.Module):
    def __init__(self, args):
        super(TextSegLoss, self).__init__()
    
    def forward(self, pred, target, ignore_value, multiscale_weights=[5, 6, 10]):
        l1_losses = []
        pos_losses = []
        neg_losses = []
        for pred_, weight in zip(pred, multiscale_weights):
            h, w = pred_.shape[2:]
            target_ = F.interpolate(target, (h, w), mode='nearest')
            l1_loss, pos_loss, neg_loss = \
                self.cal_l1_loss(pred_, target_, ignore_value)
            l1_losses.append(l1_loss * weight)
            pos_losses.append(pos_loss * weight)
            neg_losses.append(neg_loss * weight)
        l1_loss = sum(l1_losses)
        pos_loss = sum(pos_losses)
        neg_loss = sum(neg_losses)

        loss = {}
        loss['l1_loss'] = l1_loss
        loss['pos_loss'] = pos_loss
        loss['neg_loss'] = neg_loss
        return loss
    
    def cal_l1_loss(self, pred, target, ignore_value):
        l1_loss_mat = F.smooth_l1_loss(pred, target, reduction='none')

        pos_loss_list = l1_loss_mat[target == 1]
        if pos_loss_list.numel() == 0:
            pos_loss = torch.tensor(0).to(pred.device)
        else:
            pos_loss = pos_loss_list.mean()

        neg_loss_list = l1_loss_mat[target == 0]
        neg_loss = neg_loss_list.mean()
        
        l1_loss_mat = l1_loss_mat[target != ignore_value / 255]
        l1_loss = l1_loss_mat.mean()

        return l1_loss, pos_loss, neg_loss

            