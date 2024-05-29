import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.cuda.amp import autocast

class TextRemovalLoss(nn.Module):
    def __init__(self, args):
        super(TextRemovalLoss, self).__init__()
        self.style_loss_force_float32 = args.amp and args.amp_dtype in [torch.float16, None]
        if self.style_loss_force_float32:
            print('Force using float32 in style loss!')

    def forward(self, preds, mask_gt, gt):
        msr_loss = self.MSR_loss(preds['output'], mask_gt, gt)
        prc_loss = self.percetual_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'])

        if self.style_loss_force_float32:
            with autocast(enabled=False):
                style_loss = self.style_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'])
        else:
            style_loss = self.style_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'])

        losses = {'MSR_loss': msr_loss, 'prc_loss': prc_loss, 'style_loss': style_loss}
        
        return losses
    
    def percetual_loss(self, feat_output_comp, feat_output, feat_gt):
        pcr_losses = []
        for i in range(3):
            pcr_losses.append(F.l1_loss(feat_output[i], feat_gt[i]))
            pcr_losses.append(F.l1_loss(feat_output_comp[i], feat_gt[i]))
        return sum(pcr_losses)
    
    def style_loss(self, feat_output_comp, feat_output, feat_gt):
        style_losses = []
        for i in range(3):
            if self.style_loss_force_float32:
                feat_output_ = feat_output[i].float()
                feat_output_comp_ = feat_output_comp[i].float()
                feat_gt_ = feat_gt[i].float()
            else:
                feat_output_ = feat_output[i]
                feat_output_comp_ = feat_output_comp[i]
                feat_gt_ = feat_gt[i]
            style_losses.append(F.l1_loss(gram_matrix(feat_output_), gram_matrix(feat_gt_)))
            style_losses.append(F.l1_loss(gram_matrix(feat_output_comp_), gram_matrix(feat_gt_)))
        return sum(style_losses)

    def MSR_loss(self, outputs, mask, gt, scale_factors=[0.25, 0.5, 1.0], weights=[[5, 0.8], [6, 1], [10, 2]]):
        msr_losses = []
        for output, scale_factor, weight in zip(outputs, scale_factors, weights):
            if scale_factor != 1:
                mask_ = F.interpolate(mask, scale_factor=scale_factor, recompute_scale_factor=True)
                gt_ = F.interpolate(gt, scale_factor=scale_factor, recompute_scale_factor=True)
            else:
                mask_ = mask; gt_ = gt 
            msr_losses.append(weight[0] * F.l1_loss((1 - mask_) * output, (1 - mask_) * gt_))
            msr_losses.append(weight[1] * F.l1_loss(mask_ * output, mask_ * gt_))
        return sum(msr_losses)
            
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram