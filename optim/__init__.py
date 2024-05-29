import yaml
import torch
from .criterion.text_removal import TextRemovalLoss
from .criterion.text_segmentation import TextSegLoss
from .criterion.tampered_text_detection import TTDLoss


def build_criterion(args):
    if not args.eval:
        return build_criterion_multitask(args)
    else:
        return build_criterion_singletask(args)


def build_criterion_multitask(args):
    criterion_dict = {}
    for data_cfg in args.data_cfgs:
        criterion = build_criterion_singletask(args, data_cfg['TYPE'])
        criterion_dict[data_cfg['TYPE']] = criterion 

    return criterion_dict

    
def build_criterion_singletask(args, task):
    return CRITERION_DICT[task](args).to(torch.device(args.device))


def build_optimizer(model, args):
    if args.distributed:
        model_without_ddp = model.module 
    else:
        model_without_ddp = model 
    
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if (not "encoder" in n) and p.requires_grad],
         "lr": args.lr},
        {"params": [p for n, p in model_without_ddp.named_parameters() if ("encoder" in n) and p.requires_grad], 
         "lr": args.lr * args.lr_encoder_ratio},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


CRITERION_DICT = {
    'text removal': TextRemovalLoss,
    'text segmentation': TextSegLoss,
    'tampered text detection': TTDLoss
}