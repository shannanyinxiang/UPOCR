import torch 
import wandb
import torch.cuda.amp as amp

from typing import Iterable 
from utils.dist import reduce_dict, is_main_process
from utils.logger import MetricLogger, SmoothedValue

def train_one_epoch(
      model: torch.nn.Module, 
      criterion: torch.nn.Module, 
      data_loader: Iterable, 
      optimizer,
      epoch: int,
      lr_scheduler: list = [0], 
      scaler = None,
      args = None):
     
    model.train()
    for criterion_module in criterion.values():
        criterion_module.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    optimizer.param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer.param_groups[1]['lr'] = lr_scheduler[epoch] * args.lr_encoder_ratio

    device = torch.device(args.device)

    for step, (images, labels, tasks) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        

        task_set = set(tasks)
        images = images.to(device)

        with amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
            outputs = model(images, tasks=tasks)

            loss_dict = {}
            ignore_keys = []
            if 'text removal' in task_set:
                loss_dict_textremoval, ignore_keys_textremoval = cal_textremoval_loss(
                    model=model,
                    criterion=criterion['text removal'],
                    images=images,
                    outputs=outputs,
                    labels=labels['text removal'],
                    tasks=tasks,
                ) 
                loss_dict_textremoval = {'textremoval_' + k:v for k, v in loss_dict_textremoval.items()}
                ignore_keys_textremoval = ['textremoval_' + k for k in ignore_keys_textremoval]
                loss_dict.update(loss_dict_textremoval)
                ignore_keys.extend(ignore_keys_textremoval)
            
            if 'text segmentation' in task_set:
                loss_dict_textseg, ignore_keys_textseg = cal_textseg_loss(
                    criterion=criterion['text segmentation'],
                    outputs=outputs,
                    labels=labels['text segmentation'],
                    tasks=tasks
                )
                loss_dict_textseg = {'textseg_' + k:v for k, v in loss_dict_textseg.items()}
                ignore_keys_textseg = ['textseg_' + k for k in ignore_keys_textseg]
                loss_dict.update(loss_dict_textseg)
                ignore_keys.extend(ignore_keys_textseg)
            
            if 'tampered text detection' in task_set:
                loss_dict_ttd, ignore_keys_ttd = cal_ttd_loss(
                    criterion=criterion['tampered text detection'],
                    outputs=outputs,
                    labels=labels['tampered text detection'],
                    tasks=tasks
                )
                loss_dict_ttd = {'ttd_' + k:v for k, v in loss_dict_ttd.items()}
                ignore_keys_ttd = ['ttd_' + k for k in ignore_keys_ttd]
                loss_dict.update(loss_dict_ttd)
                ignore_keys.extend(ignore_keys_ttd)

            G_loss = sum([loss_dict[k] for k in loss_dict.keys() if not k in ignore_keys])

        # wandb for logging
        if args.wandb and is_main_process():
            wandb_logging(loss_dict, step=step + epoch * args.iter_per_epoch)

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(G_loss).backward()
            if args.clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            G_loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss_dict_reduced.values())

        loss_value = losses_reduced.item()
    
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cal_textremoval_loss(
      model, 
      criterion,
      images,
      outputs, 
      labels, 
      tasks):
    task_index = torch.tensor([task == 'text removal' for task in tasks])
    preds = {}
    preds['output'] = [_[task_index] for _ in outputs]
    images = images[task_index]

    gt_masks = labels['gt_mask'].to(preds['output'][-1].device)
    labels = labels['label'].to(preds['output'][-1].device)
    output_comp = gt_masks * images + (1 - gt_masks) * preds['output'][-1]
    preds['feat_output_comp'] = model.module.vgg16(output_comp)
    preds['feat_output'] = model.module.vgg16(preds['output'][-1])
    preds['feat_gt'] = model.module.vgg16(labels)

    loss_dict = criterion(preds, gt_masks, labels)
    weight_dict = {'MSR_loss': 1, 'prc_loss': 0.01, 'style_loss': 120} 
    for k in loss_dict.keys():
        loss_dict[k] *= weight_dict[k]

    return loss_dict, []


def cal_textseg_loss(
      criterion,
      outputs,
      labels,
      tasks):
    task_index = torch.tensor([task == 'text segmentation' for task in tasks])
    
    output = [_[task_index] for _ in outputs]
    target = labels['label'].to(output[0].device)

    loss_dict = criterion(
        pred=output,
        target=target,
        ignore_value=128
    )
    
    return loss_dict, ['pos_loss', 'neg_loss']


def cal_ttd_loss(
      criterion,
      outputs,
      labels,
      tasks):
    task_index = torch.tensor([task == 'tampered text detection' for task in tasks])

    output = [_[task_index] for _ in outputs]
    target = labels['label'].to(output[0].device)

    loss_dict = criterion(
        pred=output,
        target=target,
        ignore_value=128
    )

    return loss_dict, []


def wandb_logging(loss_dict, step):
    for d in loss_dict.items():
        wandb.log(d, step)
