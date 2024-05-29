import torch
from .collate_fn import CollateFN
from .text_removal import build as build_textremoval_dataset
from .text_segmentation import build as build_textseg_dataset
from .tampered_text_detection import build as build_ttd_dataset


def build_dataset(image_set, args):
    if image_set == 'train' and (not args.eval):
        return build_dataset_multitask(args)
    elif image_set == 'val' and args.eval:
        return build_dataset_singletask(args.eval_data_cfg, args)
    else:
        raise ValueError


def build_dataloader(dataset, image_set, args):
    if image_set == 'train' and (not args.eval):
        return build_dataloader_multitask(dataset, image_set, args)
    elif image_set == 'val' and args.eval:
        return build_dataloader_singletask(dataset, image_set, args)
    else:
        raise ValueError


def build_dataset_multitask(args):
    dataset_dict = {}
    for data_cfg in args.data_cfgs:
        dataset = build_dataset_singletask(data_cfg, args)
        dataset_dict[data_cfg['TYPE']] = dataset 

    return dataset_dict


def build_dataset_singletask(data_cfg, args):
    return DATASET_BUILDER[data_cfg['TYPE']](data_cfg, args)


def build_dataloader_multitask(dataset_dict, image_set, args):
    tasks = []
    dataloaders = []
    samplers = []
    for task, dataset in dataset_dict.items():
        dataloader, sampler = build_dataloader_singletask(dataset, image_set, args)
        tasks.append(task)
        dataloaders.append(dataloader)
        samplers.append(sampler)
    
    return dataloaders, samplers

    
def build_dataloader_singletask(dataset, image_set, args):
    if args.distributed:
        shuffle = True if image_set == 'train' else False
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if image_set == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        elif image_set == 'val':
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    collate_fn = CollateFN()
    if image_set == 'train':
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=args.num_workers
        )
    elif image_set == 'val':
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers
        )

    return dataloader, sampler


DATASET_BUILDER = {
    'text removal': build_textremoval_dataset,
    'text segmentation': build_textseg_dataset,
    'tampered text detection': build_ttd_dataset    
}