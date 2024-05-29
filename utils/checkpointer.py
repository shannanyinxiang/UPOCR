import torch
from utils.dist import is_main_process

class Checkpointer(object):
    def __init__(self, distributed):
        self.distributed = distributed 

    def load(self, checkpoint_path, model, optimizer=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if self.distributed:
            model = model.module 
        model.load_state_dict(checkpoint['model'])
        
        if not optimizer is None:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0 
        
        if 'dl_epoch' in checkpoint:
            dl_epoch = checkpoint['dl_epoch']
        else:
            dl_epoch = None

        return start_epoch, dl_epoch

    def save(self, checkpoint_path, model, optimizer, epoch, dl_epoch, args):
        if not is_main_process():
            return 

        if self.distributed:
            model = model.module 
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args
        }
        if not dl_epoch is None:
            checkpoint['dl_epoch'] = dl_epoch
            
        torch.save(checkpoint, checkpoint_path)
