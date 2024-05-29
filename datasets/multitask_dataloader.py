import torch

class MultiTaskDataloader(object):
    def __init__(self, dataloaders, samplers, epochs=None, args=None):
        self.dataloaders = dataloaders
        self.iters = [iter(dataloader) for dataloader in dataloaders]
        self.samplers = samplers 
        if epochs:
            self.epochs = epochs
        else:
            self.epochs = [0] * len(dataloaders)
    
        for sampler, epoch in zip(self.samplers, self.epochs):
            sampler.set_epoch(epoch)

        self.iter_per_epoch = args.iter_per_epoch

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration
        
        images = []
        tasks = []
        labels = {}
        for i in range(len(self.dataloaders)):
            data, dl_iter, epoch = get_next(self.iters[i], self.dataloaders[i], self.epochs[i], self.samplers[i])
            if epoch != self.epochs[i]:
                self.epochs[i] = epoch 
                self.iters[i] = dl_iter
            
            images.append(data.pop('image'))
            tasks.extend(data.pop('task'))
            labels[tasks[-1]] = data

        images = torch.cat(images)
        return images, labels, tasks
    
    def __len__(self):
        return self.iter_per_epoch


def get_next(dl_iter, dataloader, epoch, sampler):
    try:
        data = next(dl_iter)
    except StopIteration:
        epoch = epoch + 1
        sampler.set_epoch(epoch)
        dl_iter = iter(dataloader)
        data = next(dl_iter)
    
    return data, dl_iter, epoch