import torch 

class CollateFN(object):
    def __call__(self, batch):
        batched_data = {}

        for k in batch[0].keys():
            batched_key_data = [ele[k] for ele in batch]
            if isinstance(batched_key_data[0], torch.Tensor):
                batched_key_data = torch.stack(batched_key_data)
            batched_data[k] = batched_key_data
        
        return batched_data