import torch
import torch.nn as nn


class TaskPrompt(nn.Module):
    def __init__(self, task_list, channel):
        super(TaskPrompt, self).__init__()
        self.task_embedding = nn.Embedding(len(task_list), channel)
        self.task_list = task_list

        nn.init.kaiming_normal_(self.task_embedding.weight)

    def forward(self, tasks, size):
        task_ids = torch.tensor([self.task_list.index(task) for task in tasks], dtype=torch.long).to(torch.device('cuda'))
        task_embeddings = self.task_embedding(task_ids)
        task_embeddings = task_embeddings.unsqueeze(-1).unsqueeze(-1)
        task_embeddings = task_embeddings.expand(-1, -1, size[0], size[1])

        return task_embeddings
    

def build_task_prompt(args):
    return TaskPrompt(
        task_list=args.tasks,
        channel=args.swin_enc_embed_dim * 8
    )