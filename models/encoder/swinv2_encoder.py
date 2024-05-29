# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torch.nn as nn

from utils.dist import is_main_process
from .swin_transformer_v2 import build_swinv2_encoder


class SwinV2Encoder(nn.Module):
    def __init__(self, train_backbone, weight_path, embed_dim, depths, num_heads,
        drop_path_rate, pretrained_ws, window_size, use_checkpoint):
        super(SwinV2Encoder, self).__init__()
        self.backbone = build_swinv2_encoder(
                            embed_dim=embed_dim, 
                            depths=depths, 
                            num_heads=num_heads, 
                            drop_path_rate=drop_path_rate, 
                            pretrained_ws=pretrained_ws, 
                            window_size=window_size, 
                            use_checkpoint=use_checkpoint)
        
        if not train_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)

        if is_main_process() and weight_path != '':
            self.load_pretrained_weights(weight_path)
        self.num_channels = embed_dim * 8
    
    def forward(self, input):
        feats = self.backbone(input)
        return feats

    def load_pretrained_weights(self, pth_path):
        model_dict = self.backbone.state_dict()
        pth_dict = torch.load(pth_path, map_location='cpu')
        if 'model' in pth_dict:
            pth_dict = pth_dict['model']

        loaded_keys = []
        ignore_keys = []
        for model_key in model_dict.keys():
            if 'relative_coords_table' in model_key or \
               'relative_position_index' in model_key:
                ignore_keys.append(model_key)
                continue
            if model_key in pth_dict.keys():
                model_dict[model_key] = pth_dict[model_key]
                loaded_keys.append(model_key)
            elif 'cpb_mlp' in model_key:
                model_key_rn = model_key.replace('cpb_mlp', 'rpe_mlp')
                if model_key_rn in pth_dict.keys():
                    model_dict[model_key] = pth_dict[model_key_rn]
                    loaded_keys.append(model_key)
                    loaded_keys.append(model_key_rn)

        missing_keys = [ele for ele in model_dict.keys() if not ele in loaded_keys and not ele in ignore_keys]
        unexpected_keys = [ele for ele in pth_dict.keys() if not ele in loaded_keys and not ele in ignore_keys]

        print(f'Load pretrained SwinTransformer weights from {pth_path}')
        print('Loaded keys: ', loaded_keys)
        print('Missing keys: ', missing_keys)
        print('Unexpected keys: ', unexpected_keys)
        print('Ignored keys:', ignore_keys)

        self.backbone.load_state_dict(model_dict)