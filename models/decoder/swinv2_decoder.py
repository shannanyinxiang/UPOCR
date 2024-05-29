import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from collections import Iterable
from timm.models.layers import trunc_normal_

from models.block import build_lateral_connection, ConvWithActivation
from models.encoder.swin_transformer_v2 import SwinTransformerBlock, window_partition


class PatchSplit(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchSplit, self).__init__()
        self.dim = dim
        self.upsample = nn.Linear(dim // 4, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert(L == H * W and C % 4 == 0)

        x = x.reshape(B, H, W, 4, C//4)
        x = x[:, :, :, [0, 2, 1, 3], :]
        x = x.reshape(B, H, W, 2, 2, C//4)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H * 2 * W * 2, C//4)
        
        x = self.upsample(x)
        x = self.norm(x) # Swin V2 Post Norm

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, H, W):
        
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.upsample is not None:
            x_up = self.upsample(x, H, W)
            Wh, Ww = H * 2, W * 2
            return x, H, W, x_up, Wh, Ww
        else:
            return x, H, W, x, H, W

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerV2Decoder(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, 
                 encoder_dim=768,
                 embed_dim=768,
                 depths=[2, 6, 2, 2, 2],
                 num_heads=[24, 12, 6, 3, 2],
                 window_size=7, 
                 mlp_ratio=4., 
                 qkv_bias=True,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 use_checkpoint=False, 
                 pretrained_window_sizes=[0, 0, 0, 0, 0],
                 frozen_stages=-1,
                 skip_stages=None,
                 intermediate_output_stages=None):
        super(SwinTransformerV2Decoder, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.frozen_stages = frozen_stages
        self.skip_stages = skip_stages
        self.intermediate_output_stages = intermediate_output_stages

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 0.5 ** i_layer),
                               num_heads=num_heads[i_layer],
                               depth=depths[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               upsample=PatchSplit,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
            
        num_features = [int(embed_dim * 0.5 ** (i + 1)) for i in range(self.num_layers)]
        self.num_features = num_features
        
        self.output_conv = nn.Conv2d(num_features[self.num_layers - 1], 3, 3, 1, 1)

        if not self.skip_stages is None:
            self.lateral_connection_list = nn.ModuleList([
                build_lateral_connection(
                    int(encoder_dim * 0.5 ** (layer_idx + 1)), 
                    num_features[layer_idx] 
                )
                for layer_idx in to_layer_idx(self.skip_stages)
            ])

        if not self.intermediate_output_stages is None:
            self.intermediate_convs = nn.ModuleList([
                nn.Conv2d(num_features[layer_idx], 3, 3, 1, 1)
                for layer_idx in to_layer_idx(self.intermediate_output_stages)
            ])

        # add a norm layer for each output
        self.output_stages = self.intermediate_output_stages + [f'stage{self.num_layers + 1}']
        self.output_stages = list(set(self.output_stages))
        self.output_stages.sort()
        self.output_idx = to_layer_idx(self.output_stages)
        for layer_idx in self.output_idx:
            layer = norm_layer(num_features[layer_idx])
            layer_name = f'norm{layer_idx}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x, skip_features):
        """Forward function."""

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outputs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww)
            
            name = f'stage{i+2}'

            if name in self.skip_stages:
                skip_feature = self.lateral_connection_list[self.skip_stages.index(name)](skip_features[-i-2])
                skip_feature = skip_feature.flatten(2).transpose(1, 2)
                x = x + skip_feature

            if name in self.output_stages:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = x_out.view(-1, Wh, Ww, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

            if name in self.intermediate_output_stages:
                inter_conv = self.intermediate_convs[self.intermediate_output_stages.index(name)]
                inter_output = inter_conv(x_out)
                outputs.append(inter_output)

        output = self.output_conv(x_out)
        outputs.append(output)

        return outputs

def to_layer_idx(stages):
    if isinstance(stages, str):
        return int(stages.replace('stage', '')) - 2
    elif isinstance(stages, Iterable):
        return [int(stage.replace('stage', '')) - 2 for stage in stages]


def build_swin_v2_decoder(args):
    decoder = SwinTransformerV2Decoder(
        window_size=args.swin_dec_window_size, 
        mlp_ratio=4., 
        qkv_bias=True,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=args.swin_dec_drop_path_rate,
        norm_layer=nn.LayerNorm, 
        patch_norm=True,
        use_checkpoint=args.swin_use_checkpoint or args.swin_dec_use_checkpoint, 
        pretrained_window_sizes=[args.swin_dec_pretrained_ws] * len(args.swin_dec_depths),
        frozen_stages=-1,
        skip_stages=['stage2', 'stage3', 'stage4'],
        intermediate_output_stages=['stage4', 'stage5'] if args.intermediate_output else [],
        embed_dim=args.swin_enc_embed_dim * 8,
        depths=args.swin_dec_depths,
        num_heads=args.swin_dec_num_heads,
        encoder_dim=args.swin_enc_embed_dim * 8,
    )

    if args.pretrained_decoder:
        decoder = load_pretrained_decoder(decoder, args.pretrained_decoder)
    
    return decoder

def load_pretrained_decoder(decoder, pth_path):
    decoder_dict = decoder.state_dict()
    pth_dict = torch.load(pth_path, map_location='cpu')
    if 'model' in pth_dict:
        pth_dict = pth_dict['model']

    loaded_keys = []
    ignore_keys = []
    for dec_key in decoder_dict.keys():
        if 'relative_coords_table' in dec_key or \
           'relative_position_index' in dec_key:
            ignore_keys.append(dec_key)
            continue

        if dec_key in pth_dict.keys():
            decoder_dict[dec_key] = pth_dict[dec_key]
            loaded_keys.append(dec_key)
        
    missing_keys = [ele for ele in decoder_dict.keys() if not ele in loaded_keys and not ele in ignore_keys]
    unexpected_keys = [ele for ele in pth_dict.keys() if not ele in loaded_keys and not ele in ignore_keys]

    print(f'Load pretrained SwinTransformer Decoder weights from {pth_path}')
    print('Loaded keys: ', loaded_keys)
    print('Missing keys: ', missing_keys)
    print('Unexpected keys: ', unexpected_keys)
    print('Ignored keys:', ignore_keys)

    decoder.load_state_dict(decoder_dict)

    return decoder