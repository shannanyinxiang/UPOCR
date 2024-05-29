from .swinv2_encoder import SwinV2Encoder

def build_encoder(args):
    train_encoder = args.lr_encoder_ratio > 0
    if args.encoder == 'swinv2':
        encoder = SwinV2Encoder(train_encoder, args.pretrained_encoder, args.swin_enc_embed_dim, args.swin_enc_depths, 
            args.swin_enc_num_heads, args.swin_enc_drop_path_rate, args.swin_enc_pretrained_ws, args.swin_enc_window_size,
            args.swin_use_checkpoint or args.swin_enc_use_checkpoint)
    else:
        raise NotImplementedError
    return encoder