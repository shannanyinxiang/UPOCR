from .swinv2_decoder import build_swin_v2_decoder

def build_decoder(args):
    if args.decoder == 'swinv2':
        return build_swin_v2_decoder(args)
    raise NotImplementedError
