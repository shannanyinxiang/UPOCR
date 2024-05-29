def linear_decay_lr(args):
    warmup_lr = [args.warmup_min_lr + ((args.lr - args.warmup_min_lr) * i / args.warmup_epochs) for i in range(args.warmup_epochs)]
    decay_lr = [max(i * args.lr / args.epochs, args.min_lr) for i in range(args.epochs - args.warmup_epochs)]
    decay_lr.reverse()
    learning_rate_schedule = warmup_lr + decay_lr
    return learning_rate_schedule

def get_lr_schedule(args):
    return linear_decay_lr(args)