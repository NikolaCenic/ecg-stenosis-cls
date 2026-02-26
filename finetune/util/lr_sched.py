import math


def adjust_learning_rate_old(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.min_lr == args.blr:
        return args.min_lr
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    for param_group in optimizer.param_groups:
        if param_group["part"] == "head":
            base_lr = args.lr
        elif param_group["part"] == "backbone":
            base_lr = args.backbone_lr
        if args.min_lr == args.blr:
            return args.min_lr
        if epoch < args.warmup_epochs:
            lr = base_lr * epoch / args.warmup_epochs
        else:
            lr = args.min_lr + (base_lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.warmup_epochs)
                    / (args.epochs - args.warmup_epochs)
                )
            )

        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
