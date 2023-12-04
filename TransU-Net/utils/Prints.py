from rich import print

import torch

### --- device --- ###

def print_device_name(device: torch.cuda.device):
    
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        
        if "NVIDIA" in device_name:
            device_name = f"[#008000][b]{device_name}[/b][/#008000]"
        
        print(f"Device: {device_name}\n")

### --- device --- ###

################################################################################

### --- data --- ###

def print_data_summary(
    args, dataset_train, dataloader_train, dataset_val, dataloader_val, dataset_test, dataloader_test
):
    print(f"Total number of train slices      : {len(dataset_train)}")
    print(f"Total number of train batches     : {len(dataloader_train)}")
    print(f"Number of train batches limited to: {args.lim_num_batches_percent_train*100}%\n")
    print(f"Total number of val cases (volumes): {len(dataset_val)}")
    print(f"Total number of val batches        : {len(dataloader_val)}")
    print(f"Number of val batches limited to   : {args.lim_num_batches_percent_val*100}%\n")
    print(f"Total number of test cases (volumes): {len(dataset_test)}")
    print(f"Total number of test batches        : {len(dataloader_test)}")
    print(f"Number of test batches limited to   : {args.lim_num_batches_percent_test*100}%\n")


### --- data --- ###

################################################################################

### --- model --- ###

def print_num_trainable_parameters(model: torch.nn.Module, num_color: str = "#000000"):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Number of trainable parameters: [{num_color}][b]{n}[/b][/{num_color}]\n")

### --- model --- ###

################################################################################

### --- training --- ###

def print_end_of_epoch_summary(
    args, epoch,
    epoch_loss_ce_train, train_ce_is_best,
    epoch_loss_dice_train, train_dice_is_best,
    epoch_metric_jaccard_val, val_jaccard_is_best,
    epoch_metric_dice_val, val_dice_is_best
):
    print(
        f"[b][{args.epochs_color}]{epoch:03d}[/{args.epochs_color}][/b] | train | "
        f"Cross-Entropy loss [b][{args.train_batches_color}]{epoch_loss_ce_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_ce_is_best else args.loss_is_not_best_str} | "
        f"Dice loss   [b][{args.train_batches_color}]{epoch_loss_dice_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_dice_is_best else args.loss_is_not_best_str} |"
        f"\n"
        f"    | val   | "
        f"Jaccard metric     [b][{args.val_batches_color}]{epoch_metric_jaccard_val:02.6f}[/{args.val_batches_color}][/b] {args.metric_is_best_str if val_jaccard_is_best else args.metric_is_not_best_str} | "
        f"Dice metric [b][{args.val_batches_color}]{epoch_metric_dice_val:02.6f}[/{args.val_batches_color}][/b] {args.metric_is_best_str if val_dice_is_best else args.metric_is_not_best_str} |"
    )

    if epoch + 1 != args.num_epochs:
        print()

### --- training --- ###

################################################################################

### --- test --- ###

def print_end_of_test_summary(
        args, 
        epoch_metric_jaccard_test, test_jaccard_is_best,
        epoch_metric_dice_test, test_dice_is_best
    ):

    print(
        f"    | test  | "
        f"Jaccard metric     [b][{args.test_batches_color}]{epoch_metric_jaccard_test:02.6f}[/{args.test_batches_color}][/b] {args.metric_is_best_str if test_jaccard_is_best else args.metric_is_not_best_str} | "
        f"Dice metric [b][{args.test_batches_color}]{epoch_metric_dice_test:02.6f}[/{args.test_batches_color}][/b] {args.metric_is_best_str if test_dice_is_best else args.metric_is_not_best_str} |"
    )

    print()

################################################################################