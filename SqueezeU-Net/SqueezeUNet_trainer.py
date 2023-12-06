from rich import print
from rich.pretty import pprint

import SqueezeUnet_args as args

import torch

from utilities.Prints import print_device_name

from squeezeunet_torch import SqueezeUNet

from utilities.Prints import print_num_trainable_parameters

from data.dataset_synapse import Synapse_dataset

from torch.utils.data import DataLoader

from utilities.Prints import print_data_summary

import wandb

from utilities.Model import get_num_trainable_parameters

from utilities.Progress import get_progress_bar

from rich.progress import Progress

from wandb.wandb_run import Run

from torch.nn.modules.loss import CrossEntropyLoss
from utilities.DiceLoss import DiceLoss

import os

from utilities.Checkpointing import save_ckpt

### --- model --- ###

def _init_model(args: args, device) -> SqueezeUNet:
    
    model = SqueezeUNet(
        num_classes=args.num_classes, 
        channels_shrink_factor=args.squeeze_unet_channels_shrink_factor
    )
    model = model.to(device)

    print_num_trainable_parameters(model, args.epochs_color)
    
    return model


### --- model --- ###

################################################################################

### --- data --- ###

def _get_dataset(base_dir, list_dir, split, transform):
    return Synapse_dataset(
        base_dir=base_dir, list_dir=list_dir, split=split, transform=transform
    )

def _get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory) -> DataLoader:
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=pin_memory
    )

### --- data --- ###

################################################################################

### --- Weights and Biases --- ###

def _wandb_init(args: args, model: torch.nn.Module):

    wandb_config = args.get_args()
    wandb_config.update(
        {
            "num_trainable_parameters": get_num_trainable_parameters(model)
        }
    )

    wandb_run = wandb.init(project=args.project_name, config=wandb_config, mode=args.wandb_mode)

    if args.wandb_watch_model:
        wandb_run.watch(model, log='all')

    return wandb_run


### --- Weights and Biases --- ###

################################################################################

### --- training --- ###

def _add_train_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_train: int):

    prog_bar_epochs_task = prog_bar.add_task(description=args.epochs_task_descr, total=args.num_epochs)
    prog_bar_train_batches_task = prog_bar.add_task(description=args.train_batches_task_descr, total=num_batches_train)

    return prog_bar_epochs_task, prog_bar_train_batches_task


def _train(
    args: args, prog_bar: Progress, device, model: torch.nn.Module, 
    optimizer: torch.optim.SGD, dl_train: DataLoader, dl_val: DataLoader,
    wb_run: Run
):

    max_iterations = args.num_epochs * len(dl_train)
    iter_num = 0
    
    os.makedirs(args.checkpoint_dir) if not os.path.exists(args.checkpoint_dir) else None

    num_batches_train = int(len(dl_train) * args.lim_num_batches_percent_train)
    if num_batches_train == 0:
        num_batches_train = 1

    num_batches_val = int(len(dl_val) * args.lim_num_batches_percent_val)
    if num_batches_val == 0:
        num_batches_val = 1

    prog_bar_epochs_task, prog_bar_train_batches_task = _add_train_prog_bar_tasks(args, prog_bar, num_batches_train)
    prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task = _add_val_prog_bar_tasks(args, prog_bar, num_batches_val)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    best_epoch_loss_ce_train = torch.inf
    best_epoch_loss_dice_train = torch.inf
    best_epoch_loss_train = torch.inf
    
    best_epoch_metric_dice_val = 0
    best_epoch_metric_jaccard_val = 0

    ### --- epoch --- ###

    for epoch in range(1, args.num_epochs + 1):
        ### --- train step --- ###
        
        prog_bar.reset(prog_bar_train_batches_task)
        prog_bar.reset(prog_bar_val_batches_task)
        prog_bar.reset(prog_bar_val_slices_task)
        prog_bar.reset(prog_bar_val_metrics_task)

        running_loss_ce_train = 0
        running_loss_dice_train = 0
        running_loss_train = 0
        train_ce_loss_is_best = False
        train_dice_loss_is_best = False
        train_loss_is_best = False
        val_dice_is_best = False
        val_jaccard_is_best = False

        model.train()

        for batch_train in list(dl_train)[:num_batches_train]:

            img_batch_train  = batch_train["image"].to(device)
            gt_batch_train   = batch_train["label"].to(device)
            
            if args.train_transforms is None:
                img_batch_train = img_batch_train.unsqueeze(1)

            outputs = model(img_batch_train)
            # outputs.shape matches TransU-Net's output shape!!!

            step_loss_ce_train = ce_loss.forward(outputs, gt_batch_train[:].long())
            running_loss_ce_train += step_loss_ce_train
            step_loss_dice_train = dice_loss.forward(outputs, gt_batch_train, softmax=True)
            running_loss_dice_train += step_loss_dice_train
            loss_train = args.alpha * step_loss_ce_train + args.beta * step_loss_dice_train
            running_loss_train += loss_train

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            if args.use_lr_scheduler:
                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1

            prog_bar.advance(prog_bar_train_batches_task, 1)
            prog_bar.advance(prog_bar_epochs_task, 1 / (num_batches_train))

        epoch_loss_ce_train = running_loss_ce_train / num_batches_train
        epoch_loss_dice_train = running_loss_dice_train / num_batches_train
        epoch_loss_train = running_loss_train / num_batches_train

        wb_run.log(
            {
                "loss/full/train": epoch_loss_train,
                "loss/ce/train": epoch_loss_ce_train,
                "loss/dice/train": epoch_loss_dice_train,
            }
        )

        if epoch_loss_ce_train < best_epoch_loss_ce_train:
            best_epoch_loss_ce_train = epoch_loss_ce_train
            train_ce_loss_is_best = True
        if epoch_loss_dice_train < best_epoch_loss_dice_train:
            best_epoch_loss_dice_train = epoch_loss_dice_train
            train_dice_loss_is_best = True
        if epoch_loss_train < best_epoch_loss_train:
            best_epoch_loss_train = epoch_loss_train
            train_loss_is_best = True

        if train_ce_loss_is_best:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_ce_loss.pth")
        if train_dice_loss_is_best:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_dice_loss.pth")
        if train_loss_is_best:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_loss.pth")
        if epoch % args.log_every_n_epochs == 0:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_epoch_{epoch}.pth")
        
        ### --- train step --- ###
        
        ########################################################################

### --- training --- ###

################################################################################

### --- validation --- ###

def _add_val_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_val: int):
    prog_bar_val_batches_task = prog_bar.add_task(description=args.val_batches_task_descr, total=num_batches_val)
    prog_bar_val_slices_task = prog_bar.add_task(description=args.val_slices_task_descr, total=69)
    prog_bar_val_metrics_task = prog_bar.add_task(description=args.val_metrics_task_descr, total=args.num_classes)

    return prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task

### --- validation --- ###

################################################################################

### --- main --- ###

def main():

    # args
    print(f"Arguments:")
    pprint(args.get_args())
    print()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_name(device)

    # model
    model = _init_model(args, device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # data
    ds_train = _get_dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train", transform=args.train_transforms)
    dl_train = _get_dataloader(ds_train, args.batch_size, True, args.num_workers, pin_memory=args.pin_memory)
    ds_val = _get_dataset(base_dir=args.val_volume_path, list_dir=args.list_dir, split="val_vol", transform=args.val_transforms)
    dl_val = _get_dataloader(ds_val, 1, False, 1, True)
    ds_test = _get_dataset(base_dir=args.test_volume_path, list_dir=args.list_dir, split="test_vol", transform=args.test_transforms)
    dl_test = _get_dataloader(ds_test, 1, False, 1, True)
    print_data_summary(args, ds_train, dl_train, ds_val, dl_val, ds_test, dl_test)

    # Weights and Biases
    wb_run = _wandb_init(args, model)

    # progress
    prog_bar = get_progress_bar()
    prog_bar.start()

    # training (and validation!)
    _train(args, prog_bar, device, model, optimizer, dl_train, dl_val, wb_run)












if __name__ == "__main__":
    main()

### --- main --- ###

################################################################################