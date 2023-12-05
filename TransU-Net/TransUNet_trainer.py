### --- imports --- ###

from TransUNet_args import get_args

from rich import print
from rich.pretty import pprint

import torch

from utils.Prints import print_device_name
from utils.Prints import print_num_trainable_parameters
from utils.Prints import print_data_summary
from utils.Prints import print_end_of_epoch_summary
from utils.Prints import print_end_of_test_summary

from utils.Progress import get_progress_bar

import TransUNet_args as args

from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.vit_seg_modeling import VisionTransformer as ViT_seg

from torch import optim

from data.dataset_synapse import Synapse_dataset, RandomGenerator

from torchvision import transforms

from torch.utils.data import DataLoader

from rich.progress import *

from torch.nn.modules.loss import CrossEntropyLoss
from utils.DiceLoss import DiceLoss

from scipy.ndimage import zoom

import numpy as np

from utils.Metrics import calculate_dice_metric_per_case
from utils.Metrics import calculate_hausdorff_metric_per_case
from utils.Metrics import calculate_jaccard_metric_per_case
from utils.Metrics import metric_has_improved

import os

from utils.Model import get_num_trainable_parameters

import wandb
from wandb import wandb_run

from utils.Checkpointing import save_ckpt

### --- imports --- ###

################################################################################

### --- model --- ###

def _init_model(args: args, device) -> ViT_seg:
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    
    config_vit.n_classes = args.num_classes
    
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

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

### --- training --- ###

def _add_train_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_train: int):

    prog_bar_epochs_task = prog_bar.add_task(description=args.epochs_task_descr, total=args.num_epochs)
    prog_bar_train_batches_task = prog_bar.add_task(description=args.train_batches_task_descr, total=num_batches_train)

    return prog_bar_epochs_task, prog_bar_train_batches_task


def _train(
    args: args, prog_bar: Progress, device, model: torch.nn.Module, 
    optimizer: optim.SGD, dl_train: DataLoader, dl_val: DataLoader,
    wb_run: wandb_run
):
    
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

    for epoch in range(args.num_epochs):

        ### --- train step --- ###

        prog_bar.reset(prog_bar_train_batches_task)
        prog_bar.reset(prog_bar_val_batches_task)
        prog_bar.reset(prog_bar_val_slices_task)
        prog_bar.reset(prog_bar_val_metrics_task)

        running_loss_ce_train = 0
        running_loss_dice_train = 0
        running_loss_train = 0

        model.train()

        for batch_train in list(dl_train)[:num_batches_train]:

            img_batch_train  = batch_train["image"].to(device)
            gt_batch_train   = batch_train["label"].to(device)

            outputs = model(img_batch_train)

            step_loss_ce_train = ce_loss.forward(outputs, gt_batch_train[:].long())
            running_loss_ce_train += step_loss_ce_train
            step_loss_dice_train = dice_loss.forward(outputs, gt_batch_train, softmax=True)
            running_loss_dice_train += step_loss_dice_train
            loss_train = args.alpha * step_loss_ce_train + args.beta * step_loss_dice_train
            running_loss_train += loss_train

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            prog_bar.advance(prog_bar_train_batches_task, 1)
            prog_bar.advance(prog_bar_epochs_task, 1 / (num_batches_train))

        epoch_loss_ce_train = running_loss_ce_train / num_batches_train
        epoch_loss_dice_train = running_loss_dice_train / num_batches_train
        epoch_loss_train = running_loss_train / num_batches_train

        wb_run.log(
            {
                "epoch": epoch,
                "loss/train": epoch_loss_train,
                "loss_ce/train": epoch_loss_ce_train,
                "loss_dice/train": epoch_loss_dice_train,
            }
        )

        train_ce_loss_is_best   = metric_has_improved(epoch_loss_ce_train  , best_epoch_loss_ce_train  , "min")
        train_dice_loss_is_best = metric_has_improved(epoch_loss_dice_train, best_epoch_loss_dice_train, "min")
        train_loss_is_best      = metric_has_improved(epoch_loss_train     , best_epoch_loss_train     , "min")

        if train_ce_loss_is_best:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_ce_loss.pth")
        if train_dice_loss_is_best:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_dice_loss.pth")
        if train_loss_is_best:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_loss.pth")
        if epoch % args.log_every_n_epochs:
            save_ckpt(model, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_epoch_{epoch}.pth")


        ### --- train step --- ###
        
        ########################################################################

        ### --- validation step --- ###

        epoch_metric_dice_val, epoch_metric_jaccard_val = _validate(
            "val", epoch, args, device, model, dl_val, num_batches_val,
            prog_bar, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task,
            wb_run
        )

        val_dice_is_best    = metric_has_improved(epoch_metric_dice_val   , best_epoch_metric_dice_val, "max")
        val_jaccard_is_best = metric_has_improved(epoch_metric_jaccard_val, best_epoch_metric_jaccard_val, "max")

        # TODO add val checkpointing when train losses have been implemented in val as well

        ### --- validation step --- ###

        ########################################################################

        print_end_of_epoch_summary(
            args, epoch,
            epoch_loss_ce_train, train_ce_loss_is_best,
            epoch_loss_dice_train, train_dice_loss_is_best,
            epoch_metric_jaccard_val, val_jaccard_is_best,
            epoch_metric_dice_val, val_dice_is_best
        )

        prog_bar.update(task_id=prog_bar_epochs_task, total=args.num_epochs)



    ### --- epoch --- ###


### --- training --- ###

################################################################################

### --- validation --- ###

def _add_val_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_val: int):
    prog_bar_val_batches_task = prog_bar.add_task(description=args.val_batches_task_descr, total=num_batches_val)
    prog_bar_val_slices_task = prog_bar.add_task(description=args.val_slices_task_descr, total=69)
    prog_bar_val_metrics_task = prog_bar.add_task(description=args.val_metrics_task_descr, total=args.num_classes)

    return prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task


def _validate(
    inference_type: str, epoch: int, args: args, device, model: torch.nn.Module, dl_val: DataLoader, num_batches_val: int,
    prog_bar: Progress, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task,
    wb_run: wandb_run
):
    
    if inference_type not in args.INFERENCE_TYPES:
        raise ValueError(f"{inference_type} is an invalid inference type. Supported values: {args.INFERENCE_TYPES}")

    # one list element --> one segmentation class
    # NOTE about indexing!
    # Paper excludes black class from segmentation metrics, so we index
    # running_metric_{dice,jaccard}_val accordingly!
    running_metric_dice_val    = [0] * args.num_classes_for_metrics
    running_metric_jaccard_val = [0] * args.num_classes_for_metrics
    
    model.eval()

    for batch_val in list(dl_val)[: num_batches_val]:
        
        img_batch_val = batch_val["image"] 
        label_batch_val = batch_val["label"]
        
        image = img_batch_val.squeeze(0).cpu().detach().numpy()
        label = label_batch_val.squeeze(0).cpu().detach().numpy()
        prediction = np.zeros_like(label)

        num_slices = image.shape[0]

        prog_bar.reset(task_id=prog_bar_val_slices_task, total=num_slices)
        prog_bar.reset(task_id=prog_bar_val_metrics_task, total=args.num_classes_for_metrics)

        for ind in range(num_slices):
            slice = image[ind, :, :]

            x, y = slice.shape[0], slice.shape[1]
            
            if x != args.img_size or y != args.img_size:
                slice = zoom(slice, (args.img_size / x, args.img_size / y), order=3)

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                outputs = model(input)

                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                if x != args.img_size or y != args.img_size:
                    pred = zoom(out, (x / args.img_size, y / args.img_size), order=0)
                else:
                    pred = out

                prediction[ind] = pred

            prog_bar.advance(task_id=prog_bar_val_slices_task, advance=1)
        
        for c in range(1, args.num_classes):
            
            # NOTE about indexing!
            # Paper excludes black class from segmentation metrics, so we index
            # running_metric_{dice,jaccard}_val accordingly!
            running_metric_dice_val[c - 1]    += calculate_dice_metric_per_case(prediction == c, label == c)
            running_metric_jaccard_val[c - 1] += calculate_jaccard_metric_per_case(prediction == c, label == c)

            prog_bar.advance(task_id=prog_bar_val_metrics_task, advance=1)
        prog_bar.update(task_id=prog_bar_val_metrics_task, total=args.num_classes_for_metrics)

        prog_bar.update(task_id=prog_bar_val_batches_task, advance=1)

    epoch_metric_dice_val = np.array(running_metric_dice_val) / num_batches_val
    epoch_metric_jaccard_val = np.array(running_metric_jaccard_val) / num_batches_val

    # averaging across segmentation classes
    epoch_metric_dice_val = np.mean(epoch_metric_dice_val, axis=0)
    epoch_metric_jaccard_val = np.mean(epoch_metric_jaccard_val, axis=0)

    wb_run.log(
        {
            "epoch": epoch,
            f"metric_dice/{inference_type}": epoch_metric_dice_val,
            f"metric_jaccard/{inference_type}": epoch_metric_jaccard_val
        }
    )

    return epoch_metric_dice_val, epoch_metric_jaccard_val

### --- validation --- ###

################################################################################

### --- test --- ###

def _add_test_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_test: int):
    prog_bar_test_batches_task = prog_bar.add_task(description=args.test_batches_task_descr, total=num_batches_test)
    prog_bar_test_slices_task = prog_bar.add_task(description=args.test_slices_task_descr, total=69)
    prog_bar_test_metrics_task = prog_bar.add_task(description=args.test_metrics_task_descr, total=args.num_classes)

    return prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task

def _perform_testing(
    args: args, prog_bar: Progress, device, model: torch.nn.Module, dl_test: DataLoader,
    wb_run: wandb_run
):
    
    num_batches_test = int(len(dl_test) * args.lim_num_batches_percent_test)
    if num_batches_test == 0:
        num_batches_test = 1

    prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task = _add_test_prog_bar_tasks(args, prog_bar, num_batches_test)
    
    return _validate(
        "test", args.num_epochs, args, device, model, dl_test, num_batches_test, 
        prog_bar, prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task,
        wb_run
    )

def _test(args: args, prog_bar: Progress, device: torch.cuda.device, model: torch.nn.Module, dl_test: DataLoader, wb_run: wandb_run):
    best_epoch_metric_jaccard_test = 0
    best_epoch_metric_dice_test = 0
    
    epoch_metric_dice_test, epoch_metric_jaccard_test = _perform_testing(args, prog_bar, device, model, dl_test, wb_run)

    test_dice_is_best    = metric_has_improved(epoch_metric_dice_test   , best_epoch_metric_dice_test, "max")
    test_jaccard_is_best = metric_has_improved(epoch_metric_jaccard_test, best_epoch_metric_jaccard_test, "max")
    
    print_end_of_test_summary(
        args, 
        epoch_metric_jaccard_test, test_jaccard_is_best,
        epoch_metric_dice_test, test_dice_is_best
    )


### --- test --- ###

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

    wandb_run.watch(model, log='all')

    return wandb_run


### --- Weights and Biases --- ###

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

    optimizer = optim.SGD(
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

    # testing
    _test(args, prog_bar, device, model, dl_test, wb_run)


    





if __name__ == "__main__":
    main()

### --- main --- ###

################################################################################