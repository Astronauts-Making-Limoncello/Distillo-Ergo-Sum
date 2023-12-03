### --- imports --- ###

from TransUNet_args import get_args

from rich import print
from rich.pretty import pprint

import torch

from utils.Prints import print_device_name, print_num_trainable_parameters
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

def _get_dataset_train(args: args) -> Synapse_dataset:
    
    return Synapse_dataset(
        base_dir=args.root_path, list_dir=args.list_dir, split="train",
        transform=transforms.Compose(
            [
                RandomGenerator(output_size=[args.img_size, args.img_size])
            ]
        )
    )

def _get_dataloader_train(args: args, dataset_train: Synapse_dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=args.pin_memory
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
    optimizer: optim.SGD, dataloader_train: DataLoader
):

    num_batches_train = len(dataloader_train)
    prog_bar_epochs_task, prog_bar_train_batches_task = _add_train_prog_bar_tasks(args, prog_bar, num_batches_train)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    best_epoch_loss_ce_train = torch.inf
    best_epoch_loss_dice_train = torch.inf

    ### --- epoch --- ###

    for epoch in range(args.num_epochs):

        ### --- train step --- ###

        prog_bar.reset(prog_bar_train_batches_task)

        running_loss_ce_train = 0
        running_loss_dice_train = 0
        running_loss_train = 0

        model.train()

        for batch_train in dataloader_train:

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

        epoch_loss_ce_train_is_best_str   = "    "
        epoch_loss_dice_train_is_best_str = "    "
        if epoch_loss_ce_train < best_epoch_loss_ce_train:
            best_epoch_loss_ce_train = epoch_loss_ce_train
            epoch_loss_ce_train_is_best_str = args.loss_is_best_str
        if epoch_loss_dice_train < best_epoch_loss_dice_train:
            best_epoch_loss_dice_train = epoch_loss_dice_train
            epoch_loss_dice_train_is_best_str = args.loss_is_best_str

        ### --- train step --- ###
        
        ########################################################################

        ### --- validation step --- ###

        ### --- validation step --- ###

        ########################################################################

        prog_bar.update(task_id=prog_bar_epochs_task, total=args.num_epochs)

        print(
            f"[b][{args.epochs_color}]{epoch}[/{args.epochs_color}][/b] | train | "
            f"ce [b][{args.train_batches_color}]{round(epoch_loss_ce_train.item(), args.ndigits)}[/{args.train_batches_color}][/b] {epoch_loss_ce_train_is_best_str}, "
            f"dice [b][{args.train_batches_color}]{round(epoch_loss_dice_train.item(), args.ndigits)}[/{args.train_batches_color}][/b] {epoch_loss_dice_train_is_best_str}"
        )


    ### --- epoch --- ###


### --- training --- ###

################################################################################

### --- main --- ###

def main():

    # args
    # args = get_args()
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
    dataset_train = _get_dataset_train(args)
    print(f"Number of training slices: {len(dataset_train)}\n")

    dataloader_train = _get_dataloader_train(args, dataset_train)
    print(f"Number of train batches: {len(dataloader_train)}\n")

    # progress
    prog_bar = get_progress_bar()
    prog_bar.start()

    # training (includes validation!)
    _train(args, prog_bar, device, model, optimizer, dataloader_train)





if __name__ == "__main__":
    main()

### --- main --- ###

################################################################################