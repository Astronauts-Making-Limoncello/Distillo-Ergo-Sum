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

from scipy.ndimage import zoom

import numpy as np

from utils.Metrics import calculate_dice_metric_per_case, calculate_hausdorff_metric_per_case, calculate_jaccard_metric_per_case


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

def _get_dataset_val(args: args) -> Synapse_dataset:
    return Synapse_dataset(
        base_dir=args.volume_path, list_dir=args.list_dir, split="val_vol"
    )

def _get_dataloader_val(args: args, dataset_val: Synapse_dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset_val, batch_size=1, shuffle=True, 
        num_workers=1, pin_memory=args.pin_memory
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

    num_batches_train = len(dataloader_train) * args.lim_num_batches_percent_train
    if num_batches_train < 1:
        num_batches_train = 1
    
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

        for batch_train in list(dataloader_train)[:num_batches_train]:

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

### --- validation --- ###

def _validate(
    args: args, prog_bar: Progress, device, model: torch.nn.Module,
    dataloader_val: DataLoader
):

    num_batches_val = len(dataloader_val) * args.lim_num_batches_percent_val
    if num_batches_val < 1:
        num_batches_val = 1

    prog_bar_val_batches_task = prog_bar.add_task(description=args.val_batches_task_descr, total=num_batches_val)
    prog_bar_val_slices_task = prog_bar.add_task(description=args.val_slices_task_descr, total=69)
    prog_bar_val_metrics_task = prog_bar.add_task(description=args.val_metrics_task_descr, total=args.num_classes)

    # one list element --> one segmentation class
    running_metric_dice_val    = [0] * args.num_classes
    running_metric_jaccard_val = [0] * args.num_classes
    
    model.eval()

    for batch_val in list(dataloader_val)[: num_batches_val]:
        
        img_batch_val = batch_val["image"] 
        label_batch_val = batch_val["label"]
        
        image = img_batch_val.squeeze(0).cpu().detach().numpy()
        label = label_batch_val.squeeze(0).cpu().detach().numpy()
        prediction = np.zeros_like(label)

        num_slices = image.shape[0]

        prog_bar.reset(task_id=prog_bar_val_slices_task, total=num_slices)
        prog_bar.reset(task_id=prog_bar_val_metrics_task, total=args.num_classes)

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

            running_metric_dice_val[c]    += calculate_dice_metric_per_case(prediction == c, label == c)
            running_metric_jaccard_val[c] += calculate_jaccard_metric_per_case(prediction == c, label == c)

            prog_bar.advance(task_id=prog_bar_val_metrics_task, advance=1)


        
        prog_bar.update(task_id=prog_bar_val_batches_task, advance=1)

### --- validation --- ###

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
    print(f"Number of train batches  : {len(dataloader_train)}\n")

    dataset_val = _get_dataset_val(args)
    print(f"Number of val cases (volumes): {len(dataset_val)}\n")

    dataloader_val = _get_dataloader_val(args, dataset_val)
    print(f"Number of val batches        : {len(dataloader_val)}\n")

    # progress
    prog_bar = get_progress_bar()
    prog_bar.start()

    # validation TEMP POSITION

    _validate(args, prog_bar, device, model, dataloader_val)

    exit()

    # training (includes validation!)
    _train(args, prog_bar, device, model, optimizer, dataloader_train)





if __name__ == "__main__":
    main()

### --- main --- ###

################################################################################