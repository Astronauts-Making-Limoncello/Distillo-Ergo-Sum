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
        base_dir=args.train_root_path, list_dir=args.list_dir, split="train",
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
        base_dir=args.val_volume_path, list_dir=args.list_dir, split="val_vol"
    )

def _get_dataloader_val(args: args, dataset_val: Synapse_dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset_val, batch_size=1, shuffle=True, 
        num_workers=1, pin_memory=args.pin_memory
    )

def _get_dataset_test(args: args) -> Synapse_dataset:
    return Synapse_dataset(
        base_dir=args.test_volume_path, list_dir=args.list_dir, split="test_vol"
    )

def _get_dataloader_test(args: args, dataset_test: Synapse_dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=True, 
        num_workers=1, pin_memory=args.pin_memory
    )

### --- data --- ###

################################################################################

### --- training --- ###

def _add_train_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_train: int):

    prog_bar_epochs_task = prog_bar.add_task(description=args.epochs_task_descr, total=args.num_epochs)
    prog_bar_train_batches_task = prog_bar.add_task(description=args.train_batches_task_descr, total=num_batches_train)

    return prog_bar_epochs_task, prog_bar_train_batches_task


def _handle_train_metrics_update(
        epoch_loss_ce_train  , best_epoch_loss_ce_train,
        epoch_loss_dice_train, best_epoch_loss_dice_train
):
    train_ce_is_best, train_dice_is_best = False, False

    if epoch_loss_ce_train < best_epoch_loss_ce_train:
        best_epoch_loss_ce_train = epoch_loss_ce_train
        train_ce_is_best = True
    if epoch_loss_dice_train < best_epoch_loss_dice_train:
        best_epoch_loss_dice_train = epoch_loss_dice_train
        train_dice_is_best = True

    return train_ce_is_best, train_dice_is_best


def _train(
    args: args, prog_bar: Progress, device, model: torch.nn.Module, 
    optimizer: optim.SGD, dataloader_train: DataLoader, dataloader_val: DataLoader
):

    num_batches_train = int(len(dataloader_train) * args.lim_num_batches_percent_train)
    if num_batches_train == 0:
        num_batches_train = 1

    num_batches_val = int(len(dataloader_val) * args.lim_num_batches_percent_val)
    if num_batches_val == 0:
        num_batches_val = 1
    
    prog_bar_epochs_task, prog_bar_train_batches_task = _add_train_prog_bar_tasks(args, prog_bar, num_batches_train)
    prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task = _add_val_prog_bar_tasks(args, prog_bar, num_batches_val)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    best_epoch_loss_ce_train = torch.inf
    best_epoch_loss_dice_train = torch.inf
    best_epoch_metric_dice_val_mean_across_class = 0
    best_epoch_metric_jaccard_val_mean_across_class = 0

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

        train_ce_is_best, train_dice_is_best = _handle_train_metrics_update(
            epoch_loss_ce_train, best_epoch_loss_ce_train,
            epoch_loss_dice_train, best_epoch_loss_dice_train
        )

        ### --- train step --- ###
        
        ########################################################################

        ### --- validation step --- ###

        epoch_metric_dice_val, epoch_metric_jaccard_val = _validate(
            args, device, model, dataloader_val, num_batches_val,
            prog_bar, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task
        )

        epoch_metric_dice_val_mean_across_class = np.mean(epoch_metric_dice_val, axis=0)
        epoch_metric_jaccard_val_mean_across_class = np.mean(epoch_metric_jaccard_val, axis=0)

        val_dice_is_best, val_jaccard_is_best = _handle_val_metrics_update(
            epoch_metric_dice_val_mean_across_class, best_epoch_metric_dice_val_mean_across_class,
            epoch_metric_jaccard_val_mean_across_class, best_epoch_metric_jaccard_val_mean_across_class
        )

        ### --- validation step --- ###

        ########################################################################
        

        print(
            f"[b][{args.epochs_color}]{epoch:03d}[/{args.epochs_color}][/b] | train | "
            f"Cross-Entropy loss [b][{args.train_batches_color}]{epoch_loss_ce_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_ce_is_best else args.loss_is_not_best_str} | "
            f"Dice loss   [b][{args.train_batches_color}]{epoch_loss_dice_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_dice_is_best else args.loss_is_not_best_str} |"
            f"\n"
            f"    | val   | "
            f"Jaccard metric     [b][{args.val_batches_color}]{epoch_metric_jaccard_val_mean_across_class:02.6f}[/{args.val_batches_color}][/b] {args.metric_is_best_str if val_jaccard_is_best else args.metric_is_not_best_str} | "
            f"Dice metric [b][{args.val_batches_color}]{epoch_metric_dice_val_mean_across_class:02.6f}[/{args.val_batches_color}][/b] {args.metric_is_best_str if val_dice_is_best else args.metric_is_not_best_str} |"
        )

        if epoch + 1 != args.num_epochs:
            print()

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


def _handle_val_metrics_update(
    epoch_metric_dice_val_mean_across_class   , best_epoch_metric_dice_val_mean_across_class,
    epoch_metric_jaccard_val_mean_across_class, best_epoch_metric_jaccard_val_mean_across_class
):

    val_dice_is_best, val_jaccard_is_best = False, False
    
    if epoch_metric_dice_val_mean_across_class > best_epoch_metric_dice_val_mean_across_class:
        best_epoch_metric_dice_val_mean_across_class = epoch_metric_dice_val_mean_across_class
        val_dice_is_best = True

    if epoch_metric_jaccard_val_mean_across_class > best_epoch_metric_jaccard_val_mean_across_class:
        best_epoch_metric_jaccard_val_mean_across_class = epoch_metric_jaccard_val_mean_across_class
        val_jaccard_is_best = True

    return val_dice_is_best, val_jaccard_is_best

def _validate(
    args: args, device, model: torch.nn.Module, dataloader_val: DataLoader, num_batches_val: int,
    prog_bar: Progress, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task
):

    # one list element --> one segmentation class
    # NOTE about indexing!
    # Paper excludes black class from segmentation metrics, so we index
    # running_metric_{dice,jaccard}_val accordingly!
    running_metric_dice_val    = [0] * args.num_classes_for_metrics
    running_metric_jaccard_val = [0] * args.num_classes_for_metrics
    
    model.eval()

    for batch_val in list(dataloader_val)[: num_batches_val]:
        
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
        prog_bar.update(task_id=prog_bar_val_metrics_task, total=args.num_classes)

        prog_bar.update(task_id=prog_bar_val_batches_task, advance=1)

    epoch_metric_dice_val = np.array(running_metric_dice_val) / num_batches_val
    epoch_metric_jaccard_val = np.array(running_metric_jaccard_val) / num_batches_val

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
    args: args, prog_bar: Progress, device, model: torch.nn.Module, dataloader_test: DataLoader
):
    
    num_batches_test = int(len(dataloader_test) * args.lim_num_batches_percent_test)
    if num_batches_test == 0:
        num_batches_test = 1

    prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task = _add_test_prog_bar_tasks(args, prog_bar, num_batches_test)
    
    return _validate(
        args, device, model, dataloader_test, num_batches_test, 
        prog_bar, prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task
    )

def _test(args: args, prog_bar: Progress, device: torch.cuda.device, model: torch.nn.Module, dataloader_test: DataLoader):
    best_epoch_metric_jaccard_test_mean_across_class = 0
    best_epoch_metric_dice_test_mean_across_class = 0
    epoch_metric_dice_test, epoch_metric_jaccard_test = _perform_testing(args, prog_bar, device, model, dataloader_test)
    
    epoch_metric_dice_test_mean_across_class = np.mean(epoch_metric_dice_test, axis=0)
    epoch_metric_jaccard_test_mean_across_class = np.mean(epoch_metric_jaccard_test, axis=0)

    epoch_metric_jaccard_test_mean_across_class_is_best_str = "          "
    epoch_metric_dice_test_mean_across_class_is_best_str    = "          "
    if epoch_metric_jaccard_test_mean_across_class > best_epoch_metric_jaccard_test_mean_across_class:
        epoch_metric_jaccard_test_mean_across_class_is_best_str = args.metric_is_best_str
        best_epoch_metric_jaccard_test_mean_across_class = epoch_metric_jaccard_test_mean_across_class
    if epoch_metric_dice_test_mean_across_class > best_epoch_metric_dice_test_mean_across_class:
        epoch_metric_dice_test_mean_across_class_is_best_str = args.metric_is_best_str
        best_epoch_metric_dice_test_mean_across_class = epoch_metric_dice_test_mean_across_class

    
    print(
        f"    | test  | "
        f"Jaccard metric     [b][{args.test_batches_color}]{epoch_metric_jaccard_test_mean_across_class:02.6f}[/{args.test_batches_color}][/b] {epoch_metric_jaccard_test_mean_across_class_is_best_str} | "
        f"Dice metric [b][{args.test_batches_color}]{epoch_metric_dice_test_mean_across_class:02.6f}[/{args.test_batches_color}][/b] {epoch_metric_dice_test_mean_across_class_is_best_str} |"
    )

    print()


### --- test --- ###

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
    print(f"Total number of train slices      : {len(dataset_train)}")

    dataloader_train = _get_dataloader_train(args, dataset_train)
    print(f"Total number of train batches     : {len(dataloader_train)}")
    print(f"Number of train batches limited to: {args.lim_num_batches_percent_train*100}%\n")

    dataset_val = _get_dataset_val(args)
    print(f"Total number of val cases (volumes): {len(dataset_val)}")

    dataloader_val = _get_dataloader_val(args, dataset_val)
    print(f"Total number of val batches        : {len(dataloader_val)}")
    print(f"Number of val batches limited to   : {args.lim_num_batches_percent_val*100}%\n")
    
    dataset_test = _get_dataset_test(args)
    print(f"Total number of test cases (volumes): {len(dataset_test)}")

    dataloader_test = _get_dataloader_test(args, dataset_test)
    print(f"Total number of test batches        : {len(dataloader_test)}")
    print(f"Number of test batches limited to   : {args.lim_num_batches_percent_test*100}%\n")


    # progress
    prog_bar = get_progress_bar()
    prog_bar.start()

    # training (and validation!)
    _train(args, prog_bar, device, model, optimizer, dataloader_train, dataloader_val)

    # testing
    _test(args, prog_bar, device, model, dataloader_test)


    





if __name__ == "__main__":
    main()

### --- main --- ###

################################################################################