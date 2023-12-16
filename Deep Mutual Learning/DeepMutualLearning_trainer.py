# -----------------------------
# Import Libraries
# -----------------------------

import os
from rich import print
from rich.pretty import pprint

import torch

from torch.cuda import device

from utils.Prints import print_device_name
from utils.Prints import print_num_trainable_parameters
from utils.Prints import print_num_parameters
from utils.Prints import print_data_summary
from utils.Prints import print_end_of_epoch_summary
from utils.Prints import print_end_of_epoch_summary_mutual
from utils.Prints import print_end_of_test_summary
from data.dataset_synapse import Synapse_dataset
from torch.utils.data import DataLoader
import DeepMutualLearning_args as args
from utils.Progress import get_progress_bar
from utils.Model import get_num_trainable_parameters
from utils.Checkpointing import save_ckpt, load_ckpt, handle_resume_from_ckpt
import wandb
from rich.progress import Progress
import torch.nn as nn
from wandb.wandb_run import Run
import numpy as np
from scipy.ndimage import zoom
from utils.Metrics import calculate_dice_metric_per_case
from utils.Metrics import calculate_jaccard_metric_per_case

from torch.nn import Module
from torch.nn.modules import CrossEntropyLoss
from utils.DiceLoss import DiceLoss
from torch.nn.modules import KLDivLoss

from models.squeezeunet_torch import SqueezeUNet
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.vit_seg_modeling import VisionTransformer as ViT_seg

import time

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from torch.nn.functional import softmax, log_softmax

# -----------------------------
# Models
# -----------------------------

def load_teacher(args: args, device):
    teacher = torch.load(args.teacher_path, map_location=device)
    print_num_trainable_parameters(teacher, args.epochs_color)
    return teacher

def load_student(args: args, device):
    student = torch.load(args.student_path, map_location=device)
    print_num_trainable_parameters(student, args.epochs_color)
    return student

def _init_TransUNet_model(args: args, device) -> ViT_seg:
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    
    config_vit.n_classes = args.num_classes
    
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.trans_unet_img_size / args.vit_patches_size), int(args.trans_unet_img_size / args.vit_patches_size))
    
    model = ViT_seg(config_vit, img_size=args.trans_unet_img_size, num_classes=config_vit.n_classes)

    model = model.to(device)

    if args.trans_unet_freeze_all_params:
        model.freeze_all_params()

    print_num_parameters(model, args.epochs_color, "\[Teacher]")
    print_num_trainable_parameters(model, args.epochs_color, "\[Teacher]", "\n")

    return model


def _init_SqueezeUNet_model(args: args, device) -> SqueezeUNet:
    
    model = SqueezeUNet(
        num_classes=args.num_classes, 
        channels_shrink_factor=args.squeeze_unet_channels_shrink_factor
    )
    model = model.to(device)

    print_num_parameters(model, args.epochs_color, "\[Student]")
    print_num_trainable_parameters(model, args.epochs_color, "\[Student]", "\n")
    
    return model

### --- model --- ###

################################################################################

# -----------------------------
# Setup Hyperparameters
# -----------------------------

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

### --- optimizer --- ###

def _init_optimizer(args: args, model: Module) -> Optimizer:

    if args.optimizer_type == "Adam":
        return torch.optim.Adam(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, amsgrad=args.use_amsgrad)
    
    if args.optimizer_type == "SGD":
        return torch.optim.SGD(params=model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    raise ValueError(f"{args.optimizer_type} optimizer not supported. Valid options: SGD, Adam")

### --- optimizer --- ###

### --- learning rate scheduler --- ###

def _init_lr_scheduler(args: args, optimizer: Optimizer) -> MultiStepLR: 
    
    if args.use_lr_scheduler == True:
        if args.optimizer_type == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=args.milestones, gamma=args.gamma, 
                verbose=args.lr_scheduler_verbose
            )
        else:
            raise ValueError(f"{args.lr_scheduler_name} lr scheduler not supported. Supported: MultiStepLR")
    
    return None


### --- learning rate scheduler --- ###

### --- Weights and Biases --- ###

def _wandb_init(
    args: args, 
    model_teacher: Module, model_student: Module, 
    optimizer_teacher: Optimizer, optimizer_student: Optimizer, 
    lr_scheduler_teacher: MultiStepLR, lr_scheduler_student: MultiStepLR
):

    wandb_config = args.get_args()
    wandb_config.update(
        {
            "num_trainable_parameters_teacher": get_num_trainable_parameters(model_teacher),
            "num_trainable_parameters_student": get_num_trainable_parameters(model_student),
            "optimizer_teacher": str(optimizer_teacher),
            "optimizer_student": str(optimizer_student),
            "lr_scheduler_teacher": str(lr_scheduler_teacher),
            "lr_scheduler_student": str(lr_scheduler_student)
        }
    )

    wandb_run = wandb.init(
        project=args.wandb_project_name, group=args.wandb_group, job_type=args.wandb_job_type,
        config=wandb_config, mode=args.wandb_mode
    )

    if args.wandb_watch_model_teacher:
        wandb_run.watch(model_teacher, log='all')
    
    if args.wandb_watch_model_student:
        wandb_run.watch(model_student, log='all')

    return wandb_run


### --- Weights and Biases --- ###

# -----------------------------
# Perform Training
# -----------------------------


### --- training --- ###

def _add_train_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_train: int):

    prog_bar_epochs_task = prog_bar.add_task(description=args.epochs_task_descr, total=args.num_epochs)
    prog_bar_train_batches_task = prog_bar.add_task(description=args.train_batches_task_descr, total=num_batches_train)

    return prog_bar_epochs_task, prog_bar_train_batches_task

def _add_val_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_val: int):
    prog_bar_val_batches_task = prog_bar.add_task(description=args.val_batches_task_descr, total=num_batches_val)
    prog_bar_val_slices_task = prog_bar.add_task(description=args.val_slices_task_descr, total=69)
    prog_bar_val_metrics_task = prog_bar.add_task(description=args.val_metrics_task_descr, total=args.num_classes)

    return prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task

def train(
    args: args, starting_epoch: int, prog_bar: Progress, device: device, 
    model_teacher: Module, model_student: Module,
    optimizer_teacher: Optimizer, optimizer_student: Optimizer,
    lr_scheduler_teacher: MultiStepLR, lr_scheduler_student: MultiStepLR, 
    dl_train: DataLoader, dl_val: DataLoader, 
    wb_run: Run
) -> str:
    
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
    kl_loss = KLDivLoss(reduction="batchmean")

    model_teacher.train()  
    model_student.train()

    best_epoch_loss_ce_train_teacher = torch.inf
    best_epoch_loss_dice_train_teacher = torch.inf
    best_epoch_loss_distill_train_teacher = torch.inf
    best_epoch_loss_train_teacher = torch.inf
    
    best_epoch_loss_ce_train_student = torch.inf
    best_epoch_loss_dice_train_student = torch.inf
    best_epoch_loss_distill_train_student = torch.inf
    best_epoch_loss_train_student = torch.inf
    
    best_epoch_metric_dice_val_teacher = 0
    best_epoch_metric_dice_val_teacher = 0
    best_epoch_metric_jaccard_val_student = 0
    best_epoch_metric_jaccard_val_student = 0

    ### --- epoch --- ###

    for epoch in range(starting_epoch, starting_epoch + args.num_epochs + 1):
        ### --- train step --- ###
        
        prog_bar.reset(prog_bar_train_batches_task)
        prog_bar.reset(prog_bar_val_batches_task)
        prog_bar.reset(prog_bar_val_slices_task)
        prog_bar.reset(prog_bar_val_metrics_task)   

        running_loss_ce_train_teacher = 0
        running_loss_dice_train_teacher = 0
        running_loss_distill_train_teacher = 0
        running_loss_train_teacher = 0
        running_loss_ce_train_student = 0
        running_loss_dice_train_student = 0
        running_loss_distill_train_student = 0
        running_loss_train_student = 0
        
        train_ce_loss_is_best_teacher = False
        train_dice_loss_is_best_teacher = False
        train_soft_targets_loss_is_best_teacher = False
        train_loss_is_best_teacher = False
        val_dice_is_best_teacher = False
        val_jaccard_is_best_teacher = False
        train_ce_loss_is_best_student = False
        train_dice_loss_is_best_student = False
        train_soft_targets_loss_is_best_student = False
        train_loss_is_best_student = False
        val_dice_is_best_student = False
        val_jaccard_is_best_student = False

        for batch_train in list(dl_train)[:num_batches_train]:

            img_batch_train = batch_train["image"].to(device)
            gt_batch_train = batch_train["label"].to(device)
            
            optimizer_teacher.zero_grad()

            outputs_teacher = model_teacher(img_batch_train)
            outputs_student = model_student(img_batch_train)

            # teacher cross-entropy loss
            step_loss_ce_train_teacher = ce_loss.forward(outputs_teacher, gt_batch_train[:].long())
            running_loss_ce_train_teacher += step_loss_ce_train_teacher

            # teacher dice loss
            step_loss_dice_train_teacher = dice_loss.forward(outputs_teacher, gt_batch_train, softmax=True)
            running_loss_dice_train_teacher += step_loss_dice_train_teacher
            
            # student cross-entropy loss
            step_loss_ce_train_student = ce_loss.forward(outputs_student, gt_batch_train[:].long())
            running_loss_ce_train_student += step_loss_ce_train_student

            # student dice loss
            step_loss_dice_train_student = dice_loss.forward(outputs_student, gt_batch_train, softmax=True)
            running_loss_dice_train_student += step_loss_dice_train_student

            # teacher distillation loss
            step_loss_distill_train_teacher = kl_loss.forward(
                input=log_softmax(outputs_teacher / args.T, dim=-1),
                target=softmax(outputs_student / args.T, dim=-1).detach()
            )
            step_loss_distill_train_teacher *= (args.T**2)
            step_loss_distill_train_teacher /= args.soft_target_loss_max_value
            
            running_loss_distill_train_teacher += step_loss_distill_train_teacher
            
            # student distillation loss
            step_loss_distill_train_student = kl_loss.forward(
                input=log_softmax(outputs_student / args.T, dim=-1),
                target=softmax(outputs_teacher / args.T, dim=-1).detach()
            )
            step_loss_distill_train_student *= (args.T**2)
            step_loss_distill_train_student /= args.soft_target_loss_max_value
            
            running_loss_distill_train_student += step_loss_distill_train_student

            # full loss teacher
            loss_train_teacher = (
                args.alpha * step_loss_ce_train_teacher + \
                args.beta * step_loss_dice_train_teacher +\
                args.phi * step_loss_distill_train_teacher
            )
            running_loss_train_teacher += loss_train_teacher
            
            # full loss student
            loss_train_student = (
                args.alpha * step_loss_ce_train_student + \
                args.beta * step_loss_dice_train_student +\
                args.phi * step_loss_distill_train_student
            )
            running_loss_train_student += loss_train_student

            # GD step teacher
            loss_train_teacher.backward()
            optimizer_teacher.step()
            
            # GD step teacher
            loss_train_student.backward()
            optimizer_student.step()

            if args.use_lr_scheduler:
                lr_scheduler_teacher.step()
                lr_scheduler_student.step()

            iter_num = iter_num + 1

            prog_bar.advance(prog_bar_train_batches_task, 1)
            prog_bar.advance(prog_bar_epochs_task, 1 / (num_batches_train))

        epoch_loss_ce_train_teacher = running_loss_ce_train_teacher / num_batches_train
        epoch_loss_dice_train_teacher = running_loss_dice_train_teacher / num_batches_train
        epoch_loss_distill_train_teacher = running_loss_distill_train_teacher / num_batches_train
        epoch_loss_train_teacher = running_loss_train_teacher / num_batches_train
        
        epoch_loss_ce_train_student = running_loss_ce_train_student / num_batches_train
        epoch_loss_dice_train_student = running_loss_dice_train_student / num_batches_train
        epoch_loss_distill_train_student = running_loss_distill_train_student / num_batches_train
        epoch_loss_train_student = running_loss_train_student / num_batches_train

        wb_run.log(
            {
                "epoch": epoch,
                
                "teacher/loss/full/train": epoch_loss_train_teacher,
                "teacher/loss/ce/train": epoch_loss_ce_train_teacher,
                "teacher/loss/softlogits/train": epoch_loss_distill_train_teacher,
                "teacher/loss/dice/train": epoch_loss_dice_train_teacher,
                "teacher/hyperparameters/lr": optimizer_teacher.param_groups[0]['lr'],
                
                "student/loss/full/train": epoch_loss_train_student,
                "student/loss/ce/train": epoch_loss_ce_train_student,
                "student/loss/softlogits/train": epoch_loss_distill_train_student,
                "student/loss/dice/train": epoch_loss_dice_train_student,
                "student/hyperparameters/lr": optimizer_student.param_groups[0]['lr']
            }
        )

        if epoch_loss_ce_train_teacher < best_epoch_loss_ce_train_teacher:
            best_epoch_loss_ce_train_teacher = epoch_loss_ce_train_teacher
            train_ce_loss_is_best_teacher = True
        if epoch_loss_dice_train_teacher < best_epoch_loss_dice_train_teacher:
            best_epoch_loss_dice_train_teacher = epoch_loss_dice_train_teacher
            train_dice_loss_is_best_teacher = True
        if epoch_loss_distill_train_teacher < best_epoch_loss_distill_train_teacher:
            best_epoch_loss_distill_train_teacher = epoch_loss_distill_train_teacher
            train_distill_loss_is_best_teacher = True
        if epoch_loss_train_teacher < best_epoch_loss_train_teacher:
            best_epoch_loss_train_teacher = epoch_loss_train_teacher
            train_loss_is_best_teacher = True

        if train_ce_loss_is_best_teacher:
            save_ckpt(model_student, optimizer_teacher, epoch, args.get_args(), f"{args.checkpoint_dir}/teacher_ckpt_train_best_ce_loss.pth")
        if train_dice_loss_is_best_teacher:
            save_ckpt(model_student, optimizer_teacher, epoch, args.get_args(), f"{args.checkpoint_dir}/teacher_ckpt_train_best_dice_loss.pth")
        if train_loss_is_best_teacher:
            save_ckpt(model_student, optimizer_teacher, epoch, args.get_args(), f"{args.checkpoint_dir}/teacher_ckpt_train_best_loss.pth")
        if epoch % args.log_every_n_epochs == 0:
            save_ckpt(model_student, optimizer_teacher, epoch, args.get_args(), f"{args.checkpoint_dir}/teacher_ckpt_epoch_{epoch}.pth")
        
        if epoch_loss_ce_train_student < best_epoch_loss_ce_train_student:
            best_epoch_loss_ce_train_student = epoch_loss_ce_train_student
            train_ce_loss_is_best_student = True
        if epoch_loss_dice_train_student < best_epoch_loss_dice_train_student:
            best_epoch_loss_dice_train_student = epoch_loss_dice_train_student
            train_dice_loss_is_best_student = True
        if epoch_loss_distill_train_student < best_epoch_loss_distill_train_student:
            best_epoch_loss_distill_train_student = epoch_loss_distill_train_student
            train_distill_loss_is_best_student = True
        if epoch_loss_train_student < best_epoch_loss_train_student:
            best_epoch_loss_train_student = epoch_loss_train_student
            train_loss_is_best_student = True

        if train_ce_loss_is_best_student:
            save_ckpt(model_student, optimizer_student, epoch, args.get_args(), f"{args.checkpoint_dir}/student_ckpt_train_best_ce_loss.pth")
        if train_dice_loss_is_best_student:
            save_ckpt(model_student, optimizer_student, epoch, args.get_args(), f"{args.checkpoint_dir}/student_ckpt_train_best_dice_loss.pth")
        if train_distill_loss_is_best_student:
            save_ckpt(model_student, optimizer_student, epoch, args.get_args(), f"{args.checkpoint_dir}/student_ckpt_train_best_distill_loss.pth")
        if train_loss_is_best_student:
            save_ckpt(model_student, optimizer_student, epoch, args.get_args(), f"{args.checkpoint_dir}/student_ckpt_train_best_loss.pth")
        if epoch % args.log_every_n_epochs == 0:
            save_ckpt(model_student, optimizer_student, epoch, args.get_args(), f"{args.checkpoint_dir}/student_ckpt_epoch_{epoch}.pth")
        
        ### --- train step --- ###

        ########################################################################

        ### --- validation step --- ###

        # epoch_metric_dice_val, epoch_metric_jaccard_val = _validate(
        #     "val", epoch, args, device, student, dl_val, num_batches_val,
        #     prog_bar, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task,
        #     wb_run
        # )

        # if epoch_metric_dice_val > best_epoch_metric_dice_val:
        #     best_epoch_metric_dice_val = epoch_metric_dice_val
        #     val_dice_is_best = True
        # if epoch_metric_jaccard_val > best_epoch_metric_jaccard_val:
        #     best_epoch_metric_jaccard_val = epoch_metric_jaccard_val
        #     val_jaccard_is_best = True

        # if val_dice_is_best:
        #     best_val_ckpt_path = f"{args.checkpoint_dir}/ckpt_val_best_dice_loss.pth"
        #     save_ckpt(student, optimizer, epoch, args.get_args(), best_val_ckpt_path)
        # if val_jaccard_is_best:
        #     save_ckpt(student, optimizer, epoch, args.get_args(), f"{args.checkpoint_dir}/ckpt_val_best_jaccard_loss.pth")

        ### --- validation step --- ###

        ########################################################################

        print_end_of_epoch_summary_mutual(
            # general
            args, epoch,
            
            # train teacher
            epoch_loss_ce_train_teacher, train_ce_loss_is_best_teacher,
            epoch_loss_dice_train_teacher, train_dice_loss_is_best_teacher,
            epoch_loss_distill_train_teacher, train_distill_loss_is_best_teacher,
            
            # train student
            epoch_loss_ce_train_student, train_ce_loss_is_best_student,
            epoch_loss_dice_train_student, train_dice_loss_is_best_student,
            epoch_loss_distill_train_student, train_distill_loss_is_best_student,
            
            # val
            # epoch_metric_jaccard_val, val_jaccard_is_best,
            # epoch_metric_dice_val, val_dice_is_best
        )

        prog_bar.update(task_id=prog_bar_epochs_task, total=args.num_epochs)

    # return best_val_ckpt_path
    return ""

### --- training --- ###

################################################################################

### --- validation --- ###

def _add_val_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_val: int):
    prog_bar_val_batches_task = prog_bar.add_task(description=args.val_batches_task_descr, total=num_batches_val)
    prog_bar_val_slices_task = prog_bar.add_task(description=args.val_slices_task_descr, total=69)
    prog_bar_val_metrics_task = prog_bar.add_task(description=args.val_metrics_task_descr, total=args.num_classes)

    return prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task

def _validate(
    inference_type: str, epoch: int, args: args, device, student: Module, dl_val: DataLoader, num_batches_val: int,
    prog_bar: Progress, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task,
    wb_run: Run
):
    if inference_type not in args.INFERENCE_TYPES:
        raise ValueError(f"{inference_type} is an invalid inference type. Supported values: {args.INFERENCE_TYPES}")

    # one list element --> one segmentation class
    # NOTE about indexing!
    # Paper excludes black class from segmentation metrics, so we index
    # running_metric_{dice,jaccard}_val accordingly!
    running_metric_dice_val    = [0] * args.num_classes_for_metrics
    running_metric_jaccard_val = [0] * args.num_classes_for_metrics
    
    student.eval()

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
            
            if x != args.squeeze_unet_img_size or y != args.squeeze_unet_img_size:
                slice = zoom(slice, (args.squeeze_unet_img_size / x, args.squeeze_unet_img_size / y), order=3)

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                outputs = student(input)

                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                if x != args.squeeze_unet_img_size or y != args.squeeze_unet_img_size:
                    pred = zoom(out, (x / args.squeeze_unet_img_size, y / args.squeeze_unet_img_size), order=0)
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
            f"loss/dice/{inference_type}": 1 - epoch_metric_dice_val,
            f"metric/dice/{inference_type}": epoch_metric_dice_val,
            f"metric/jaccard/{inference_type}": epoch_metric_jaccard_val
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
    args: args, prog_bar: Progress, device, student: Module, dl_test: DataLoader,
    wb_run: Run
):
    
    num_batches_test = int(len(dl_test) * args.lim_num_batches_percent_test)
    if num_batches_test == 0:
        num_batches_test = 1

    prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task = _add_test_prog_bar_tasks(args, prog_bar, num_batches_test)
    
    return _validate(
        "test", args.num_epochs, args, device, student, dl_test, num_batches_test, 
        prog_bar, prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task,
        wb_run
    )

def _test(args: args, prog_bar: Progress, device: device, model: Module, dl_test: DataLoader, wb_run: Run):
    
    epoch_metric_dice_test, epoch_metric_jaccard_test = _perform_testing(args, prog_bar, device, model, dl_test, wb_run)
    
    print_end_of_test_summary(
        args, 
        epoch_metric_jaccard_test, True,
        epoch_metric_dice_test, True
    )


### --- test --- ###

################################################################################

### --- main --- ###

def main():

    seed = round(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)

    # args
    print(f"Arguments:")
    pprint(args.get_args())
    print()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_name(device)

    # models
    model_teacher = _init_TransUNet_model(args, device)
    model_student = _init_SqueezeUNet_model(args, device)
    
    # optimizer
    optimizer_teacher = _init_optimizer(args, model_teacher)
    optimizer_student = _init_optimizer(args, model_student)

    # learning rate scheduler
    lr_scheduler_teacher = _init_lr_scheduler(args, optimizer_teacher)
    lr_scheduler_student = _init_lr_scheduler(args, optimizer_student)

    # data
    ds_train = _get_dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train", transform=args.train_transforms)
    dl_train = _get_dataloader(ds_train, args.batch_size, True, args.num_workers, pin_memory=args.pin_memory)
    # TODO add assert-based sanity check to make sure that sampling is actually the same!

    ds_val = _get_dataset(base_dir=args.val_volume_path, list_dir=args.list_dir, split="val_vol", transform=None)
    dl_val = _get_dataloader(ds_val, 1, False, 1, True)
    
    ds_test = _get_dataset(base_dir=args.test_volume_path, list_dir=args.list_dir, split="test_vol", transform=None)
    dl_test = _get_dataloader(ds_test, 1, False, 1, True)
    
    print_data_summary(args, ds_train, dl_train, ds_val, dl_val, ds_test, dl_test)

    # Weights and Biases
    wb_run = _wandb_init(
        args, 
        model_teacher, model_student, optimizer_teacher, optimizer_student,  
        lr_scheduler_teacher, lr_scheduler_student
    )

    # progress
    prog_bar = get_progress_bar()
    prog_bar.start()

    # training (and validation!)
    best_val_ckpt_path = train(
        args, 1, prog_bar, device, 
        model_teacher, model_student, 
        optimizer_teacher, optimizer_student, 
        lr_scheduler_teacher, lr_scheduler_student,
        dl_train, dl_val, wb_run
    )
    
    print(f"TODO implement test with support for teacher and student both!")
    exit()

    # loading best val checkpoint to perform test on it!
    model = load_ckpt(student, best_val_ckpt_path)
    _test(args, prog_bar, device, model, dl_test, wb_run)


if __name__ == "__main__":
    main()

### --- main --- ###

################################################################################