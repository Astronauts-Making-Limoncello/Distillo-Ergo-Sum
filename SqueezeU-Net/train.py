import os
import numpy as np
from DiceLoss import DiceLoss
import torch
from rich.progress import Progress
from rich import print
from rich.console import Console
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from squeezeunet_torch import SqueezeUNet
import utils
import SqueezeUnet_args as args
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

console = Console()
console.print("Torch version:", torch.__version__)
console.print("Is CUDA enabled?", torch.cuda.is_available())

base_lr = 1e-5
num_classes = args.num_classes
batch_size = args.batch_size
max_epochs = args.num_epochs


prog_bar = utils.get_progress_bar()
prog_bar.start()

def get_trainloader():
    return utils.get_dataset_train(batch_size=batch_size, num_workers=1, pin_memory=True)

def get_validationloader():
    return utils.get_dataset_validation(batch_size=1, num_workers=1, pin_memory=True)

def get_testloader():
    return utils.get_dataset_test(batch_size=1, num_workers=1, pin_memory=True)

def init_model():
    model = SqueezeUNet(num_classes=num_classes)
    model.to('cuda')
    console.print(f"Total trainable parameters: [bold green]{sum(p.numel() for p in model.parameters() if p.requires_grad)}[/bold green]")
    return model

def _add_train_prog_bar_tasks(args, prog_bar: Progress, num_batches_train: int):

    prog_bar_epochs_task = prog_bar.add_task(description=args.epochs_task_descr, total=max_epochs)
    prog_bar_train_batches_task = prog_bar.add_task(description=args.train_batches_task_descr, total=num_batches_train)

    return prog_bar_epochs_task, prog_bar_train_batches_task

def _add_val_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_val: int):
    prog_bar_val_batches_task = prog_bar.add_task(description=args.val_batches_task_descr, total=num_batches_val)
    prog_bar_val_slices_task = prog_bar.add_task(description=args.val_slices_task_descr, total=69)
    prog_bar_val_metrics_task = prog_bar.add_task(description=args.val_metrics_task_descr, total=args.num_classes)

    return prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task

def train(args):
    iter_num = 0
    os.makedirs(args.checkpoint_dir) if not os.path.exists(args.checkpoint_dir) else None

    dataloader_train = get_trainloader()

    dataloader_validation = get_validationloader()

    max_iterations = args.num_epochs * len(dataloader_train)
    
    model = init_model()

    num_batches_train = int(len(dataloader_train) * args.lim_num_batches_percent_train)
    if num_batches_train == 0:
        num_batches_train = 1

    num_batches_val = int(len(dataloader_validation) * args.lim_num_batches_percent_val)
    if num_batches_val == 0:
        num_batches_val = 1
    
    prog_bar_epochs_task, prog_bar_train_batches_task = _add_train_prog_bar_tasks(args, prog_bar, num_batches_train)
    prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task = _add_val_prog_bar_tasks(args, prog_bar, num_batches_val)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    best_epoch_loss_ce_train = torch.inf
    best_epoch_loss_dice_train = torch.inf
    best_epoch_loss_train = torch.inf
    best_epoch_metric_dice_val = 0
    best_epoch_metric_jaccard_val = 0
    
    train_ce_loss_is_best = False
    
    val_dice_is_best = False
    val_jaccard_is_best = False
    
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch_num in range(max_epochs):
        
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

        model.train()
    
        for sampled_batch in list(dataloader_train)[: num_batches_train]:

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            
            outputs = model(image_batch)
            step_loss_ce_train = ce_loss.forward(outputs, label_batch[:].long())
            running_loss_ce_train += step_loss_ce_train
            step_loss_dice_train = dice_loss.forward(outputs, label_batch, softmax=True)
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
            utils.save_ckpt(model, optimizer, epoch_num, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_ce_loss.pth")
        if train_dice_loss_is_best:
            utils.save_ckpt(model, optimizer, epoch_num, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_dice_loss.pth")
        if train_loss_is_best:
            utils.save_ckpt(model, optimizer, epoch_num, args.get_args(), f"{args.checkpoint_dir}/ckpt_train_best_loss.pth")
        if epoch_num % args.log_every_n_epochs:
            utils.save_ckpt(model, optimizer, epoch_num, args.get_args(), f"{args.checkpoint_dir}/ckpt_epoch_{epoch_num}.pth")


        epoch_metric_dice_val, epoch_metric_jaccard_val = validate(
            args, "cuda", model, dataloader_validation, num_batches_val,
            prog_bar, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task
        )

        val_dice_is_best    = utils.metric_has_improved(epoch_metric_dice_val   , best_epoch_metric_dice_val, "max")
        val_jaccard_is_best = utils.metric_has_improved(epoch_metric_jaccard_val, best_epoch_metric_jaccard_val, "max")

        print(
            f"[b][{args.epochs_color}]{epoch_num:03d}[/{args.epochs_color}][/b] | train | "
            f"Cross-Entropy loss [b][{args.train_batches_color}]{epoch_loss_ce_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_ce_loss_is_best else args.loss_is_not_best_str} | "
            f"Dice loss   [b][{args.train_batches_color}]{epoch_loss_dice_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_dice_loss_is_best else args.loss_is_not_best_str} |"
            f"\n"
            f" | val | "
            f"Jaccard metric     [b][{args.val_batches_color}]{epoch_metric_jaccard_val:02.6f}[/{args.val_batches_color}][/b] {args.metric_is_best_str if val_jaccard_is_best else args.metric_is_not_best_str} | "
            f"Dice metric [b][{args.val_batches_color}]{epoch_metric_dice_val:02.6f}[/{args.val_batches_color}][/b] {args.metric_is_best_str if val_dice_is_best else args.metric_is_not_best_str} |"
        )

        if epoch_num + 1 != max_epochs:
            print()
            print("Epoch", epoch_num + 1, "of", max_epochs, "complete.")
            print()

        prog_bar.update(task_id=prog_bar_epochs_task, total=max_epochs) 

    console.print("[bold green]Training Finished![/bold green]")
    return model

def validate( args: args, device, model: torch.nn.Module, dl_val: DataLoader, num_batches_val: int,
    prog_bar: Progress, prog_bar_val_batches_task, prog_bar_val_slices_task, prog_bar_val_metrics_task):
    
    running_metric_dice_val    = [0] * args.num_classes_for_metrics
    running_metric_jaccard_val = [0] * args.num_classes_for_metrics
    
    model.eval()

    for batch_val in list(dl_val)[: num_batches_val]:
        
        img_batch_val = batch_val["image"] 
        label_batch_val = batch_val["label"]
        
        image = img_batch_val.cpu().detach().numpy()
        label = label_batch_val.cpu().detach().numpy()
        prediction = np.zeros_like(label)

        num_slices = image.shape[0]

        prog_bar.reset(task_id=prog_bar_val_slices_task, total=num_slices)
        prog_bar.reset(task_id=prog_bar_val_metrics_task, total=args.num_classes_for_metrics)

        for ind in range(num_slices):
            slice = image[ind, :, :]

            x, y = slice.shape[0], slice.shape[1]
            
            if x != args.img_size or y != args.img_size:
                slice = zoom(slice, (args.img_size / x, args.img_size / y), order=3)

            input = torch.from_numpy(slice).float().to(device)

            with torch.no_grad():
                outputs = model(input.unsqueeze(0))

                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                if x != args.img_size or y != args.img_size:
                    pred = zoom(out, (x / args.img_size, y / args.img_size), order=0)
                else:
                    pred = out

                prediction[ind] = pred

            prog_bar.advance(task_id=prog_bar_val_slices_task, advance=1)
        
        for c in range(1, args.num_classes):
            
            running_metric_dice_val[c - 1]    += utils.calculate_dice_metric_per_case(prediction == c, label == c)
            running_metric_jaccard_val[c - 1] += utils.calculate_jaccard_metric_per_case(prediction == c, label == c)

            prog_bar.advance(task_id=prog_bar_val_metrics_task, advance=1)
        prog_bar.update(task_id=prog_bar_val_metrics_task, total=args.num_classes_for_metrics)

        prog_bar.update(task_id=prog_bar_val_batches_task, advance=1)

    epoch_metric_dice_val = np.array(running_metric_dice_val) / num_batches_val
    epoch_metric_jaccard_val = np.array(running_metric_jaccard_val) / num_batches_val

    # averaging across segmentation classes
    epoch_metric_dice_val = np.mean(epoch_metric_dice_val, axis=0)
    epoch_metric_jaccard_val = np.mean(epoch_metric_jaccard_val, axis=0)

    return epoch_metric_dice_val, epoch_metric_jaccard_val


def _add_test_prog_bar_tasks(args: args, prog_bar: Progress, num_batches_test: int):
    prog_bar_test_batches_task = prog_bar.add_task(description=args.test_batches_task_descr, total=num_batches_test)
    prog_bar_test_slices_task = prog_bar.add_task(description=args.test_slices_task_descr, total=69)
    prog_bar_test_metrics_task = prog_bar.add_task(description=args.test_metrics_task_descr, total=args.num_classes)

    return prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task

def test(args: args, prog_bar: Progress, device: torch.cuda.device, model: torch.nn.Module, dl_test: DataLoader):
    best_epoch_metric_jaccard_test = 0
    best_epoch_metric_dice_test = 0

    num_batches_test = int(len(dl_test) * args.lim_num_batches_percent_test)
    if num_batches_test == 0:
        num_batches_test = 1

    prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task = _add_test_prog_bar_tasks(args, prog_bar, num_batches_test)
    
    epoch_metric_dice_test, epoch_metric_jaccard_test = validate(
        args, device, model, dl_test, num_batches_test, 
        prog_bar, prog_bar_test_batches_task, prog_bar_test_slices_task, prog_bar_test_metrics_task
    )

    test_dice_is_best    = utils.metric_has_improved(epoch_metric_dice_test   , best_epoch_metric_dice_test, "max")
    test_jaccard_is_best = utils.metric_has_improved(epoch_metric_jaccard_test, best_epoch_metric_jaccard_test, "max")
    
    print(
        f"    | test  | "
        f"Jaccard metric     [b][{args.test_batches_color}]{epoch_metric_jaccard_test:02.6f}[/{args.test_batches_color}][/b] {args.metric_is_best_str if test_jaccard_is_best else args.metric_is_not_best_str} | "
        f"Dice metric [b][{args.test_batches_color}]{epoch_metric_dice_test:02.6f}[/{args.test_batches_color}][/b] {args.metric_is_best_str if test_dice_is_best else args.metric_is_not_best_str} |"
    )

    print()

if __name__ == "__main__":
    model = train(args)
    dl_test = get_testloader()
    test(args, prog_bar, "cuda", model, dl_test)

   
