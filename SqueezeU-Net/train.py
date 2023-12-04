from DiceLoss import DiceLoss
import torch
from tqdm import tqdm
from rich.progress import Progress
from rich.console import Console
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from squeezeunet_torch import SqueezeUNet
import utils

console = Console()
console.print("Torch version:", torch.__version__)
console.print("Is CUDA enabled?", torch.cuda.is_available())

base_lr = 1e-5
num_classes = 8
batch_size = 8
max_iterations = 3000
max_epochs = 30
iter_num = 0

prog_bar = utils.get_progress_bar()
prog_bar.start()

def get_trainloader():
    return utils.get_dataset_train(batch_size=batch_size, num_workers=4, pin_memory=True)

def init_model():
    model = SqueezeUNet(num_classes=num_classes)
    model.to('cuda')
    console.print(f"Total trainable parameters: [bold green]{sum(p.numel() for p in model.parameters() if p.requires_grad)}[/bold green]")
    return model

def _add_train_prog_bar_tasks(args, prog_bar: Progress, num_batches_train: int):

    prog_bar_epochs_task = prog_bar.add_task(description="epochs", total=max_epochs)
    prog_bar_train_batches_task = prog_bar.add_task(description="train_batches", total=num_batches_train)

    return prog_bar_epochs_task, prog_bar_train_batches_task

def train(args):
    dataloader_train = get_trainloader()
    model = init_model()

    num_batches_train = int(len(dataloader_train) * 0.20)
    if num_batches_train == 0:
        num_batches_train = 1
    
    prog_bar_epochs_task, prog_bar_train_batches_task = _add_train_prog_bar_tasks(args, prog_bar, num_batches_train)
    

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    best_epoch_loss_ce_train = torch.inf
    best_epoch_loss_dice_train = torch.inf
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    for epoch_num in range(max_epochs):
        prog_bar.reset(prog_bar_train_batches_task)
        
        running_loss_ce_train = 0
        running_loss_dice_train = 0
        running_loss_train = 0
        model.train()
    
        for _, sampled_batch in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch_num}/{max_epochs}", leave=False)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            
            outputs = model(image_batch)
            step_loss_ce_train = ce_loss.forward(outputs, label_batch[:].long())
            running_loss_ce_train += step_loss_ce_train
            step_loss_dice_train = dice_loss.forward(outputs, label_batch, softmax=True)
            running_loss_dice_train += step_loss_dice_train
            loss_train = 0.5 * step_loss_ce_train + 0.5 * step_loss_dice_train
            running_loss_train += loss_train
        
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            prog_bar.advance(prog_bar_train_batches_task, 1)
            prog_bar.advance(prog_bar_epochs_task, 1 / (num_batches_train))
        
        epoch_loss_ce_train = running_loss_ce_train / num_batches_train
        epoch_loss_dice_train = running_loss_dice_train / num_batches_train

        train_ce_is_best   = utils.metric_has_improved(epoch_loss_ce_train  , best_epoch_loss_ce_train, "min")
        train_dice_is_best = utils.metric_has_improved(epoch_loss_dice_train, best_epoch_loss_dice_train, "min")

        print(
            f"[b][{args.epochs_color}]{epoch_num:03d}[/{args.epochs_color}][/b] | train | "
            f"Cross-Entropy loss [b][{args.train_batches_color}]{epoch_loss_ce_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_ce_is_best else args.loss_is_not_best_str} | "
            f"Dice loss   [b][{args.train_batches_color}]{epoch_loss_dice_train.item():02.6f}[/{args.train_batches_color}][/b] {args.loss_is_best_str if train_dice_is_best else args.loss_is_not_best_str} |"
            f"\n"
        )

        if epoch_num + 1 != max_epochs:
            print()
            print("Epoch", epoch_num + 1, "of", max_epochs, "complete.")
            print()

        prog_bar.update(task_id=prog_bar_epochs_task, total=max_epochs) 

    console.print("[bold green]Training Finished![/bold green]")

if __name__ == "__main__":
    train(None)
   

   
