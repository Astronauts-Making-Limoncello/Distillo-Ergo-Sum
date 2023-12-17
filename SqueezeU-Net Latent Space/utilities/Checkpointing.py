import torch

from rich import print

def save_ckpt(
    model: torch.nn.Module, optimizer: torch.optim, lr_scheduler: torch.optim.lr_scheduler.MultiStepLR | None, epoch: int, args: dict, ckpt_file_full_path: str
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else torch.zeros((1, 1)),
            "epoch": epoch,
            "args": args
        },
        ckpt_file_full_path
    )

def load_ckpt(model: torch.nn.Module, ckpt_file_path: str) -> torch.nn.Module:
    
    ckpt = torch.load(ckpt_file_path)

    model.load_state_dict(state_dict=ckpt["model_state_dict"])

    return model

def handle_resume_from_ckpt(args, model: torch.nn.Module, optimizer: torch.optim, lr_scheduler: torch.optim.lr_scheduler.MultiStepLR | None):

    starting_epoch = 1
    
    if args.resume_from_ckpt:

        ckpt_to_resume_from = torch.load(args.checkpoint_dir_to_resume_from)

        model.load_state_dict(ckpt_to_resume_from["model_state_dict"])
        optimizer.load_state_dict(ckpt_to_resume_from["optimizer_state_dict"])
        
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt_to_resume_from["lr_scheduler_state_dict"])

        starting_epoch = ckpt_to_resume_from["epoch"] + 1

        print(f"Resuming training from epoch {ckpt_to_resume_from['epoch']} of checkpoint [b][{args.train_batches_color}]{args.run_id_to_resume_from}[/{args.train_batches_color}][/b]\n")

    return starting_epoch, model, optimizer, lr_scheduler