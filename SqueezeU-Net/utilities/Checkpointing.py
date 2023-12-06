import torch

def save_ckpt(
    model: torch.nn.Module, optimizer: torch.optim, epoch: int, args: dict, ckpt_file_full_path: str
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": args
        },
        ckpt_file_full_path
    )