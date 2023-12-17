import torch

from rich import print

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

def load_ckpt(model: torch.nn.Module, ckpt_file_path: str) -> torch.nn.Module:
    
    ckpt = torch.load(ckpt_file_path)

    model.load_state_dict(state_dict=ckpt["model_state_dict"])

    return model

def handle_resume_from_ckpt(args, model: torch.nn.Module, optimizer: torch.optim.Optimizer):

    starting_epoch = 1
    
    if args.resume_from_ckpt:

        ckpt_to_resume_from = torch.load(args.checkpoint_dir_to_resume_from)
        
        if args.ckpt_to_resume_from_has_latent_space:

            model.load_state_dict(ckpt_to_resume_from["model_state_dict"])
            optimizer.load_state_dict(ckpt_to_resume_from["optimizer_state_dict"])
            
            if not args.restart_epoch:
                starting_epoch = ckpt_to_resume_from["epoch"] + 1


        else:

            transformer_state_dict = {}

            for layer_name in model.transformer.state_dict().keys():
                transformer_state_dict[layer_name] = model.transformer.state_dict()[layer_name]
            
            model.transformer.load_state_dict(transformer_state_dict)

            decoder_state_dict = {}

            for layer_name in model.decoder.state_dict().keys():
                decoder_state_dict[layer_name] = model.decoder.state_dict()[layer_name]
            
            model.decoder.load_state_dict(decoder_state_dict)
            
            segmentation_head_state_dict = {}

            for layer_name in model.segmentation_head.state_dict().keys():
                segmentation_head_state_dict[layer_name] = model.segmentation_head.state_dict()[layer_name]
            
            model.segmentation_head.load_state_dict(segmentation_head_state_dict)
        
        print(f"Resuming training from epoch {ckpt_to_resume_from['epoch']} of checkpoint [b][{args.train_batches_color}]{args.run_id_to_resume_from}[/{args.train_batches_color}][/b]\n")

    return starting_epoch, model, optimizer