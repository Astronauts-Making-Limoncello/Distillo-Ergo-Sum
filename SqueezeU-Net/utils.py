import torch
from dataset import RandomGenerator, Synapse_dataset
#from torchvision import transforms
from torch.utils.data import DataLoader
from rich.progress import *
from rich import print
from medpy import metric

def get_progress_bar() -> Progress:

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.2f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[#00008B]{task.speed} it/s"),
        SpinnerColumn()
    )

def get_dataset_train(batch_size: int, num_workers: int, pin_memory: bool) -> Synapse_dataset:
    root_path = r"C:\Users\giaco\Documents\GIACOMO\SAPIENZA\AML\Project\project_TransUNet\data\Synapse\train_npz"
    list_dir = r'C:\Users\giaco\Documents\GIACOMO\SAPIENZA\AML\Project\project_TransUNet\TransUNet\lists\lists_Synapse'

    train_dataset = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split='train',
        # transform=transforms.Compose(
        #     [
        #         RandomGenerator(output_size=[512, 512])
        #     ]
        # )
    )
    return DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )

def get_dataset_test(batch_size: int, num_workers: int, pin_memory: bool) -> Synapse_dataset:
    root_path = r"C:\Users\giaco\Documents\GIACOMO\SAPIENZA\AML\Project\project_TransUNet\data\Synapse\test_vol_h5"
    list_dir = r'C:\Users\giaco\Documents\GIACOMO\SAPIENZA\AML\Project\project_TransUNet\TransUNet\lists\lists_Synapse'

    test_dataset = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split='test_vol',
        # transform=transforms.Compose(
        #     [
        #         RandomGenerator(output_size=[512, 512])
        #     ]
        # )
    )

    return DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )


def get_dataset_validation(batch_size: int, num_workers: int, pin_memory: bool) -> Synapse_dataset:
    root_path = r"C:\Users\giaco\Documents\GIACOMO\SAPIENZA\AML\Project\project_TransUNet\data\Synapse\val_vol_h5"
    list_dir = r'C:\Users\giaco\Documents\GIACOMO\SAPIENZA\AML\Project\project_TransUNet\TransUNet\lists\lists_Synapse'

    val_dataset = Synapse_dataset(base_dir=root_path, list_dir=list_dir, split='val_vol',
        # transform=transforms.Compose(
        #     [
        #         RandomGenerator(output_size=[512, 512])
        #     ]
        # )
    )

    return DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )

def metric_has_improved(metric, metric_best, lt_or_gt: str):
    is_best = False

    if lt_or_gt == "min":
        if metric < metric_best:
            metric_best = metric
            is_best = True
    elif lt_or_gt == "max":
        if metric > metric_best:
            metric_best = metric
            is_best = True

    else:
        raise ValueError(f"lt_or_gt should be either \"min\" or \"max\", got {lt_or_gt} instead")

    return is_best

def calculate_dice_metric_per_case(pred, gt):

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum()>0:
        
        return metric.binary.dc(pred, gt)
    
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    
    else:
        return 0
    
def calculate_jaccard_metric_per_case(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum()>0:

        jaccard = metric.binary.jc(pred, gt)
        
        return jaccard
    
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    
    else:
        return 0
    

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