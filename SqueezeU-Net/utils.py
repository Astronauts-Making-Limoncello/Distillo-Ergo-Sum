from dataset import RandomGenerator, Synapse_dataset
#from torchvision import transforms
from torch.utils.data import DataLoader
from rich.progress import *

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