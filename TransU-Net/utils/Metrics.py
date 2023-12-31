from rich import print

from medpy import metric

def calculate_dice_metric_per_case(pred, gt):

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum()>0:
        
        return metric.binary.dc(pred, gt)
    
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    
    else:
        return 0
    

def calculate_hausdorff_metric_per_case(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum()>0:

        return metric.binary.hd95(pred, gt)
    
    elif pred.sum() > 0 and gt.sum()==0:
        return 0
    
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


# original code. Keep it this way to make sure new codes are correct
def calculate_dice_hausdorff_jaccard_metrics_per_case(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)

        hd95 = metric.binary.hd95(pred, gt)

        jaccard = metric.binary.jc(pred, gt)
        
        return dice, hd95, jaccard
    
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1
    
    else:
        return 0, 0, 0
    
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