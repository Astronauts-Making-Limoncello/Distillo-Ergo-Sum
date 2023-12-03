import logging
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import dataset
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from squeezeunet_torch import SqueezeUNet


if __name__ == "__main__":
    print("Torch version:",torch.__version__)

    print("Is CUDA enabled?",torch.cuda.is_available())

    base_lr = 1e-5
    num_classes = 8
    batch_size = 8
    max_iterations = 3000
    max_epochs = 3
    iter_num = 0

    train_dataset = dataset.main()

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = SqueezeUNet(num_classes=num_classes)
    model.to('cuda')
    model.train()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #max_iterations = max_epochs * len(trainloader)
    max_iterations = 5
    print("max_iteration: ", max_iterations)
    for epoch_num in range(max_epochs):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            print("label_batch: ", label_batch.shape)
            with autocast():
                outputs = model(image_batch)
                print("outputs: ", outputs.shape)
                print(outputs)
                break
                loss = ce_loss(outputs, label_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            logging.info('iteration %d : loss_ce: %f : epoch: %f' % (iter_num, loss.item(), epoch_num))
        print('iteration %d : loss_ce: %f : epoch: %f' % (iter_num, loss.item(), epoch_num))

    print("Training Finished!")