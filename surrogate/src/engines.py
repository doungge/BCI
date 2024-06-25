import torch

from src.utils import AverageMeter

def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, epoch, device):
    model.train()
    loss_mean = AverageMeter()
    metric_mean = AverageMeter()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs, epoch)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric)

        scheduler.step()

    summary = {'loss': loss_mean.avg, 'metric': metric_mean.avg}
    

    return summary


def evaluate(loader, model, loss_fn, metric_fn, epoch, device):
    model.eval()
    loss_mean = AverageMeter()
    metric_mean = AverageMeter()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs, epoch)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric)
    
    summary = {'loss': loss_mean.avg, 'metric': metric_mean.avg}

    return summary