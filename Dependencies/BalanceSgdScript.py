from Dependencies.UtilsScript import *
import torchvision
import math
import torch
from torch.utils.data import DataLoader
from typing import Callable
import torch.nn as nn
from Dependencies import UtilsScript as us
from Dependencies import TrainTestScript


def run_info(trainable_model, fixed_model, ds, epochs, bs, batches, optimizer, lr_cps):
    print('{} - {} using fixed {}:'.format(us.get_function_name(depth=2), trainable_model.title, fixed_model.title))
    print('\tData: {}'.format(us.data_set_size_to_str(ds)))
    print('\tsaving on each loss improvment')
    msg = '\tepochs={}, bs={}, #batches={}'
    print(msg.format(epochs, bs, batches))
    print('\toptimizer = {}'.format(us.opt_to_str(optimizer)))
    if len(lr_cps) > 0:
        print('\tif epoch in {}: lr/=10'.format(lr_cps))
    return


def balance_using_dl(dl: DataLoader, trainable_model: nn.Module, fixed_model: nn.Module, opt: torch.optim,
                     f: Callable, epochs: int, save_models: bool = False, b_interval: int = -1,
                     lr_cps: list = None) -> None:
    bal_time = get_current_time()

    bs = dl.batch_size
    samples = len(dl.dataset)
    total_batches = len(dl)
    lr_cps = [] if lr_cps is None else [math.ceil(epochs * cp) for cp in lr_cps]

    run_info(trainable_model, fixed_model, dl.dataset, epochs, bs, total_batches, opt, lr_cps)

    lowest_loss = float("inf")
    best_epoch = -1
    eps = 1e-8  # not to divide in zero

    print('Balancing:')
    for epoch in range(1, epochs + 1):
        trainable_model.train()
        fixed_model.eval()
        if b_interval > 0:
            print('Epoch {}/{}:'.format(epoch, epochs))
        if epoch in lr_cps:
            us.set_lr(opt, us.get_lr(opt) / 10)
            print('\toptimizer changed = {}'.format(us.opt_to_str(opt)))
        sum_batches_loss_p, sum_batches_loss_c, sum_batches_loss = 0.0, 0.0, 0.0
        for b_i, (batchX, batch_y) in enumerate(dl):
            batchX, batch_y = add_cuda(batchX), add_cuda(batch_y)

            batch_y_pred_vectors_p = fixed_model(batchX)
            loss_p = f(batch_y_pred_vectors_p, batch_y, reduction='mean')

            batch_y_pred_vectors_c = trainable_model(batchX)
            loss_c = f(batch_y_pred_vectors_c, batch_y, reduction='mean')

            # on rare cases, loss_p could be 0
            loss = torch.abs(1 - loss_c / loss_p) if loss_p > eps else torch.abs(1 - loss_c / (loss_p + eps))
            # ============ Backward ============
            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_batches_loss_p += loss_p.item()
            sum_batches_loss_c += loss_c.item()
            sum_batches_loss += loss.item()

            if b_interval > 0 and b_i % b_interval == 0:
                msg = '\tBatch {:3}/{:3} ({:5}/{:5}) - Loss: {:,.6f}, loss_c:{:,.6f}, loss_p:{:,.6f}'
                print(msg.format(b_i, total_batches, b_i * bs, samples, loss.item(), loss_c.item(), loss_p.item()))

        # for each batch we summed its mean. divide by total_batches
        average_p = sum_batches_loss_p / total_batches
        average_c = sum_batches_loss_c / total_batches
        average_loss = sum_batches_loss / total_batches

        msg = '\tEpoch [{}/{}] loss:{:,.6f}, loss_c:{:,.6f}, loss_p:{:,.6f}, time {}'
        if average_loss < lowest_loss or epoch == 1:
            lowest_loss = average_loss
            best_epoch = epoch
            msg += ' New best found'
            if save_models:
                save_model(trainable_model, ack_print=False)

        print(msg.format(epoch, epochs, average_loss, average_c, average_p, get_time_str(bal_time)))

    print('Done balancing: On epoch {}: loss={:.6f}'.format(best_epoch, lowest_loss))
    return


def balance_using_datasets(ds: torchvision.datasets, bs: int, trainable_model: nn.Module, fixed_model: nn.Module,
                           opt: torch.optim, f: Callable, epochs: int, save_models: bool = False, shuffle: bool = False,
                           b_interval: int = -1, lr_cps: list = None) -> None:
    bal_time = get_current_time()

    samples = len(ds)
    total_batches = math.ceil(samples / bs)
    lr_cps = [] if lr_cps is None else [math.ceil(epochs * cp) for cp in lr_cps]

    run_info(trainable_model, fixed_model, ds, epochs, bs, total_batches, opt, lr_cps)

    lowest_loss = float("inf")
    best_epoch = -1
    eps = 1e-8  # not to divide in zero

    print('Balancing:')
    for epoch in range(1, epochs + 1):
        trainable_model.train()
        fixed_model.eval()
        if shuffle:
            shuffle_ds(ds)
        if b_interval > 0:
            print('Epoch {}/{}:'.format(epoch, epochs))
        if epoch in lr_cps:
            us.set_lr(opt, us.get_lr(opt) / 10)
            print('\toptimizer changed = {}'.format(us.opt_to_str(opt)))
        sum_batches_loss_p, sum_batches_loss_c, sum_batches_loss = 0.0, 0.0, 0.0
        for i in range(0, samples, bs):
            batchX, batch_y = add_cuda(ds.data[i:i + bs]), add_cuda(ds.targets[i:i + bs])

            batch_y_pred_vectors_p = fixed_model(batchX)
            loss_p = f(batch_y_pred_vectors_p, batch_y, reduction='mean')

            batch_y_pred_vectors_c = trainable_model(batchX)
            loss_c = f(batch_y_pred_vectors_c, batch_y, reduction='mean')
            # on rare cases, loss_p could be 0
            loss = torch.abs(1 - loss_c / loss_p) if loss_p > eps else torch.abs(1 - loss_c / (loss_p + eps))
            # ============ Backward ============
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_batches_loss_p += loss_p.item()
            sum_batches_loss_c += loss_c.item()
            sum_batches_loss += loss.item()

            batch_ind = int(i / bs)
            if b_interval > 0 and batch_ind % b_interval == 0:
                msg = '\tBatch {:3}/{:3} ({:5}/{:5}) - Loss: {:,.6f}, loss_c:{:,.6f}, loss_p:{:,.6f}'
                print(msg.format(batch_ind, total_batches, batch_ind * bs, samples, loss.item(), loss_c.item(),
                                 loss_p.item()))
        # for each batch we summed its mean. divide by total_batches
        average_p = sum_batches_loss_p / total_batches
        average_c = sum_batches_loss_c / total_batches
        average_loss = sum_batches_loss / total_batches

        msg = '\tEpoch [{}/{}] loss:{:,.6f}, loss_c:{:,.6f}, loss_p:{:,.6f}, time {}'
        if average_loss < lowest_loss or epoch == 1:
            lowest_loss = average_loss
            best_epoch = epoch
            msg += ' New best found'
            if save_models:
                save_model(trainable_model, ack_print=False)

        print(msg.format(epoch, epochs, average_loss, average_c, average_p, get_time_str(bal_time)))

    print('Done balancing: On epoch {}: loss={:.6f}'.format(best_epoch, lowest_loss))
    return


def effective_epsilon_stats_dataloader(loader: DataLoader, p: nn.Module, c: nn.Module, f: Callable) -> None:
    print('Epsilon ({} vs {})'.format(p.title, c.title))
    ratio = 100. * (1 - c.params / p.params)
    print('p_model {} params, c_model {} params, compression ratio={:,.2f}%'.format(p.params, c.params, ratio))
    print('\tdl: {}'.format(data_set_size_to_str(loader.dataset)))
    loss_p, acc_p, _ = TrainTestScript.test_using_dl(dl=loader, model=p, f=f)
    loss_c, acc_c, _ = TrainTestScript.test_using_dl(dl=loader, model=c, f=f)

    eps = 1e-8  # not to divide in zero

    eps_loss = abs(1 - loss_c / loss_p) if loss_p > eps else abs(1 - loss_c / (loss_p + eps))
    print('\tloss_c:{:,.4f}, loss_p:{:,.4f}, eps_loss:{:,.4f}'.format(loss_c, loss_p, eps_loss))

    eps_acc = abs(1 - acc_c / acc_p) if acc_p > eps else abs(1 - acc_c / (acc_p + eps))
    print('\tacc_c:{:,.4f}, acc_p:{:,.4f}, eps_acc:{:,.4f}'.format(acc_c, acc_p, eps_acc))
    return


def effective_epsilon_stats_dataset(ds: torchvision.datasets, p: nn.Module, c: nn.Module,
                                    f: Callable, bs: int) -> None:
    print('Epsilon ({} vs {})'.format(p.title, c.title))
    ratio = 100. * (1 - c.params / p.params)
    print('p_model {} params, c_model {} params, compression ratio={:,.2f}%'.format(p.params, c.params, ratio))
    print('\tds: {}'.format(data_set_size_to_str(ds)))
    loss_p, acc_p, _ = TrainTestScript.test_using_dataset(ds=ds, model=p, f=f, bs=bs)
    loss_c, acc_c, _ = TrainTestScript.test_using_dataset(ds=ds, model=c, f=f, bs=bs)

    eps = 1e-8  # not to divide in zero

    eps_loss = abs(1 - loss_c / loss_p) if loss_p > eps else abs(1 - loss_c / (loss_p + eps))
    print('\tloss_c:{:,.4f}, loss_p:{:,.4f}, eps_loss:{:,.4f}'.format(loss_c, loss_p, eps_loss))

    eps_acc = abs(1 - acc_c / acc_p) if acc_p > eps else abs(1 - acc_c / (acc_p + eps))
    print('\tacc_c:{:,.4f}, acc_p:{:,.4f}, eps_acc:{:,.4f}'.format(acc_c, acc_p, eps_acc))
    return


def balance_example_using_dls():
    """
    Balancing:
    Epoch 1/2:
        Batch   0/469 (    0/60000) - Loss: 0.038401, loss_c:2.375069, loss_p:2.287236
        Batch 300/469 (38400/60000) - Loss: 0.003460, loss_c:2.278306, loss_p:2.286216
        Epoch [1/2] loss:0.004335, loss_c:2.289958, loss_p:2.288310, time 00:00:09 New best found
    Epoch 2/2:
        optimizer changed = Adam (Parameter Group 0 amsgrad: False betas: (0.9, 0.999) eps: 1e-08
            lr: 0.0001 weight_decay: 0.0001)
        Batch   0/469 (    0/60000) - Loss: 0.001942, loss_c:2.282794, loss_p:2.287236
        Batch 300/469 (38400/60000) - Loss: 0.000319, loss_c:2.285487, loss_p:2.286216
        Epoch [2/2] loss:0.002182, loss_c:2.288429, loss_p:2.288310, time 00:00:19 New best found
    Done balancing: On epoch 2: loss=0.002182
    """
    import torch.nn.functional as func
    from Dependencies import GetDataScript
    from Dependencies.models.LeNet import LeNetMnist

    us.make_cuda_invisible()
    us.set_cuda_scope_and_seed(seed=42)

    p = LeNetMnist(path='./p_LeNetMnist_300_100.pt', fc1=300, fc2=100)
    c = LeNetMnist(path='./c_LeNetMnist_30_10.pt', fc1=30, fc2=10)

    train_loader = GetDataScript.get_data_loader(ds_name='mnist', data_root='../Datasets', is_train=True, bs=128,
                                                 shuffle=False, transform=None, data_limit=0)

    opt = torch.optim.Adam(c.parameters(), lr=0.001, weight_decay=0.0001)
    balance_using_dl(dl=train_loader, trainable_model=c, fixed_model=p, opt=opt, f=func.cross_entropy, epochs=2,
                     save_models=False, b_interval=300, lr_cps=[0.51])
    effective_epsilon_stats_dataloader(loader=train_loader, p=p, c=c, f=func.cross_entropy)
    return


def balance_example_using_datasets():
    """
    Balancing:
    Epoch 1/2:
        Batch   0/469 (    0/60000) - Loss: 0.038401, loss_c:2.375069, loss_p:2.287236
        Batch 300/469 (38400/60000) - Loss: 0.003460, loss_c:2.278306, loss_p:2.286216
        Epoch [1/2] loss:0.004335, loss_c:2.289958, loss_p:2.288310, time 00:00:02 New best found
    Epoch 2/2:
        optimizer changed = Adam (Parameter Group 0 amsgrad: False betas: (0.9, 0.999) eps: 1e-08
            lr: 0.0001 weight_decay: 0.0001)
        Batch   0/469 (    0/60000) - Loss: 0.001942, loss_c:2.282794, loss_p:2.287236
        Batch 300/469 (38400/60000) - Loss: 0.000319, loss_c:2.285487, loss_p:2.286216
        Epoch [2/2] loss:0.002182, loss_c:2.288429, loss_p:2.288310, time 00:00:04 New best found
    Done balancing: On epoch 2: loss=0.002182
    """
    import torch.nn.functional as func
    from Dependencies import GetDataScript
    from Dependencies.models.LeNet import LeNetMnist

    us.make_cuda_invisible()
    us.set_cuda_scope_and_seed(seed=42)

    p = LeNetMnist(path='./p_LeNetMnist_300_100.pt', fc1=300, fc2=100)
    c = LeNetMnist(path='./c_LeNetMnist_30_10.pt', fc1=30, fc2=10)

    train_ds = GetDataScript.get_data_set_transformed(ds_name='mnist', data_root='../Datasets', is_train=True,
                                                      transform=None,
                                                      data_limit=0, download=False, ack=True)

    opt = torch.optim.Adam(c.parameters(), lr=0.001, weight_decay=0.0001)
    balance_using_datasets(ds=train_ds, bs=128, trainable_model=c, fixed_model=p, opt=opt, f=func.cross_entropy,
                           epochs=2, save_models=False, b_interval=300, lr_cps=[0.51])
    effective_epsilon_stats_dataset(ds=train_ds, p=p, c=c, f=func.cross_entropy, bs=1000)
    return


if __name__ == "__main__":
    # us.make_cuda_invisible()
    # balance_example_using_dls()
    balance_example_using_datasets()
