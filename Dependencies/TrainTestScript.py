import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
import torchvision
import math
from Dependencies import UtilsScript as us
from Dependencies import GetDataScript


def run_info(model, ds_train, ds_test, use_acc, epochs, train_bs, total_train_batches, test_bs, total_test_batches,
             optimizer, lr_cps):
    print('{} - {}:'.format(us.get_function_name(depth=2), model.title))
    print('\tTrain data: {}'.format(us.data_set_size_to_str(ds_train)))
    print('\tTest data: {}'.format(us.data_set_size_to_str(ds_test)))
    print('\tsaving on each {} improvment'.format('accuracy' if use_acc else 'loss'))
    msg = '\tepochs={}, bs_train={}, #trainBatches={}, bs_test={}, #testBatches={}'
    print(msg.format(epochs, train_bs, total_train_batches, test_bs, total_test_batches))
    print('\toptimizer = {}'.format(us.opt_to_str(optimizer)))
    if len(lr_cps) > 0:
        print('\tif epoch in {}: lr/=10'.format(lr_cps))
    return


# CLASSIFICATION PROBLEM
def train_using_dls(train_dl: DataLoader, test_dl: DataLoader, model: nn.Module, opt: torch.optim, f: Callable,
                    epochs: int, use_acc: bool, save_models: bool, b_interval: int = -1, lr_cps: list = None) -> None:
    """
    training using dataloaders - no input weight supported.
    :param train_dl: DataLoader
    :param test_dl: DataLoader
    :param model: nn model
    :param opt:
    :param f: loss function. f(pred, y, reduction='mean' or 'sum')
    :param epochs:
    :param use_acc: save model metric (loss or accuracy)
    :param save_models: if False, you will get last epoch model (not necessarily the best one)
    :param b_interval: batch interval. if batch_ind % b_interval == 0: prints batch stats. -1 is no batch stats
    :param lr_cps: list of floats. e.g. lr_cps=[0.5, 0.75]. in epoch=ceil(epochs*0.5) and ceil(epochs*0.75): lr/=10
    :return: None - train model by reference
    you should load model after training
    """
    train_time = us.get_current_time()

    train_bs = train_dl.batch_size
    train_size = len(train_dl.dataset)
    train_batches = len(train_dl)
    test_bs = test_dl.batch_size
    test_batches = len(test_dl)
    lr_cps = [] if lr_cps is None else [math.ceil(epochs * cp) for cp in lr_cps]

    run_info(model, train_dl.dataset, test_dl.dataset, use_acc, epochs, train_bs, train_batches, test_bs, test_batches,
             opt, lr_cps)

    # saves the stats by loss
    by_loss_lowest_loss = float("inf")
    by_loss_acc = -1
    by_loss_best_epoch = -1

    # saves the stats by acc
    by_acc_highest_acc = -1
    by_acc_loss = float("inf")
    by_acc_best_epoch = -1

    print('Training:')
    for epoch in range(1, epochs + 1):
        model.train()  # test sets model.eval()
        if b_interval > 0:
            print('Epoch {}/{}:'.format(epoch, epochs))
        if epoch in lr_cps:
            us.set_lr(opt, us.get_lr(opt) / 10)
            print('\toptimizer changed = {}'.format(us.opt_to_str(opt)))
        sum_batches_loss = 0.0
        for b_i, (batchX, batch_y) in enumerate(train_dl):
            batchX, batch_y = us.add_cuda(batchX), us.add_cuda(batch_y)
            batch_y_pred_vectors = model(batchX)
            loss = f(batch_y_pred_vectors, batch_y, reduction='mean')
            # ============ Backward ============
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_batches_loss += loss.item()
            if b_interval > 0 and b_i % b_interval == 0:
                msg = '\tBatch {:3}/{:3} ({:5}/{:5}) - Loss: {:.6f}'
                print(msg.format(b_i, train_batches, b_i * train_bs, train_size, loss.item()))

        # for each batch we summed its mean. divide by train_batches
        train_loss = sum_batches_loss / train_batches
        test_loss, test_acc, test_res_str = test_using_dl(test_dl, model, f)

        imp_acc, imp_loss = False, False
        if test_acc > by_acc_highest_acc:  # found new highest acc so far
            by_acc_highest_acc = test_acc
            by_acc_loss = test_loss
            by_acc_best_epoch = epoch
            imp_acc = True
        if test_loss < by_loss_lowest_loss:  # found new lowest loss so far
            by_loss_lowest_loss = test_loss
            by_loss_acc = test_acc
            by_loss_best_epoch = epoch
            imp_loss = True

        # save by use_acc value (or in first epoch)
        msg = '\tEpoch [{:3}/{:3}] train loss:{:.6f},test {}, time so far {}'
        if (use_acc and imp_acc) or (not use_acc and imp_loss) or (epoch == 1):
            msg += ' New best found'
            if save_models:
                us.save_model(model, ack_print=False)
        print(msg.format(epoch, epochs, train_loss, test_res_str, us.get_time_str(train_time)))

    end_msg1 = 'Done training by {}: On epoch {}: loss={:.6f}, accuracy = {}%'
    end_msg2 = 'If trained by {}   : On epoch {}: loss={:.6f}, accuracy = {}%'
    if use_acc:
        print(end_msg1.format('accuracy', by_acc_best_epoch, by_acc_loss, by_acc_highest_acc))
        print(end_msg2.format('loss    ', by_loss_best_epoch, by_loss_lowest_loss, by_loss_acc))
    else:
        print(end_msg1.format('loss    ', by_loss_best_epoch, by_loss_lowest_loss, by_loss_acc))
        print(end_msg2.format('accuracy', by_acc_best_epoch, by_acc_loss, by_acc_highest_acc))
    return


def test_using_dl(dl: DataLoader, model: nn.Module, f: Callable, print_res: bool = False) -> (float, float, str):
    """
    test using dataloaders - no input weight supported.
    :param dl: DataLoader
    :param model: nn model
    :param f: loss function. f(pred, y, reduction='mean' or 'sum')
    :param print_res:
    """
    model.eval()
    total_images = len(dl.dataset)

    with torch.no_grad():
        sum_losses, correct = 0, 0
        for batchX, batch_y in dl:
            batchX, batch_y = us.add_cuda(batchX), us.add_cuda(batch_y)
            batch_y_pred_vectors = model(batchX)

            sum_losses += f(batch_y_pred_vectors, batch_y, reduction='sum').item()  # sum up batch loss
            predictions = batch_y_pred_vectors.argmax(dim=1, keepdim=True)
            correct += predictions.eq(batch_y.view_as(predictions)).sum().item()

    avg_loss = sum_losses / total_images
    acc = 100. * correct / total_images
    result_str = 'loss={:.6f}, accuracy={}/{} ({:.2f}%)'.format(avg_loss, correct, total_images, acc)
    if print_res:
        print(result_str)
    return avg_loss, acc, result_str


def train_using_datasets(train_ds: torchvision.datasets, test_ds: torchvision.datasets, model: nn.Module,
                         opt: torch.optim, f: Callable, train_bs: int, test_bs: int, epochs: int, shuffle: bool = False,
                         use_acc: bool = True, save_models: bool = True, b_interval: int = -1,
                         lr_cps: list = None) -> None:
    """
    training using datasets - no input weight supported.
    :param train_ds: torchvision.datasets
    :param test_ds: torchvision.datasets
    :param model: nn model
    :param opt:
    :param f: loss function. f invoked: f(labels_predictions, labels, weight=wB, reduction='mean')
    :param train_bs:
    :param test_bs:
    :param epochs:
    :param shuffle: if True: shuffle train.ds each epoch
    :param use_acc: capture improvment using accuracy or loss
    :param save_models: if False, you will get last epoch model (not necessarily the best one)
    :param b_interval: batch interval. if batch_ind % b_interval == 0: prints batch stats. -1 is no batch stats
    :param lr_cps: list of floats. e.g. lr_cps=[0.5, 0.75]. in epoch=ceil(epochs*0.5) and ceil(epochs*0.75): lr/=10
    :return: None - train model by reference
    you should load model after training
    """
    train_time = us.get_current_time()

    train_size = len(train_ds)
    train_batches = math.ceil(train_size / train_bs)
    test_size = len(test_ds)
    test_batches = math.ceil(test_size / test_bs)
    lr_cps = [] if lr_cps is None else [math.ceil(epochs * cp) for cp in lr_cps]

    run_info(model, train_ds, test_ds, use_acc, epochs, train_bs, train_batches, test_bs, test_batches, opt, lr_cps)

    # saves the stats by loss
    by_loss_lowest_loss = float("inf")
    by_loss_acc = -1
    by_loss_best_epoch = -1

    # saves the stats by acc
    by_acc_highest_acc = -1
    by_acc_loss = float("inf")
    by_acc_best_epoch = -1

    print('Training:')
    for epoch in range(1, epochs + 1):
        model.train()  # test sets model.eval()
        if b_interval > 0:
            print('Epoch {}/{}:'.format(epoch, epochs))
        if epoch in lr_cps:
            us.set_lr(opt, us.get_lr(opt) / 10)
            print('\toptimizer changed = {}'.format(us.opt_to_str(opt)))
        if shuffle:
            us.shuffle_ds(train_ds)
        sum_batches_loss = 0.0
        for i in range(0, train_size, train_bs):
            batchX, batch_y = us.add_cuda(train_ds.data[i:i + train_bs]), us.add_cuda(train_ds.targets[i:i + train_bs])
            batch_y_pred_vectors = model(batchX)
            loss = f(batch_y_pred_vectors, batch_y, reduction='mean')
            # ============ Backward ============
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_batches_loss += loss.item()
            batch_ind = int(i / train_bs)
            if b_interval > 0 and batch_ind % b_interval == 0:
                msg = '\tBatch {:3}/{:3} ({:5}/{:5}) - Loss: {:.6f}'
                print(msg.format(batch_ind, train_batches, i, train_size, loss.item()))

        train_loss = sum_batches_loss / train_batches
        test_loss, test_acc, test_res_str = test_using_dataset(test_ds, model, f, bs=test_bs)

        imp_acc, imp_loss = False, False
        if test_acc > by_acc_highest_acc:  # found new highest acc so far
            by_acc_highest_acc = test_acc
            by_acc_loss = test_loss
            by_acc_best_epoch = epoch
            imp_acc = True
        if test_loss < by_loss_lowest_loss:  # found new lowest loss so far
            by_loss_lowest_loss = test_loss
            by_loss_acc = test_acc
            by_loss_best_epoch = epoch
            imp_loss = True

        # save by use_acc value (or in first epoch)
        msg = '\tEpoch [{:3}/{:3}] train loss:{:.6f},test {}, time so far {}'
        if (use_acc and imp_acc) or (not use_acc and imp_loss) or (epoch == 1):
            msg += ' New best found'
            if save_models:
                us.save_model(model, ack_print=False)
        print(msg.format(epoch, epochs, train_loss, test_res_str, us.get_time_str(train_time)))

    end_msg1 = 'Done training by {}: On epoch {}: loss={:.6f}, accuracy = {}%'
    end_msg2 = 'If trained by {}   : On epoch {}: loss={:.6f}, accuracy = {}%'
    if use_acc:
        print(end_msg1.format('accuracy', by_acc_best_epoch, by_acc_loss, by_acc_highest_acc))
        print(end_msg2.format('loss    ', by_loss_best_epoch, by_loss_lowest_loss, by_loss_acc))
    else:
        print(end_msg1.format('loss    ', by_loss_best_epoch, by_loss_lowest_loss, by_loss_acc))
        print(end_msg2.format('accuracy', by_acc_best_epoch, by_acc_loss, by_acc_highest_acc))
    return


def test_using_dataset(ds: torchvision.datasets, model: nn.Module, f: Callable, bs: int, print_res: bool = False) -> (
        float, float, str):
    """
    test using dataloaders - no input weight supported.
    :param ds: torchvision.datasets
    :param model: nn model
    :param f: loss function. f(pred, y, reduction='mean' or 'sum')
    :param bs:
    :param print_res:
    """
    model.eval()
    total_images = ds.targets.shape[0]

    with torch.no_grad():
        sum_losses, correct = 0, 0
        for i in range(0, total_images, bs):
            batchX, batch_y = us.add_cuda(ds.data[i:i + bs]), us.add_cuda(ds.targets[i:i + bs])
            batch_y_pred_vectors = model(batchX)

            sum_losses += f(batch_y_pred_vectors, batch_y, reduction='sum').item()  # sum up batch loss
            predictions = batch_y_pred_vectors.argmax(dim=1, keepdim=True)
            correct += predictions.eq(batch_y.view_as(predictions)).sum().item()

    avg_loss = sum_losses / total_images
    acc = 100. * correct / total_images
    result_str = 'loss={:.6f}, accuracy={}/{} ({:.2f}%)'.format(avg_loss, correct, total_images, acc)
    if print_res:
        print(result_str)
    return avg_loss, acc, result_str


# def train_using_tensors(train_X: torch.Tensor, train_y: torch.Tensor, test_X: torch.Tensor, test_y: torch.Tensor,
#                         model: nn.Module, optimizer: torch.optim, f: Callable, bs_train: int, bs_test: int,
#                         epochs: int,
#                         shuffle: bool = False, use_acc: bool = True, save_models: bool = True,
#                         w: torch.Tensor = None) -> None:
#     """
#     train using tensors
#     train_X: |train_X|=n1xd
#     train_y: |train_y|=n1
#     test_X:  |test_X|=n2xd
#     test_y:  |test_y|=n2
#     see train_using_datasets function
#     todo convert Train_X and train_y to train_ds
#     todo convert test_X and test_y to test_ds
#     todo invoke train_using_datasets
#     """
#     # todo add lr_cps
#     # todo add after testing model.train()
#     return


def test_datasets_check(train_ds: torchvision.datasets, test_ds: torchvision.datasets, model: nn.Module, f: Callable,
                        bs: int):
    train_loss, train_acc, train_result_str = test_using_dataset(train_ds, model, f, bs=bs)
    test_loss, test_acc, test_result_str = test_using_dataset(test_ds, model, f, bs=bs)
    print('{} test_datasets:'.format(model.title))
    print('\ttrain {}'.format(train_result_str))
    print('\ttest  {}'.format(test_result_str))
    return


def test_loaders_check(train_dl: DataLoader, test_dl: DataLoader, model: nn.Module, f: Callable):
    train_loss, train_acc, train_result_str = test_using_dl(train_dl, model, f)
    test_loss, test_acc, test_result_str = test_using_dl(test_dl, model, f)
    print('{} test loaders:'.format(model.title))
    print('\ttrain {}'.format(train_result_str))
    print('\ttest  {}'.format(test_result_str))
    return


def speed_test_datasets_vs_dataloaders(m1: nn.Module, m2: nn.Module, f: Callable, ds_name: str, ds_root: str,
                                       seed: int, bs_train: int, bs_test: int, data_limit: int, epochs: int,
                                       opt_d: dict, b_interval: int):
    """
    datasets are much faster than dataloaders (when we pre-process the transform)
    mnist LeNet 300 100:
    loading data:
        dataloaders:
            CPU: 0.031 seconds, GPU: 0.016 seconds
        as tensor (preprocess transform):
            CPU: 6.275 seconds, GPU: 11.326 seconds
    train 5 epochs:
    Notice - to be clear - same exact output(on all epochs) for training on same device
        dataloaders:
            CPU: 64.139 seconds, GPU: 55.497 seconds
        as tensor (preprocess transform):
            CPU: 25.102 seconds, GPU: 17.318 seconds
    """
    print('dataloaders:')
    t = us.get_current_time()
    us.set_cuda_scope_and_seed(seed=seed)
    train_loader = GetDataScript.get_data_loader(ds_name, ds_root, is_train=True, bs=bs_train, transform=None,
                                                 shuffle=False, data_limit=data_limit, download=False)
    test_loader = GetDataScript.get_data_loader(ds_name, ds_root, is_train=False, bs=bs_test, transform=None,
                                                shuffle=False, data_limit=data_limit, download=False)
    print('\tloading time {}'.format(us.get_mili_seconds_str(t)))

    t = us.get_current_time()
    opt = us.get_opt_by_name(opt_d, m1)
    train_using_dls(train_loader, test_loader, m1, opt, f, epochs=epochs, use_acc=False, save_models=False,
                    b_interval=b_interval, lr_cps=None)
    print('\ttraining time {}'.format(us.get_mili_seconds_str(t)))

    print('datasets:')
    t = us.get_current_time()
    us.set_cuda_scope_and_seed(seed=seed)
    train_ds = GetDataScript.get_data_set_transformed(ds_name=ds_name, data_root=ds_root, is_train=True, transform=None,
                                                      data_limit=data_limit, download=False, ack=True)
    test_ds = GetDataScript.get_data_set_transformed(ds_name=ds_name, data_root=ds_root, is_train=False, transform=None,
                                                     data_limit=data_limit, download=False, ack=True)
    print('\tloading time {}'.format(us.get_mili_seconds_str(t)))
    t = us.get_current_time()
    opt = us.get_opt_by_name(opt_d, m2)
    train_using_datasets(train_ds, test_ds, m2, opt, f, train_bs=bs_train, test_bs=bs_test, epochs=epochs,
                         shuffle=False, use_acc=False, save_models=False, b_interval=b_interval, lr_cps=None)
    print('\ttraining time {}'.format(us.get_mili_seconds_str(t)))
    return

# def train_using_datasets(train_ds: torchvision.datasets, test_ds: torchvision.datasets, model: nn.Module,
#                          optimizer: torch.optim, f: Callable, bs_train: int, bs_test: int, epochs: int,
#                          shuffle: bool = False, use_acc: bool = True, save_models: bool = True,
#                          w_train: torch.Tensor = None, w_test: torch.Tensor = None) -> None:
#     """
#     example using weighted input - see LossesScript for an example loss function with W
#     :param w_train: weights for train_ds.data, |w|=|train_ds.data|.
#     :param w_test: weights for test_ds.data, |w|=|test_ds.data|.
#               if not None, f invoked: f(labels_predictions, labels, weight=wB, reduction='mean')
#     """
#
#     for epoch in range(1, epochs + 1):
#         sum_batch_loss = 0.0
#         for i in range(0, total_images, bs_train):
#             XB, yB = UtilsScript.add_cuda(train_ds.data[i:i + bs_train]), UtilsScript.add_cuda(
#                 train_ds.targets[i:i + bs_train])
#             wB = UtilsScript.add_cuda(w_train[i:i + bs_train]) if w_train is not None else None
#             y_predictions_vectorsB = model(XB)
#             loss = f(y_predictions_vectorsB, yB, weight=wB, reduction='mean')
#             # if we want to check if w is considered
#             # loss2 = f(labels_predictions_vectorsB, yB, weight=None, reduction='mean')
#             # ============ Backward ============
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             sum_batch_loss += loss.item()
#
#         train_loss = sum_batch_loss / total_batches
#         test_loss, test_acc, test_res_str = test_ds(test_ds, model, f, bs=bs_test, w=w_test)
#         msg = '\tepoch [{:3}/{:3}] train loss:{:.6f},test {}, time so far {}'
#         print(msg.format(epoch, epochs, train_loss, test_res_str, UtilsScript.get_time_str(train_time)))
#     return
#
#
# def test_using_dataset(ds: torchvision.datasets, model: nn.Module, f: Callable, bs: int,
#                        print_res: bool = False, w: torch.Tensor = None) -> (float, float, str):
#     """
#     example using weighted input - see LossesScript for an example loss function with W
#     :param f: loss function. f invoked: f(labels_predictions, labels, weight=wB, reduction='mean')
#     :param w: weights for ds.data, |w|=|ds.data|.
#               if not None, f invoked: f(labels_predictions, labels, weight=wB, reduction='mean')
#     :return:
#     """
#     with torch.no_grad():
#         sum_loss, correct = 0, 0
#         for i in range(0, total_images, bs):
#             XB, yB = UtilsScript.add_cuda(ds.data[i:i + bs]), UtilsScript.add_cuda(ds.targets[i:i + bs])
#             wB = UtilsScript.add_cuda(w[i:i + bs]) if w is not None else None
#             labels_predictions_vectorsB = model(XB)
#
#             sum_loss += f(labels_predictions_vectorsB, yB, weight=wB, reduction='sum').item()  # sum up batch loss
#             predictions = labels_predictions_vectorsB.argmax(dim=1, keepdim=True)
#             correct += predictions.eq(yB.view_as(predictions)).sum().item()
#
#     avg_loss = sum_loss / total_images
#     acc = 100. * correct / total_images
#     msg = 'loss={:.6f}, accuracy={}/{} ({:.2f}%)'
#     result_str = msg.format(avg_loss, correct, total_images, acc)
#     print(result_str)
#     return avg_loss, acc, result_str
