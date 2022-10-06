import torch
import torch.nn.functional as func
import torch.nn as nn
import torchvision
from typing import Callable
import sys
import os

from Dependencies.models.LeNet import LeNetMnist
from Dependencies import GetDataScript
from Dependencies import TrainTestScript
from Dependencies import BalanceSgdScript
from Dependencies import UtilsScript as us


def get_datasets(argv: dict) -> (torchvision.datasets, torchvision.datasets):
    assert argv['ds']['name'] in ['cifar10', 'mnist'], 'invalid ds name: {}'.format(argv['ds']['name'])
    train_ds = GetDataScript.get_data_set_transformed(ds_name=argv['ds']['name'], data_root=argv['ds']['root'],
                                                      is_train=True, transform=None,
                                                      data_limit=argv['ds']['data_limit'], download=False, ack=True)
    test_ds = GetDataScript.get_data_set_transformed(ds_name=argv['ds']['name'], data_root=argv['ds']['root'],
                                                     is_train=False, transform=None,
                                                     data_limit=argv['ds']['data_limit'], download=False, ack=True)
    return train_ds, test_ds


def get_model(argv: dict, model: nn.Module, train_ds: torchvision.datasets, test_ds: torchvision.datasets, opt_d: dict,
              epochs: int, bs_train: int, shuffle: bool):
    if not os.path.exists(model.path):
        print('{} doesn\'t exists...'.format(model.title))
        us.set_model_status(model, status=True, status_print=True)
        # print(us.model_summary_to_string(model, input_size=argv['ds']['input_size'], batch_size=-1))
        us.model_params_print(model)
        opt = us.get_opt_by_name(opt_d, model)
        TrainTestScript.train_using_datasets(train_ds, test_ds, model, opt, argv['global']['f'], train_bs=bs_train,
                                             test_bs=argv['global']['bs_test'], epochs=epochs,
                                             use_acc=argv['global']['use_acc'],
                                             save_models=argv['global']['save_models'],
                                             shuffle=shuffle)
    if argv['global']['save_models']:
        us.load_model(model, ack_print=True)

    us.set_model_status(model, status=False, status_print=True)
    return


def balance_layer_i(argv: dict, p_model: nn.Module, ci_model: nn.Module, ci_prev_model: nn.Module, tr_layers: list,
                    ds: torchvision.datasets):
    if not os.path.exists(ci_model.path):
        print('{} doesn\'t exists'.format(ci_model.title))
        us.freeze_layers(ci_model, trainable_layers=tr_layers)
        us.model_params_print(ci_model)
        us.copy_models(ci_prev_model, ci_model, dont_copy_layers=tr_layers, ack=True)
        opt_ci = us.get_opt_by_name(argv['coreset']['opt_dict'], ci_model)
        BalanceSgdScript.balance_using_datasets(ds=ds, bs=argv['coreset']['bs_train'], trainable_model=ci_model,
                                                fixed_model=p_model, opt=opt_ci, f=argv['global']['f'],
                                                epochs=argv['coreset']['epochs'],
                                                save_models=argv['global']['save_models'],
                                                shuffle=argv['coreset']['shuffle'], b_interval=-1, lr_cps=None)
    if argv['global']['save_models']:
        us.load_model(ci_model, ack_print=True)
    return


def get_argv():
    # noinspection PyDictCreation
    argv = {}

    # dataset params
    argv['ds'] = {
        'name': 'mnist',
        'root': '../../Datasets',
        'data_limit': 0,
        'input_size': (1, 28, 28),
    }

    # global params
    argv['global'] = {
        'seed': 42,
        'mode': 'fs',  # fs for full_sweep, lbl for layer by layer
        'f': func.cross_entropy,  # func.nll_loss, func.cross_entropy
        'save_models': True,
        'use_acc': False,
        'bs_test': 1000
    }

    ADAM_OPT_L2 = {'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0.0001}
    ADAM_OPT = {'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0}
    # SGD_0_01_OPT = {'name': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}
    # SGD_0_001_OPT = {'name': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0001}

    # full model
    argv['full_model'] = {
        'bs_train': 64,
        'shuffle': False,
        'epochs': 40,
        'fc1': 300,
        'fc2': 100,
        'opt_dict': ADAM_OPT,
        'path': './p_LeNetMnist_300_100.pt'
    }

    # coreset: compressed model
    # BEN: LeNet-300-100
    # full model err 2.16 (Acc=97.84) 267K params
    # c    model err 2.03 (Acc=97.97) 26K params

    b_fc1, b_fc2 = 30, 100
    if argv['global']['mode'] == 'fs':
        # all 1 balance - sizes are fixed
        c_model_base_path = './results/full_sweep_c_LeNetMnist_{}_{}'.format(b_fc1, b_fc2)
    elif argv['global']['mode'] == 'lbl':
        # each layer sizes are different
        # layer index and sizes will be inserted later by layer
        c_model_base_path = './results/lbl_c{}_LeNetMnist_{}_{}'
    else:
        c_model_base_path = '.'

    argv['coreset'] = {
        'bs_train': 500,
        'shuffle': False,
        'epochs': 400,
        'fc1': b_fc1,
        'fc2': b_fc2,
        'opt_dict': ADAM_OPT,
        'path_base': '{}.pt'.format(c_model_base_path)
    }

    # fine tune coreset
    argv['ft'] = {
        'bs_train': 128,
        'shuffle': False,
        'epochs': 10,  # post each layer balance - ft some epochs. 0 for no ft post layer
        'epochs_end': 400,  # the last ft - more epochs
        'opt_dict': ADAM_OPT_L2,
        'path_base': '{}_ft.pt'.format(c_model_base_path)
    }
    argv['ft']['path_base'] = argv['ft']['path_base'].replace('.pt', '_exp2_shuffle.pt')

    # # DEBUG values
    # argv['ds']['data_limit'] = 10
    # argv['global']['save_models'] = False
    # argv['full_model']['epochs'] = 1
    # argv['full_model']['path'] = './p_LeNetMnist_300_100_debug.pt'
    # argv['coreset']['path_base'] = '{}_debug.pt'.format(c_model_base_path)
    # argv['ft']['path_base'] = '{}_ft_debug.pt'.format(c_model_base_path)
    # argv['coreset']['epochs'] = 1
    # argv['ft']['epochs'] = 1
    # argv['ft']['epochs_end'] = 1

    for k, v in argv.items():
        print('\t{}: {}'.format(k, v))

    return argv


def main_Lenet_full_sweep(argv: dict, train_ds: torchvision.datasets, test_ds: torchvision.datasets, p_model: nn.Module,
                          bs_test: int, loss_f: Callable):
    print('\nmain_Lenet_full_sweep():')
    c_model_path = argv['coreset']['path_base']
    c_model = LeNetMnist(path=c_model_path, fc1=argv['coreset']['fc1'], fc2=argv['coreset']['fc2'])

    us.model_params_print(p_model)
    us.model_params_print(c_model)

    tr_layers_indices = list(c_model.get_mapping_dict().keys())  # all layers trainable on full sweep
    tr_layers_values = c_model.get_layers_components(tr_layers_indices)

    balance_layer_i(argv=argv, p_model=p_model, ci_model=c_model, ci_prev_model=p_model, tr_layers=tr_layers_values,
                    ds=train_ds)
    TrainTestScript.test_datasets_check(train_ds=train_ds, test_ds=test_ds, model=c_model, f=loss_f, bs=bs_test)

    # fine tune
    c_model.set_new_path(new_path=argv['ft']['path_base'])
    get_model(argv=argv, model=c_model, train_ds=train_ds, test_ds=test_ds, opt_d=argv['ft']['opt_dict'],
              epochs=argv['ft']['epochs_end'], bs_train=argv['ft']['bs_train'], shuffle=argv['ft']['shuffle'])
    TrainTestScript.test_datasets_check(train_ds=train_ds, test_ds=test_ds, model=c_model, f=loss_f, bs=bs_test)

    print('\neffective_epsilons:')
    BalanceSgdScript.effective_epsilon_stats_dataset(ds=train_ds, p=p_model, c=c_model, f=loss_f,
                                                     bs=bs_test)
    BalanceSgdScript.effective_epsilon_stats_dataset(ds=test_ds, p=p_model, c=c_model, f=loss_f, bs=bs_test)

    log_path = c_model.path.replace('.pt', '.txt')
    if argv['global']['save_models'] and not os.path.exists(log_path):
        f = open(log_path, 'w')
        f.close()
    return


def main_Lenet_lbl(argv: dict, train_ds: torchvision.datasets, test_ds: torchvision.datasets, p_model: nn.Module,
                   bs_test: int, loss_f: Callable):
    print('\nmain_Lenet_lbl():')

    # balance layer fc1
    print('c1_model:')
    post_layer = 1
    c1_fc1, c1_fc2 = argv['coreset']['fc1'], argv['full_model']['fc2']
    c1_model_path = argv['coreset']['path_base'].format(post_layer, c1_fc1, c1_fc2)
    c1_model = LeNetMnist(path=c1_model_path, fc1=c1_fc1, fc2=c1_fc2)

    tr_layers_indices = [1, 2]  # layer 1 and 2 are trainable
    tr_layers_values = c1_model.get_layers_components(tr_layers_indices)
    balance_layer_i(argv=argv, p_model=p_model, ci_model=c1_model, ci_prev_model=p_model, tr_layers=tr_layers_values,
                    ds=train_ds)
    TrainTestScript.test_datasets_check(train_ds=train_ds, test_ds=test_ds, model=c1_model, f=loss_f, bs=bs_test)

    if argv['ft']['epochs'] > 0:  # fine tune mode
        # fine tune layer fc1
        c1_model.set_new_path(new_path=argv['ft']['path_base'].format(post_layer, c1_fc1, c1_fc2))
        get_model(argv=argv, model=c1_model, train_ds=train_ds, test_ds=test_ds, opt_d=argv['ft']['opt_dict'],
                  epochs=argv['ft']['epochs'], bs_train=argv['ft']['bs_train'], shuffle=argv['ft']['shuffle'])
        TrainTestScript.test_datasets_check(train_ds=train_ds, test_ds=test_ds, model=c1_model, f=loss_f, bs=bs_test)

    # balance layer fc2
    print('\nc2_model:')
    post_layer = 2
    c2_fc1, c2_fc2 = argv['coreset']['fc1'], argv['coreset']['fc2']
    c2_model_path = argv['coreset']['path_base'].format(post_layer, c2_fc1, c2_fc2)
    c2_model = LeNetMnist(path=c2_model_path, fc1=c2_fc1, fc2=c2_fc2)

    tr_layers_indices = [2, 3]  # layer 2 and 3 are trainable
    tr_layers_values = c1_model.get_layers_components(tr_layers_indices)
    balance_layer_i(argv=argv, p_model=p_model, ci_model=c2_model, ci_prev_model=c1_model,
                    tr_layers=tr_layers_values, ds=train_ds)
    TrainTestScript.test_datasets_check(train_ds=train_ds, test_ds=test_ds, model=c1_model, f=loss_f, bs=bs_test)

    # fine tune layer fc2
    c2_model.set_new_path(new_path=argv['ft']['path_base'].format(post_layer, c2_fc1, c2_fc2))
    get_model(argv=argv, model=c2_model, train_ds=train_ds, test_ds=test_ds, opt_d=argv['ft']['opt_dict'],
              epochs=argv['ft']['epochs_end'], bs_train=argv['ft']['bs_train'], shuffle=argv['ft']['shuffle'])
    TrainTestScript.test_datasets_check(train_ds=train_ds, test_ds=test_ds, model=c2_model, f=loss_f, bs=bs_test)

    print('\neffective_epsilons:')
    BalanceSgdScript.effective_epsilon_stats_dataset(ds=train_ds, p=p_model, c=c2_model, f=loss_f,
                                                     bs=bs_test)
    BalanceSgdScript.effective_epsilon_stats_dataset(ds=test_ds, p=p_model, c=c2_model, f=loss_f,
                                                     bs=bs_test)
    log_path = c2_model.path.replace('.pt', '.txt')
    if argv['global']['save_models'] and not os.path.exists(log_path):
        f = open(log_path, 'w')
        f.close()
    return


def main():
    argv = get_argv()
    bs_test = argv['global']['bs_test']
    loss_f = argv['global']['f']
    us.set_cuda_scope_and_seed(seed=argv['global']['seed'])
    train_ds, test_ds = get_datasets(argv)

    # parent model
    print('\np_model:')
    p_model = LeNetMnist(path=argv['full_model']['path'], fc1=argv['full_model']['fc1'], fc2=argv['full_model']['fc2'])
    get_model(argv, model=p_model, train_ds=train_ds, test_ds=test_ds, opt_d=argv['full_model']['opt_dict'],
              epochs=argv['full_model']['epochs'], bs_train=argv['full_model']['bs_train'],
              shuffle=argv['full_model']['shuffle'])
    log_path = p_model.path.replace('.pt', '.txt')
    if argv['global']['save_models'] and not os.path.exists(log_path):
        f = open(log_path, 'w')
        f.close()
    TrainTestScript.test_datasets_check(train_ds, test_ds, p_model, loss_f, bs=bs_test)

    if argv['global']['mode'] == 'fs':
        main_Lenet_full_sweep(argv, train_ds, test_ds, p_model, bs_test, loss_f)
    elif argv['global']['mode'] == 'lbl':
        main_Lenet_lbl(argv, train_ds, test_ds, p_model, bs_test, loss_f)
    return


if __name__ == "__main__":
    us.make_cuda_invisible()
    g_start_time = us.get_current_time()
    print("Python Version {}".format(sys.version))
    print("PyTorch Version {}".format(torch.__version__))
    print("Working directory {}".format(os.path.abspath(os.getcwd())))
    main()
    print('Total run time {}'.format(us.get_time_str(g_start_time)))
