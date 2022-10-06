import torch
from torch.utils.data import DataLoader
import torch.nn.functional as func
import torch.nn as nn
import torchvision
import sys
import os

from Dependencies.models.Vgg import VggCifar10
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


def get_dataloaders(ds_name: str, ds_root: str, bs_train: int, bs_test: int, shf_train: bool, shf_test: bool = False,
                    data_limit: int = 0) -> (DataLoader, DataLoader):
    assert ds_name in ['cifar10', 'mnist'], 'invalid ds name: {}'.format(ds_name)
    train_loader = GetDataScript.get_data_loader(ds_name=ds_name, data_root=ds_root, is_train=True, bs=bs_train,
                                                 shuffle=shf_train, transform=None, data_limit=data_limit)
    test_loader = GetDataScript.get_data_loader(ds_name=ds_name, data_root=ds_root, is_train=False, bs=bs_test,
                                                shuffle=shf_test, transform=None, data_limit=data_limit)
    return train_loader, test_loader


def get_model(argv: dict, model: nn.Module, train_dl: DataLoader, test_dl: DataLoader, opt_d: dict, epochs: int,
              lr_cps: list):
    if not os.path.exists(model.path):
        print('{} doesn\'t exists...'.format(model.title))
        us.set_model_status(model, status=True, status_print=True)
        # print(us.model_summary_to_string(model, input_size=argv['ds']['input_size'], batch_size=-1))
        # us.model_params_print(model)
        opt = us.get_opt_by_name(opt_d, model)
        TrainTestScript.train_using_dls(train_dl=train_dl, test_dl=test_dl, model=model, opt=opt,
                                        f=argv['global']['f'], epochs=epochs, use_acc=argv['global']['use_acc'],
                                        save_models=argv['global']['save_models'],
                                        b_interval=argv['global']['batch_interval'], lr_cps=lr_cps)

    if argv['global']['save_models']:
        us.load_model(model, ack_print=True)

    us.set_model_status(model, status=False, status_print=True)
    return


def balance_layer_i(argv: dict, p_model: nn.Module, ci_model: nn.Module, ci_prev_model: nn.Module, tr_layers: list,
                    dl: DataLoader):
    if not os.path.exists(ci_model.path):
        print('{} doesn\'t exists'.format(ci_model.title))
        us.freeze_layers(ci_model, trainable_layers=tr_layers)
        us.model_params_print(ci_model)
        us.copy_models(ci_prev_model, ci_model, dont_copy_layers=tr_layers, ack=True)
        opt_ci = us.get_opt_by_name(argv['coreset']['opt_dict'], ci_model)
        BalanceSgdScript.balance_using_dl(dl=dl, trainable_model=ci_model, fixed_model=p_model, opt=opt_ci,
                                          f=argv['global']['f'], epochs=argv['coreset']['epochs'],
                                          save_models=argv['global']['save_models'],
                                          b_interval=argv['coreset']['batch_interval'],
                                          lr_cps=argv['coreset']['lr_cps'])
    if argv['global']['save_models']:
        us.load_model(ci_model, ack_print=True)
    return


def get_argv():
    # noinspection PyDictCreation
    argv = {}

    # dataset params
    argv['ds'] = {
        'name': 'cifar10',
        'root': '../../../Datasets',
        'data_limit': 0,
        'input_size': (3, 32, 32),  # if you want model summary
    }

    # global params
    argv['global'] = {
        'seed': 42,
        'mode': 'fs',  # fs for full_sweep, lbl for layer by layer
        'f': func.cross_entropy,
        'save_models': True,
        'use_acc': True,
        'bs_test': 256,
        'shuffle_test': False,
        'batch_interval': -1  # each 100 batches print batch stats
    }
    ADAM_OPT_NO_L2 = {'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0}
    # ADAM_OPT = {'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0.0001}
    SGD_OPT = {'name': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0001}

    # full model
    argv['full_model'] = {
        'bs_train': 64,
        'shuffle': False,
        'epochs': 160,
        'vgg_type': '19',
        'path': './vgg19-cifar10.pt',
        'opt': SGD_OPT,
        'lr_cps': [0.5, 0.75]  # checkpoints for opt.lr /= 10
    }

    c_model_base_path = './full_sweep_vgg19-cifar10_coreset'  # all 1 balance - no layers

    argv['coreset'] = {
        'bs_train': 512,
        'shuffle': False,
        'epochs': 180,
        'opt_dict': ADAM_OPT_NO_L2,
        'path_base': '{}.pt'.format(c_model_base_path),
        'vgg_type': '19_coreset',
        'batch_interval': 30,
        'lr_cps': [0.33, 0.66]  # checkpoints for opt.lr /= 10
    }

    # fine tune coreset
    argv['ft'] = {
        'bs_train': 64,
        'shuffle': True,
        'epochs': 10,  # post each layer balance - ft some epochs. 0 for no ft post layer
        'epochs_end': 180,  # the last ft - more epochs
        'opt_dict': SGD_OPT,
        'path_base': '{}_ft_11.pt'.format(c_model_base_path),
        'lr_cps': [0.5, 0.75]  # checkpoints for opt.lr /= 10
    }

    for k, v in argv.items():
        print('\t{}: {}'.format(k, v))

    return argv


def main_vgg19_full_sweep(argv: dict, p_model: nn.Module, test_dl: DataLoader):
    print('\nmain_vgg19_full_sweep():')
    print('c_model:')
    train_dl_bal = GetDataScript.get_data_loader(ds_name=argv['ds']['name'], data_root=argv['ds']['root'],
                                                 is_train=True, bs=argv['coreset']['bs_train'],
                                                 shuffle=argv['coreset']['shuffle'], transform=None,
                                                 data_limit=argv['ds']['data_limit'])
    c_model = VggCifar10(vgg_type=argv['coreset']['vgg_type'], path=argv['coreset']['path_base'])

    us.model_params_print(p_model)
    us.model_params_print(c_model)

    tr_layers_indices = list(c_model.get_mapping_dict().keys())  # all layers trainable on full sweep
    tr_layers_values = c_model.get_layers_components(tr_layers_indices)

    balance_layer_i(argv=argv, p_model=p_model, ci_model=c_model, ci_prev_model=p_model,
                    tr_layers=tr_layers_values, dl=train_dl_bal)
    TrainTestScript.test_loaders_check(train_dl_bal, test_dl, c_model, argv['global']['f'])

    # fine tune
    train_dl_ft = GetDataScript.get_data_loader(ds_name=argv['ds']['name'], data_root=argv['ds']['root'],
                                                is_train=True, bs=argv['ft']['bs_train'],
                                                shuffle=argv['ft']['shuffle'], transform=None,
                                                data_limit=argv['ds']['data_limit'])
    c_model.set_new_path(new_path=argv['ft']['path_base'])
    get_model(argv=argv, model=c_model, train_dl=train_dl_ft, test_dl=test_dl, opt_d=argv['ft']['opt_dict'],
              epochs=argv['ft']['epochs_end'], lr_cps=argv['ft']['lr_cps'])
    # TrainTestScript.test_loaders_check(train_dl_ft, test_dl, c_model, argv['global']['f'])

    print('\neffective_epsilons:')
    BalanceSgdScript.effective_epsilon_stats_dataloader(loader=train_dl_ft, p=p_model, c=c_model,
                                                        f=argv['global']['f'])
    BalanceSgdScript.effective_epsilon_stats_dataloader(loader=test_dl, p=p_model, c=c_model,
                                                        f=argv['global']['f'])

    # log_path = c_model.path.replace('.pt', '.txt')
    # if argv['global']['save_models'] and not os.path.exists(log_path):
    #     f = open(log_path, 'w')
    #     f.close()
    return


def main():
    """
    # coreset: compressed model
    # RITA: cifar10-vgg19
    # full model err 6.23 (Acc=93.77) #params=20,035,018
    # c    model err 6.02 (Acc=93.98) #params= 2,366,600 88.2%
    :return:
    """
    argv = get_argv()
    us.set_cuda_scope_and_seed(seed=argv['global']['seed'])
    train_dl_full, test_dl = get_dataloaders(ds_name=argv['ds']['name'], ds_root=argv['ds']['root'],
                                             bs_train=argv['full_model']['bs_train'], bs_test=argv['global']['bs_test'],
                                             shf_train=argv['full_model']['shuffle'],
                                             shf_test=argv['global']['shuffle_test'],
                                             data_limit=argv['ds']['data_limit'])

    # parent model
    print('\np_model:')
    p_model = VggCifar10(vgg_type=argv['full_model']['vgg_type'], path=argv['full_model']['path'])
    get_model(argv, model=p_model, train_dl=train_dl_full, test_dl=test_dl,
              opt_d=argv['full_model']['opt'], epochs=argv['full_model']['epochs'], lr_cps=argv['full_model']['lr_cps'])
    # log_path = p_model.path.replace('.pt', '.txt')
    # if argv['global']['save_models'] and not os.path.exists(log_path):
    #     f = open(log_path, 'w')
    #     f.close()
    # TrainTestScript.test_loaders_check(train_dl_full, test_dl, p_model, argv['global']['f'])  # todo remove com

    if argv['global']['mode'] == 'fs':
        main_vgg19_full_sweep(argv, p_model, test_dl)
    return


if __name__ == "__main__":
    # us.make_cuda_invisible()
    g_start_time = us.get_current_time()
    print("Python Version {}".format(sys.version))
    print("PyTorch Version {}".format(torch.__version__))
    print("Working directory {}".format(os.path.abspath(os.getcwd())))
    main()
    print('Total run time {}'.format(us.get_time_str(g_start_time)))
