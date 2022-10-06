# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn
import torch.nn.functional as func
from Dependencies.models.BaseModelScript import BaseModel
from Dependencies import UtilsScript
from Dependencies import GetDataScript
from Dependencies import TrainTestScript


class LeNetMnist(BaseModel):
    """
    p__LeNetMnist_300_100(40 epochs Adam optimizer):
        train loss=0.021430, accuracy=59571/60000 (99.28%)
        test  loss=0.091702, accuracy=9793/10000 (97.93%)
    if fc1==300 and fc2==100: 266,610 params
    """

    def __init__(self, path, fc1=300, fc2=100):
        super().__init__(path)
        self.fc1 = nn.Linear(784, fc1)  # 28^2 to fc1. if fc1==300: 235,500 params
        self.fc2 = nn.Linear(fc1, fc2)  # fc1 to fc2. if fc1==300 and fc2==100: 30,100 params
        self.fc3 = nn.Linear(fc2, 10)  # fc2 to 10 classes if fc2==100: 1,010 params
        self.init(full_model_params=266610)
        return

    def create_mapping(self):
        # create layers mapping:
        mapping_list, layer_list = [], []
        for name, param in self.named_parameters():
            # print(name, param.shape)
            layer_list.append(name)
            if len(layer_list) == 2:  # in LeNet - each 2 are an fc layer
                mapping_list.append(layer_list)
                layer_list = []
        return mapping_list

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        h_relu1 = func.relu(self.fc1(x))
        h_relu2 = func.relu(self.fc2(h_relu1))
        logits = self.fc3(h_relu2)
        return func.log_softmax(logits, dim=1)


def mnist_lenet_example():
    UtilsScript.make_cuda_invisible()
    UtilsScript.set_cuda_scope_and_seed(seed=42)

    model = LeNetMnist(path='./p_LeNetMnist_300_100.pt', fc1=200, fc2=100)
    UtilsScript.model_params_print(model)
    print(UtilsScript.model_summary_to_string(model=model, input_size=(1, 28, 28), batch_size=-1))

    # test model (it's untrained - random guess is expected)
    root = '../../Datasets'
    train_ds = GetDataScript.get_data_set_transformed(ds_name='mnist', data_root=root, is_train=True, transform=None,
                                                      data_limit=0, download=False, ack=True)
    test_ds = GetDataScript.get_data_set_transformed(ds_name='mnist', data_root=root, is_train=False, transform=None,
                                                     data_limit=0, download=False, ack=True)
    test_bs = 1000

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    TrainTestScript.train_using_datasets(train_ds, test_ds, model, opt, func.cross_entropy, train_bs=64,
                                         test_bs=test_bs, epochs=10, use_acc=False, save_models=False, shuffle=False)

    TrainTestScript.test_using_dataset(test_ds, model, func.cross_entropy, bs=test_bs, print_res=True)

    # output:
    # Training p_LeNetMnist_300_100:
    # 	train_ds: |X|=(60000, 1, 28, 28), |y|=(60000,)
    # 	test_ds: |X|=(10000, 1, 28, 28), |y|=(10000,)
    # 	saving on each loss improvment
    # 	epochs=10, bs_train=64, #trainBatches=938, bs_test=1000, #testBatches=10
    # 	optimizer = Adam (Parameter Group 0 amsgrad: False betas: (0.9, 0.999) eps: 1e-08 lr: 0.001 weight_decay: 0.0001)
    # ...
    # ...
    # ...
    # Done training by loss    : On epoch 9: loss=0.087244, accuracy = 97.68%
    # If trained by accuracy   : On epoch 8: loss=0.091319, accuracy = 97.68%
    # loss=0.133171, accuracy=9663/10000 (96.63%)  -> result on model post epoch 10 (you should load best model)
    #
    # Process finished with exit code 0

    return


def speed_test():
    """
    Created p_LeNetMnist_300_100
    params=266,610, path=D:\workspace\2019SGD\Dependencies\models\p_LeNetMnist_300_100.pt
    p_LeNetMnist_300_100 is fully trainable
    p_LeNetMnist_300_100_clone is fully frozen
    dataloaders:
    working on CPU
    get_data_loader(mnist: train dataset): |X|=(60000, 1, 28, 28), |y|=(60000,)
    get_data_loader(mnist: test dataset): |X|=(10000, 1, 28, 28), |y|=(10000,)
        loading time Process time: 6.465 seconds
    Training p_LeNetMnist_300_100:
        Train data: |X|=(60000, 1, 28, 28), |y|=(60000,)
        Test data: |X|=(10000, 1, 28, 28), |y|=(10000,)
        saving on each loss improvment
        epochs=5, bs_train=64, #trainBatches=938, bs_test=1000, #testBatches=10
        opt = Adam (Parameter Group 0 amsgrad: False betas: (0.9, 0.999) eps: 1e-08 lr: 0.001 weight_decay: 0.0001)
        epoch [  1/  5] train loss:0.241974,test loss=0.133123, accuracy=9590/10000 (95.90%), time so far 00:00:20 New
        epoch [  2/  5] train loss:0.103134,test loss=0.098165, accuracy=9665/10000 (96.65%), time so far 00:00:34 New
        epoch [  3/  5] train loss:0.069771,test loss=0.088453, accuracy=9743/10000 (97.43%), time so far 00:00:48 New
        epoch [  4/  5] train loss:0.052205,test loss=0.177893, accuracy=9528/10000 (95.28%), time so far 00:01:02
        epoch [  5/  5] train loss:0.043862,test loss=0.105041, accuracy=9720/10000 (97.20%), time so far 00:01:16
    Done training by loss    : On epoch 3: loss=0.088453, accuracy = 97.43%
    If trained by accuracy   : On epoch 3: loss=0.088453, accuracy = 97.43%
        training time Process time: 76.228 seconds
    p_LeNetMnist_300_100 is fully frozen
    p_LeNetMnist_300_100_clone is fully trainable
    datasets:
    working on CPU
    get_data_set_transformed(mnist: train dataset): |X|=(60000, 1, 28, 28), |y|=(60000,)
    get_data_set_transformed(mnist: test dataset): |X|=(10000, 1, 28, 28), |y|=(10000,)
        loading time Process time: 6.677 seconds
    Training p_LeNetMnist_300_100_clone:
        train_ds: |X|=(60000, 1, 28, 28), |y|=(60000,)
        test_ds: |X|=(10000, 1, 28, 28), |y|=(10000,)
        saving on each loss improvment
        epochs=5, bs_train=64, #trainBatches=938, bs_test=1000, #testBatches=10
        opt = Adam (Parameter Group 0 amsgrad: False betas: (0.9, 0.999) eps: 1e-08 lr: 0.001 weight_decay: 0.0001)
        epoch [  1/  5] train loss:0.241974,test loss=0.133123, accuracy=9590/10000 (95.90%), time so far 00:00:04 New
        epoch [  2/  5] train loss:0.103134,test loss=0.098165, accuracy=9665/10000 (96.65%), time so far 00:00:08 New
        epoch [  3/  5] train loss:0.069771,test loss=0.088453, accuracy=9743/10000 (97.43%), time so far 00:00:12 New
        epoch [  4/  5] train loss:0.052205,test loss=0.177893, accuracy=9528/10000 (95.28%), time so far 00:00:17
        epoch [  5/  5] train loss:0.043862,test loss=0.105041, accuracy=9720/10000 (97.20%), time so far 00:00:21
    Done training by loss    : On epoch 3: loss=0.088453, accuracy = 97.43%
    If trained by accuracy   : On epoch 3: loss=0.088453, accuracy = 97.43%
        training time Process time: 21.470 seconds
    :return:
    """
    UtilsScript.make_cuda_invisible()
    UtilsScript.set_cuda_scope_and_seed(seed=42)
    m1 = LeNetMnist(path='./p1_LeNetMnist_300_100.pt', fc1=300, fc2=100)
    m2 = UtilsScript.clone_model(m1)
    m2.set_new_path('./p2_LeNetMnist_300_100.pt')
    ADAM_OPT_V2 = {
        'name': 'ADAM',
        'lr': 0.001,
        'weight_decay': 0.0001
    }
    TrainTestScript.speed_test_datasets_vs_dataloaders(m1=m1, m2=m2, f=func.cross_entropy, ds_name='mnist',
                                                       ds_root='../../Datasets', seed=42, bs_train=64, bs_test=1000,
                                                       data_limit=0, epochs=2, opt_d=ADAM_OPT_V2, b_interval=-1)
    return


def sim_forward():
    # simulate forward
    model = LeNetMnist(path='./p_LeNetMnist_300_100.pt', fc1=300, fc2=100)
    batch_x = UtilsScript.torch_normal(shape=(1000, 1, 28, 28), miu=0, std=10)
    y_pred = model(batch_x)
    print(UtilsScript.var_to_string(batch_x, 'batch_x', with_data=False))
    print(UtilsScript.var_to_string(y_pred, 'y_pred', with_data=False))
    return


def temp():
    _ = LeNetMnist(path='./p_LeNetMnist_300_100.pt', fc1=300, fc2=100)
    return


if __name__ == '__main__':
    # mnist_lenet_example()
    # speed_test()
    sim_forward()
    # temp()

