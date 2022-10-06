from sklearn import datasets
import os
import pandas as pd
import numpy as np
from PIL import Image
# import h5py
import json
# from tensorflow import keras
from collections import defaultdict

import torch
import torch.utils.data as tu_data
import torchvision

from Dependencies import UtilsScript as utils
import wizzi_utils as wu

# found that these are the most used transforms for mnist and cifar10
DEFAULT_MNIST_TR = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

DEFAULT_CIFAR10_TR_TRAIN = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

DEFAULT_CIFAR10_TR_TEST = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


def get_model_net_data_loader(data_root: str, is_train: bool, points_per_sample: int = 2048, with_shapes: bool = False,
                              bs: int = 32, shuffle: bool = True, data_limit: int = 0,
                              ack: bool = True) -> tu_data.DataLoader:
    """
    root = '../Datasets'
    with_shapes = True
    train_dl = get_model_net_data_loader(data_root=root, is_train=True, with_shapes=with_shapes, bs=32, shuffle=True)
    test_dl = get_model_net_data_loader(data_root=root, is_train=False, with_shapes=with_shapes, bs=100, shuffle=True)
    """
    ds = get_model_net_data_set(data_root=data_root, is_train=is_train, points_per_sample=points_per_sample,
                                with_shapes=with_shapes, data_limit=data_limit, ack=ack)
    data_loader = tu_data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2)
    return data_loader


def get_model_net_data_set(data_root: str, is_train: bool, points_per_sample: int = 2048, with_shapes: bool = False,
                           data_limit: int = 0, ack: bool = True) -> tu_data.Dataset:
    """
    root = '../Datasets'
    with_shapes = True
    train_ds = get_model_net_data_set(data_root=root, is_train=True, with_shapes=with_shapes)
    test_ds = get_model_net_data_set(data_root=root, is_train=False, with_shapes=with_shapes)
    """
    data_set = ModelNetCls(data_root=data_root, is_train=is_train, points_per_sample=points_per_sample,
                           include_shapes=with_shapes, data_limit=data_limit, ack=ack)
    return data_set


def get_data_set(ds_name: str, data_root: str, is_train: bool, transform=None, download: bool = False,
                 data_limit: int = 0, ack: bool = True) -> torchvision.datasets:
    """
    if transform is None - sets default transform by my choice
    if data_limit > 0: ds = ds[:data_limit]
    e.g.
        ds_name = 'mnist' # 'cifar10'
        ds_root = '../Datasets'
        is_train = True
        train_ds = get_data_set(ds_name=ds_name, data_root=ds_root, is_train=is_train, transform=None,
                                download = False, data_limit= 0, ack=True)
    """
    if ds_name == 'cifar10':
        if transform is None:  # default cifar10 transforms
            transform = DEFAULT_CIFAR10_TR_TRAIN if is_train else DEFAULT_CIFAR10_TR_TEST
        data_set = torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=download, transform=transform)
    elif ds_name == 'mnist':
        if transform is None:  # default mnist transform
            transform = DEFAULT_MNIST_TR
        data_set = torchvision.datasets.MNIST(root=data_root, train=is_train, download=download, transform=transform)
    else:
        print('data_set not valid!')
        return
    if data_limit > 0:
        data_set.data = data_set.data[:data_limit]
        data_set.targets = data_set.targets[:data_limit]

    if ack:
        prefix = 'train' if is_train else 'test'
        print('get_data_set({}: {} dataset): {}'.format(ds_name, prefix, utils.data_set_size_to_str(data_set)))
    return data_set


def get_data_loader(ds_name: str, data_root: str, is_train: bool, bs: int, transform=None,
                    shuffle: bool = False, data_limit: int = 0, download: bool = False) -> tu_data.DataLoader:
    """
    e.g.
        ds_name = 'mnist' # 'cifar10'
        ds_root = '../Datasets'
        is_train = True
        train_loader = GetDataScript.get_data_loader(ds_name=argv['ds_name'], data_root=argv['ds_root'], is_train=True,
                                                 bs=64, shuffle=False, transform=None, data_limit=0)
    """
    data_set = get_data_set(ds_name, data_root, is_train, transform, download, data_limit, ack=False)
    data_loader = tu_data.DataLoader(data_set, batch_size=bs, shuffle=shuffle, num_workers=2)
    prefix = 'train' if is_train else 'test'
    print('get_data_loader({}: {} dataset): {}'.format(ds_name, prefix,
                                                       utils.data_set_size_to_str(data_loader.dataset)))
    return data_loader


def get_data_set_transformed(ds_name: str, data_root: str, is_train: bool, transform=None, download: bool = False,
                             data_limit: int = 0, ack=True) -> torchvision.datasets:
    """
    if transform is None - sets default transform by my choice
    if data_limit > 0: ds = ds[:data_limit]
    preprocess data once - notice we lose data original values (e.g. can't show image)
    loading is much slower !
    but training is much faster:
        Notice - to be clear - same exact output(on all epochs) for training on same device
        5 epochs training on LeNet_300_100:
            with dataloader(regular dataset that on each epoch get transformed):
                CPU: 64.139 seconds
                GPU: 55.497 seconds
            as dataset transformed:
                CPU: 25.102 seconds
                GPU: 17.318 seconds
    e.g.
        ds_name = 'mnist' # 'cifar10'
        ds_root = '../Datasets'
        is_train = True
        train_ds = get_data_set(ds_name=ds_name, data_root=ds_root, is_train=is_train, transform=None,
                                download = False, data_limit= 0, ack=True)
    """
    if ds_name == 'cifar10':
        data_set = torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=download, transform=None)

        if transform is None:  # default cifar10 transforms
            transform = DEFAULT_CIFAR10_TR_TRAIN if is_train else DEFAULT_CIFAR10_TR_TEST
        if data_limit > 0:
            data_set.data = data_set.data[:data_limit]
            data_set.targets = data_set.targets[:data_limit]

        X = torch.empty(size=(data_set.data.shape[0], 3, 32, 32))
        for i, x in enumerate(data_set.data):
            if is_train:
                x = Image.fromarray(x)
            x = transform(x)  # normalize and change shape from 32, 32, 3 to 3, 32, 32
            X[i] = x
        data_set.data = X

        data_set.targets = torch.as_tensor(data_set.targets).type(torch.int64)

    elif ds_name == 'mnist':
        data_set = torchvision.datasets.MNIST(root=data_root, train=is_train, download=download, transform=None)
        if transform is None:  # default mnist transform
            transform = DEFAULT_MNIST_TR
        if data_limit > 0:
            data_set.data = data_set.data[:data_limit]
            data_set.targets = data_set.targets[:data_limit]
        X = torch.empty(size=(data_set.data.shape[0], 1, 28, 28))
        for i, x in enumerate(data_set.data):
            x = transform(x.view(28, 28, 1).numpy())  # normalize and change shape from 28,28,1 to 1,28,28
            X[i] = x
        data_set.data = X
    else:
        print('data_set not valid!')
        return

    if ack:
        prefix = 'train' if is_train else 'test'
        print('get_data_set_transformed({}: {} dataset): {}'.format(ds_name, prefix,
                                                                    utils.data_set_size_to_str(data_set)))
    return data_set


def get_data_set_as_A(ds_name: str, data_root: str, is_train: bool = True, transform: bool = None,
                      flatten: bool = False, data_limit: int = 0) -> torch.Tensor:
    """
    :param ds_name: currently hard coded for Mnist (see assert)
    :param data_root: dataset location
    :param is_train: train/test
    :param transform: if none - uses default set by me
    :param flatten: (60K/10K, 1, 28, 28) or (60K/10K, 784)
    :param data_limit: less data (0 for all data)
    :return:
        if flatten: returns A=X|y. to do so, needed to cast y to X type. when you run validation:
            for i in range(0, A.shape[0], bs):
                batchA = A[i:i + bs]
                images, labels = de_augment_torch(batchA)
                print(var_to_string(batchA, title='batchA', with_data=False))
                print(var_to_string(images, title='images', with_data=False))
                print(var_to_string(labels, title='labels', with_data=True))

                images = images.view(batchA.shape[0], 1, 28, 28)
                print(var_to_string(images, title='images', with_data=False))

                labels = labels.type(torch.int64).view(labels.shape[0])
                print(var_to_string(labels, title='labels', with_data=True))

                images, labels = add_cuda(images), add_cuda(labels)
                labels_predictions_vectors = q(images)

                sum_batch_loss += f(labels_predictions_vectors, labels, reduction='sum').item()
                predictions = labels_predictions_vectors.argmax(dim=1, keepdim=True)
                correct += predictions.eq(labels.view_as(predictions)).sum().item()
        else: returns dict with keys X and y. when you run validation:
            for i in range(0, A['X'].shape[0], bs):
                images, labels = A['X'][i:i + bs], A['y'][i:i + bs]
                print(var_to_string(images, title='images', with_data=False))
                print(var_to_string(labels, title='labels', with_data=True))
                images, labels = add_cuda(images), add_cuda(labels)
                labels_predictions_vectors = q(images)

                sum_batch_loss += f(labels_predictions_vectors, labels, reduction='sum').item()
                predictions = labels_predictions_vectors.argmax(dim=1, keepdim=True)
                correct += predictions.eq(labels.view_as(predictions)).sum().item()

    """
    assert ds_name == 'mnist', 'implemented only for MNIST. need to alter for other datasets'
    if ds_name == 'mnist':
        if transform is None:  # default mnist transform
            transform = DEFAULT_MNIST_TR

    data_set = get_data_set(ds_name, data_root, is_train, data_limit=data_limit)
    # print('Original dataset info:')
    # print(var_to_string(data_set.data, title='\tdata_set.data'))
    # print(var_to_string(data_set.targets, title='\tdata_set.targets'))

    # transform is activated only when using the iterator of the data loader, so we want to it now
    X_temp = data_set.data
    # print(var_to_string(X_temp, title='X before transform'))
    X = torch.empty(size=(X_temp.shape[0], 1, 28, 28))
    for i, x in enumerate(X_temp):
        x_new = x.view(28, 28, 1)
        x_new = transform(utils.torch_to_numpy(x_new))  # normalize and change shape from a,b,c to c,a,b
        # print(var_to_string(x_new, title='x_new'))
        X[i] = x_new
    # print(var_to_string(X, title='X after transform'))

    if flatten:
        X = torch.flatten(X, start_dim=1)
        # X = torch.flatten(X, start_dim=1).type(X.dtype)
        # print(var_to_string(X, title='X flat'))

    X = utils.add_cuda(X)
    if flatten:
        y = utils.add_cuda(data_set.targets.type(X.dtype))  # notice the cast to X.type to concat X|y -> A
    else:
        y = utils.add_cuda(data_set.targets)
    # print('Our data info:')
    # print(var_to_string(X, title='\tX'))
    # print(var_to_string(y, title='\ty'))

    # if False:  # change to True if you want to see that our data is the same as if you called get data loader
    #     print('Comparing our data to data loader:')
    #     fake_bs = 64
    #     data_loader = get_data_loader(ds_name, data_root, is_train, bs=fake_bs, shuffle=False, transform=transform,
    #                                   data_limit=data_limit)
    #     images, labels = iter(data_loader).next()  # get first batch
    #     images, labels = utils.add_cuda(images), utils.add_cuda(labels)
    #
    #     print(utils.var_to_string(images, title='\tdata loader images info', with_data=False))
    #     print(utils.var_to_string(X[:fake_bs], title='\tout data    images info', with_data=False))
    #
    #     print(utils.var_to_string(labels, title='\tdata loader labels info', with_data=False))
    #     print(utils.var_to_string(y[:fake_bs], title='\tout data y  images info', with_data=False))
    #
    #     x0_dl = images[0].flatten() if flatten else images[0]
    #     x0_us = X[0]
    #     print('\tx0 data loader == x0 our data ? {}'.format('True' if torch.all(torch.eq(x0_dl, x0_us)) else 'False'))
    #
    #     y0_dl = labels[0].flatten() if flatten else labels[0]
    #     y0_us = y[0]
    #     print('\ty0 data loader == y0 our data ? {}'.format('True' if y0_dl.item() == y0_us.item() else 'False'))

    if flatten:
        A = utils.augment_x_y_torch(X, y)
        print(utils.var_to_string(A, title='\tA'))
    else:
        A = {'X': X, 'y': y}
        print(utils.var_to_string(X, title='\tX'))
        print(utils.var_to_string(y, title='\ty'))
    return A


def create_data_loader_from_array(X: torch.Tensor, y: torch.Tensor, bs: int = 64,
                                  shuffle: bool = False) -> tu_data.DataLoader:
    # tensor_X = torch.stack([i for i in X])  # transform to torch tensors
    # tensor_Y = torch.empty(size=y.shape, dtype=y.dtype)
    # for i, y_i in enumerate(y):
    #     tensor_Y[i] = torch.Tensor([y_i])
    my_dataset = tu_data.TensorDataset(X, y)
    my_data_loader = tu_data.DataLoader(my_dataset, batch_size=bs, shuffle=shuffle, num_workers=0)

    # if False:  # change to True if you want to see that our data loader is the same as iterating X,y manually
    #     print('Comparing our data loader to origin X and y:')
    #     images, labels = iter(my_data_loader).next()  # get first batch
    #     images, labels = utils.add_cuda(images), utils.add_cuda(labels)
    #
    #     print(utils.var_to_string(images, title='\tour data loader images info', with_data=False))
    #     print(utils.var_to_string(X[:bs], title='\torigin data images info    ', with_data=False))
    #
    #     print(utils.var_to_string(labels, title='\tour data loader labels info', with_data=False))
    #     print(utils.var_to_string(y[:bs], title='\torigin data y images info  ', with_data=False))
    #
    #     if shuffle:
    #         print('\tdata loader was shuffled. cant compare value of data, only sizes and types')
    #     else:
    #         x0_dl = images[0]
    #         x0_orig = X[0]
    #         msg = '\tx0 our data loader == x0 origin data ? {}'
    #         print(msg.format('True' if torch.all(torch.eq(x0_dl, x0_orig)) else 'False'))
    #
    #         msg = '\ty0({}) out data loader == y0({}) origin data ? {}'
    #         y0_dl = labels[0].item()
    #         y0_orig = y[0].item()
    #         print(msg.format(y0_dl, y0_orig, 'True' if y0_dl == y0_orig else 'False'))

    print('create_data_loader_from_array: data size {}'.format(utils.data_set_size_to_str(my_dataset)))
    return my_data_loader


def get_data_X_y_sklearn(ds_name: str, data_root: str = None) -> (np.array, np.array):
    if ds_name == 'diabetes':
        X, y = datasets.load_diabetes(return_X_y=True)  # |X| = 442x10 samples
    elif ds_name == 'breast_cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)  # |X| = 569x30 samples
    elif ds_name == 'covtype':
        X, y = datasets.fetch_covtype(return_X_y=True, data_home=data_root)  # |X|=581012x54 samples,7 classes
    else:
        print('data_set not valid!')
        return
    return X, y


def get_data_A_sklearn(ds_name: str, data_root: str = None) -> np.array:
    X, y = get_data_X_y_sklearn(ds_name, data_root)
    A = utils.augment_x_y_numpy(X, y)
    return A


def get_data_A_htru(ds_name: str, data_root: str = None) -> np.array:
    """
    # (i) HTRU [19] — 17; 898 radio emissions of the Pulsar star each consisting of 9 features.
    """
    from sklearn.preprocessing import StandardScaler

    abs_folder = os.path.abspath('{}/{}/{}.csv'.format(data_root, ds_name, ds_name))
    df = pd.read_csv(abs_folder)
    X = df.iloc[:, 0:8]
    y_pd = df.iloc[:, 8]

    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    y = np.zeros(shape=(X.shape[0]))
    for i, y_i_pd in enumerate(y_pd):
        y[i] = y_i_pd
    utils.count_keys(y)
    A = utils.augment_x_y_numpy(X, y)
    return A


def get_data_A_3dRoad(ds_name: str, data_root: str, normalize: bool = True) -> np.array:
    """
    original TXT specs:
    https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29

    Abstract: 3D road network with highly accurate elevation information (+-20cm) from Denmark used in eco-routing and
    fuel/Co2-estimation routing algorithms.

    This dataset was constructed by adding elevation information to a 2D road network in North Jutland, Denmark
    (covering a region of 185 x 135 km^2). Elevation values where extracted from a publicly available massive Laser
    Scan Point Cloud for Denmark (available at : [Web Link] (Bottom-most dataset)). This 3D road network was eventually
    used for benchmarking various fuel and CO2 estimation algorithms. This dataset can be used by any applications that
    require to know very accurate elevation information of a road network to perform more accurate routing for
    eco-routing, cyclist routes etc. For the data mining and machine learning community, this dataset can be used as
    'ground-truth' validation in spatial mining techniques and satellite image processing. It has no class labels, but
    can be used in unsupervised learning and regression to guess some missing elevation information for some points on
    the road. The work was supported by the Reduction project that is funded by the European Comission as
    FP7-ICT-2011-7 STREP project number 288254.

    Attribute Information:
        1. OSM_ID: OpenStreetMap ID for each road segment or edge in the graph.
        2. LONGITUDE: Web Mercaptor (Google format) longitude
        3. LATITUDE: Web Mercaptor (Google format) latitude
        4. ALTITUDE: Height in meters.

    Note: OSM_ID is the ID assigned by OpenStreetMaps ([Web Link]) to the road segments. Each (long,lat,altitude) point
    n a road segment (with unique OSM ID) is sorted in the same order as they appear on the road. So a 3D-polyline can
    be drawn by joining points of each row for each OSM_ID road segment.

    Data Set Characteristics: Sequential, Text
    Number of Instances: 434874
    Area: Computer
    Attribute Characteristics: Real
    Number of Attributes: 4
    Date Donated 2013-04-16
    Associated Tasks: Regression, Clustering
    Missing Values? N/A
    Number of Web Hits: 196959

    parsing specs: https://notebook.community/AlJohri/DAT-DC-12/notebooks/clustering_spatial

    -----------------------------------------------------
    3droad.mat specs:
    https://notebook.community/jrg365/gpytorch/examples/05_Scalable_GP_Regression_Multidimensional/SVDKL_Regression_GridInterp_CUDA
    For this example notebook, we'll be using the 3droad UCI dataset used in the paper. Running the next cell downloads
    a copy of the dataset that has already been scaled and normalized appropriately. For this notebook, we'll simply be
    splitting the data using the first 80% of the data as training and the last 20% as testing.
    Note: Running the next cell will attempt to download a ~136 MB file to the current directory.
    link to download mat 'https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1'

        from scipy.io import loadmat  # consider removing col 1
        mat_path = os.path.join(data_root, ds_name, '3droad.mat')
        data = loadmat(mat_path)['data']
        # data = torch.Tensor(loadmat('3droad.mat')['data'])
        print(utils.var_to_string(data, '3droad', with_data=False))
        print(utils.var_to_string(data[0], '3droad[0]', with_data=True))
        print('data[0][0] = {}, type={}'.format(data[0][0], type(data[0][0])))
        X = data[:, :-1]
        X = X - X.min(0)[0]
        X = 2 * (X / X.max(0)[0]) - 1
        y = data[:, -1]
        print(utils.var_to_string(X, 'X', with_data=False))
        print(utils.var_to_string(y, 'y', with_data=False))
        A = utils.augment_x_y_numpy(X, y)
        print(utils.var_to_string(A, 'A', with_data=False))
        print(utils.var_to_string(A[0], 'A[0]', with_data=True))
        print('A[0][0] = {}, type={}'.format(A[0][0], type(A[0][0])))
        return A

    @param ds_name:
    @param data_root:
    @param normalize:
    @return:
    """
    txt_path = os.path.join(data_root, ds_name, '3D_spatial_network.txt')
    roads = pd.read_csv(txt_path, header=None, names=['osm', 'lat', 'lon', 'alt'])

    data_pd = roads.drop(['osm'], axis=1)
    # data_pd = roads.drop(['osm'], axis=1).sample(100000)
    # print(data_pd.head())
    if normalize:
        from sklearn.preprocessing import StandardScaler
        X_pd = data_pd.iloc[:, 0:2]
        y = data_pd.iloc[:, 2].to_numpy()

        sc = StandardScaler()  # Feature Scaling
        X = sc.fit_transform(X_pd)
        # X = X_pd.to_numpy()
        # X = (X - X.mean(axis=0)) / X.std(axis=0)
        data = utils.augment_x_y_numpy(X, y)
    else:
        data = data_pd.to_numpy()
    # print(wu.to_str(data, '\t3dRoad_data', data_chars=100, float_precision=2))
    return data


def get_data_A_htru_v2(ds_name: str, data_root: str = None, preprocess_data: bool = True) -> np.array:
    """
    Murad's get data version
    https://archive.ics.uci.edu/ml/datasets/HTRU2
    @return:
    """
    from sklearn.preprocessing import StandardScaler

    abs_folder = os.path.abspath('{}/{}/{}.csv'.format(data_root, ds_name, ds_name))
    df = pd.read_csv(abs_folder)
    X = df.iloc[:, 0:8]
    y_pd = df.iloc[:, 8]

    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    y = np.zeros(shape=(X.shape[0]))
    for i, y_i_pd in enumerate(y_pd):
        y[i] = y_i_pd
    utils.count_keys(y)
    P = utils.augment_x_y_numpy(X, y)
    P = P.astype('float64')

    y = P[:, -1]
    min_value = np.min(y)
    max_value = np.max(y)
    y[np.where(y == min_value)[0]] = -1
    y[np.where(y == max_value)[0]] = 1
    utils.count_keys(y)

    if preprocess_data:
        norms = np.sqrt(np.sum(P[:, :-1] ** 2, axis=1))
        max_norm = np.max(norms)
        if max_norm > 1:
            P[:, :-1] /= max_norm
        P[:, :-1] = StandardScaler().fit_transform(X=P[:, :-1], y=P[:, -1])

        P[:, -1] = y
        # P = Normalizer(norm='max').fit_transform(X=P[:, :-1], y=P[:, -1])
        # P = np.hstack((P, y[:, np.newaxis]))
    return P


def get_data_A_parkinson(ds_name: str, data_root: str = None) -> np.array:
    # https://colab.research.google.com/drive/1hJb5DCsCuWxSXnJF7noUr8ZTqwLRTWIe#scrollTo=WR8zKNnw5dBk

    def normalize(x):
        return (x - ds_X_stats['mean']) / ds_X_stats['std']

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
    # first download and save if not exists
    url_suffix = url.split('/')[-1]  # for file name
    abs_folder = os.path.abspath('{}/{}'.format(data_root, ds_name))
    print(abs_folder)
    if not os.path.exists(abs_folder):
        os.mkdir(abs_folder)
    full_path = os.path.abspath('{}/{}'.format(abs_folder, url_suffix))
    # noinspection PyUnresolvedReferences
    dataset = keras.utils.get_file(full_path, url)

    # read csv
    print('reading file {}'.format(full_path))
    ds = pd.read_csv(dataset, na_values="?", comment='\t', sep=",", skipinitialspace=True)
    # print(ds.head())
    # print(ds.shape)
    # print(ds.columns)
    # print(parkinsons.isna().sum())  # Check if there are any NAs in the rows
    ds = ds.drop(['subject#'], axis=1)

    # train_dataset = parkinsons2.sample(frac=0.8, random_state=0)
    # test_dataset = parkinsons2.drop(train_dataset.index)

    # pop labels
    # label set 1
    ds_y_1 = ds.pop('motor_UPDRS')
    ds_y_1 = ds_y_1.to_numpy()
    # label set 2 - ignored for now
    ds.pop('total_UPDRS')
    # ds_y_2 = ds.pop('total_UPDRS')
    # ds_y_2 = ds_y_2.to_numpy()

    # rest is X
    ds_X = ds
    ds_X_stats = ds_X.describe().transpose()
    ds_X_normed = normalize(ds_X)
    ds_X_normed = ds_X_normed.to_numpy()
    # print(ds_X_stats)

    A = utils.augment_x_y_numpy(ds_X_normed, ds_y_1)
    return A


def get_data_A_by_name(ds_name: str, data_root: str = None, out_torch: bool = True, out_torch_double=True,
                       data_limit: int = 0, two_d: bool = False) -> torch.Tensor:
    """
    e.g.
        A = get_data_A_by_name(ds_name='diabetes')
        A = get_data_A_by_name(ds_name='breast_cancer')
        A = get_data_A_by_name(ds_name='covtype', data_root='../Datasets/')
        A = get_data_A_by_name(ds_name='parkinsons_updrs', data_root='../Datasets/')

    use two_d=True for debug (gets the second feature only - base on diabetes example)
    use data_limit=True for debug (gets the first data_limit features)
    use split_A(A) if train and test split needed
    """

    A = None
    if ds_name in ['diabetes', 'breast_cancer', 'covtype']:
        A = get_data_A_sklearn(ds_name, data_root)
    elif ds_name in ['parkinsons_updrs']:
        A = get_data_A_parkinson(ds_name, data_root)
    elif ds_name in ['HTRU_2']:
        # A = get_data_A_htru(ds_name, data_root)
        A = get_data_A_htru_v2(ds_name, data_root)
    elif ds_name in ['3DRoad']:
        A = get_data_A_3dRoad(ds_name, data_root, normalize=False)
    if two_d:
        if isinstance(A, torch.Tensor):
            X, y = wu.tt.de_augment_torch(A)
            X = X[:, 2]  # usually for debugging
            A = wu.tt.augment_x_y_torch(X, y)
        elif isinstance(A, np.ndarray):
            X, y = wu.de_augment_numpy(A)
            X = X[:, 2]  # usually for debugging
            A = wu.augment_x_y_numpy(X, y)
    if out_torch:
        A = wu.tt.numpy_to_torch(A, to_double=out_torch_double)
    if data_limit > 0:
        A = A[:data_limit]

    print(wu.tt.to_str(A, title='\t{}_data'.format(ds_name), chars=100, fp=2))
    return A


class ModelNetCls(tu_data.Dataset):
    # Transform Classes
    class PointCloudToTensor(object):
        def __call__(self, points):
            return torch.from_numpy(points).float()

    class PointCloudRotate(object):
        def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
            self.axis = axis

        @staticmethod
        def angle_axis(angle, axis):
            # type: (float, np.ndarray) -> float
            """
            :param angle: float Angle to rotate by
            :param axis: np.ndarray Axis to rotate about
            :return: torch.Tensor 3x3 rotation matrix
            Returns a 4x4 rotation matrix that performs a rotation around axis by angle
            """
            u = axis / np.linalg.norm(axis)
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                       [u[2], 0.0, -u[0]],
                                       [-u[1], u[0], 0.0]])
            R = torch.from_numpy(
                cos_val * np.eye(3)
                + sin_val * cross_prod_mat
                + (1.0 - cos_val) * np.outer(u, u))
            return R.float()

        def __call__(self, points):
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = self.angle_axis(rotation_angle, self.axis)

            normals = points.size(1) > 3
            if not normals:
                # noinspection PyUnresolvedReferences
                return torch.matmul(points, rotation_matrix.t())
            else:
                pc_xyz = points[:, 0:3]
                pc_normals = points[:, 3:]
                # noinspection PyUnresolvedReferences
                points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
                # noinspection PyUnresolvedReferences
                points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

                return points

    class PointCloudScale(object):
        def __init__(self, lo=0.8, hi=1.25):
            self.lo, self.hi = lo, hi

        def __call__(self, points):
            scalar = np.random.uniform(self.lo, self.hi)
            points[:, 0:3] *= scalar
            return points

    class PointCloudTranslate(object):
        def __init__(self, translate_range=0.1):
            self.translate_range = translate_range

        def __call__(self, points):
            translation = np.random.uniform(-self.translate_range, self.translate_range)
            points[:, 0:3] += translation
            return points

    class PointCloudJitter(object):
        def __init__(self, std=0.01, clip=0.05):
            self.std, self.clip = std, clip

        def __call__(self, points):
            jittered_data = (
                points.new(points.size(0), 3).normal_(mean=0.0, std=self.std).clamp_(-self.clip, self.clip)
            )
            points[:, 0:3] += jittered_data
            return points

    @staticmethod
    def _get_data_files(headers_file: str) -> list:
        headers_file_obg, files_suffix = None, []
        try:
            headers_file_obg = open(headers_file, 'r')
            files_suffix = [l.rstrip().split('/')[-1] for l in headers_file_obg]
        finally:
            headers_file_obg.close()
        return files_suffix

    @staticmethod
    def _load_data_file(full_path: str) -> (np.array, np.array):
        f = h5py.File(full_path)
        data = f["data"][:]
        label = f["label"][:]
        f.close()
        return data, label

    def __init__(self, data_root: str, is_train: bool, points_per_sample: int = 2048, include_shapes: bool = False,
                 data_limit: int = 0, ack: bool = True):
        """
        :param data_root: where directory modelnet40_ply_hdf5_2048/ exists
        :param is_train: train\test set
        :param points_per_sample: max 2048. 3d points per sample
        :param include_shapes: if True, get item returns a trio of X,y,exact label. else X,y
        :param data_limit:
        :param ack:
        """
        super().__init__()
        CONST_FOLDER_NAME = "modelnet40_ply_hdf5_2048"
        CONST_FILES_DIR_PATH = os.path.abspath(os.path.join(data_root, CONST_FOLDER_NAME))
        assert os.path.exists(CONST_FILES_DIR_PATH), '{} doesnt exits'.format(CONST_FILES_DIR_PATH)

        CONST_TRANSFORMS = torchvision.transforms.Compose([
            self.PointCloudToTensor(),
            self.PointCloudRotate(axis=np.array([1, 0, 0])),
            self.PointCloudScale(),
            self.PointCloudTranslate(),
            self.PointCloudJitter()])

        self.transforms = CONST_TRANSFORMS if is_train else None  # if test no trans so we get the same results

        # curl doesnt work yet - download form zip and extract to Datasets folder under the name CONST_FOLDER_NAME
        # CONST_ZIP_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        # if download and not os.path.exists(self.data_dir):
        #     zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
        #     # print(zipfile, self.url)
        #     subprocess.check_call(
        #         shlex.split("curl {} -o {}".format(self.url, zipfile))
        #     )
        #     subprocess.check_call(
        #         shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
        #     )
        #     subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        files_headers_file = "train_files.txt" if is_train else "test_files.txt"
        files = self._get_data_files(headers_file='{}/{}'.format(CONST_FILES_DIR_PATH, files_headers_file))
        point_list, label_list = [], []
        for file_name in files:
            points, labels = self._load_data_file(full_path='{}/{}'.format(CONST_FILES_DIR_PATH, file_name))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

        if data_limit > 0:
            self.points = self.points[:data_limit]
            self.labels = self.labels[:data_limit]

        self.num_points = min(self.points.shape[1], points_per_sample)
        self.points = self.points[:, :self.num_points, :]

        # if np.ndim(self.labels) == 1:
        #     self.labels = np.expand_dims(self.labels, axis=1)

        self.shapes = []
        self.include_shapes = include_shapes

        if self.include_shapes:
            N = len(files)
            T = "train" if is_train else "test"
            for n in range(N):
                j_name = '{}/ply_data_{}_{}_id2file.json'.format(CONST_FILES_DIR_PATH, T, n)
                with open(j_name, "r") as f:
                    shapes = json.load(f)
                    self.shapes += shapes
                    f.close()
            if data_limit > 0:
                self.shapes = self.shapes[:data_limit]

        self.labels_ind_to_str = {}
        labels_text_file_obj, files_suffix = None, []
        try:
            labels_text_file_obj = open('{}/{}'.format(CONST_FILES_DIR_PATH, 'shape_names.txt'), 'r')
            for class_ind, line in enumerate(labels_text_file_obj):
                self.labels_ind_to_str[class_ind] = line.rstrip()
        finally:
            labels_text_file_obj.close()

        if ack:
            T = "train" if is_train else "test"
            print('Loading modelnet40_ply_hdf5_2048 {} data({} blobs per cloud):'.format(T, self.num_points))
            print('\t|data_set|={}'.format(len(self)))
            print('\tclasses headers({})={}'.format(len(self.labels_ind_to_str), self.labels_ind_to_str))

            # first_sample, first_label = self[0][0], self[0][1]
            # print(utils.var_to_string(first_sample, '\tfirst_sample', with_data=True))
            # print(utils.var_to_string(first_label,
            #                           '\tfirst_label({})'.format(self.labels_ind_to_str[first_label.item()]),
            #                           with_data=True))
            if self.include_shapes:
                counter = defaultdict(int)
                for shape in self.shapes:
                    label_base = shape.split('/')[0]
                    label_ind = -1
                    for ind, text in self.labels_ind_to_str.items():
                        if text == label_base:
                            label_ind = ind
                    counter[label_ind] += 1
                print('\tclasses count={}'.format(counter))
                # print('\tfirst shape {}'.format(self[0][2]))
        return

    def __getitem__(self, idx):
        # removed mixing the sample - just for compression
        current_points = self.points[idx].copy()  # original - replace line with bottom 2
        # perm = np.random.permutation(self.num_points)
        # current_points = self.points[idx, perm].copy()

        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)[0]  # [0] to replace flatten

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        current_points = np.transpose(current_points, (1, 0))  # shape=(|blobs|,3) -> shape=(3,|blobs|)

        if self.include_shapes:
            shape = self.shapes[idx]
            return current_points, label, shape
        return current_points, label

    def __len__(self):
        return self.points.shape[0]


# # # transform options:
#
# Default: transform = transforms.Compose([transforms.ToTensor(), ])
# MNIST: transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# CIFAR10:
# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# CIFAR10 train:
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
# CIFAR10 test:
# transform_test = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

def main():
    utils.make_cuda_invisible()
    utils.set_cuda_scope_and_seed(seed=42)
    root = '../Datasets'
    with_shapes = True
    # train_ds = get_model_net_data_set(data_root=root, is_train=True, with_shapes=with_shapes)
    # test_ds = get_model_net_data_set(data_root=root, is_train=False, with_shapes=with_shapes)
    train_dl = get_model_net_data_loader(data_root=root, is_train=False, points_per_sample=500, with_shapes=with_shapes,
                                         bs=1, shuffle=True, data_limit=32)

    # test_dl = get_model_net_data_loader(data_root=root, is_train=False, with_shapes=with_shapes, bs=100, shuffle=True)
    print('running on loader:')
    # noinspection PyUnresolvedReferences
    labels_ind_str_dict = train_dl.dataset.labels_ind_to_str
    with torch.no_grad():
        for batchX, batch_y, batchShapes in train_dl:
            if batch_y[0].item() == 7:
                print(utils.var_to_string(batchX, '\tbatchX', with_data=False))
                print(utils.var_to_string(batch_y, '\tbatch_y', with_data=True))
                print('\tbatchShapes {}'.format(batchShapes))
                print('\tfirst label={}({})'.format(batch_y[0], labels_ind_str_dict[batch_y[0].item()]))
                utils.plot3d_scatter(batchX[0], title='3dplot', label=labels_ind_str_dict[batch_y[0].item()],
                                     marker='o')
                exit(22)
            # batchX, batch_y = utils.add_cuda(batchX), utils.add_cuda(batch_y)
    return


if __name__ == "__main__":
    # get_data_A_3dRoad(ds_name='3dRoad', data_root='../Datasets')
    # ds_name: str, data_root: str = None, out_torch: bool = True, out_torch_double = True,
    # data_limit: int = 0, two_d: bool = False
    g_A = get_data_A_by_name(ds_name='3DRoad', data_root='../Datasets')
