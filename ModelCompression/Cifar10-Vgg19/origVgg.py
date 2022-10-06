import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Dependencies.models.Vgg import VggCifar10
from Dependencies import UtilsScript
from Dependencies import GetDataScript


def get_argv():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')
    parser.add_argument('--depth', default=19, type=int,
                        help='depth of the neural network')
    args = parser.parse_args()
    print(args)
    return args


# additional subgradient descent on the sparsity-induced penalty term
def updateBN(args, model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1
    return


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        if args.sr:
            updateBN(args, model)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
    return


def test(args, model, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def main():
    args = get_argv()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = GetDataScript.get_data_loader('cifar10', '../../Datasets', is_train=True, bs=args.batch_size,
                                                 shuffle=False)
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../../Datasets', train=True, download=False,
    #                      transform=transforms.Compose([
    #                          transforms.Pad(4),
    #                          transforms.RandomCrop(32),
    #                          transforms.RandomHorizontalFlip(),
    #                          transforms.ToTensor(),
    #                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                      ])),
    #     batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = GetDataScript.get_data_loader('cifar10', '../../Datasets', is_train=False, bs=args.test_batch_size,
                                                shuffle=False)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../../Datasets', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     ])),
    #     batch_size=args.test_batch_size, shuffle=False, **kwargs)

    vgg_type = '19'
    model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))
    if args.refine:
        # checkpoint = torch.load(args.refine)
        # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])

        # model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))
        # model.load_state_dict(checkpoint['state_dict'])
        UtilsScript.load_model(model)
    # else:
    #     model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(args, model, train_loader, optimizer, epoch)
        _ = test(args, model, test_loader)
        # print(UtilsScript.var_to_string(prec1, with_data=True))
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        UtilsScript.save_model(model)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, filepath=args.save)

    # print("Best accuracy: ", best_prec1)


if __name__ == '__main__':
    main()
