from os.path import splitext, basename, dirname, join
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def load_module(fn, name):
    mod_name = splitext(basename(fn))[0]
    mod_path = dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_model(model_fn, model_name, args=None):
    model = load_module(model_fn, model_name)
    model1 = model(**args) if args else model()
    return model1


def save_checkpoints(net, optimizer, epoch, model_path):
    latest = {}
    latest['epoch'] = epoch
    latest['net'] = net.state_dict()
    latest['optim'] = optimizer.state_dict()
    if epoch % 40 == 0:
        torch.save(latest, join(model_path, 'latest.pth%d.tar' % epoch))


def return_loaders(root, batch_size, **kwargs):
    """
    Return the loader for the data. This is used both for training and for
    validation. Currently, hardcoded to CIFAR10. 
    :param root: (str) Path of the root for finding the appropriate pkl/npy.
    :param batch_size: (int) The batch size for training.
    :param kwargs:
    :return: The train and validation time loaders.
    """
    trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_train = transforms.Compose(trans)
    transform_test = transforms.Compose(trans[-2:])
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size, shuffle=False)
    return train_loader, test_loader
