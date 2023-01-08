# Train CIFAR10 with pytorch
from __future__ import print_function
import yaml
import sys
from time import time
import random
from os.path import abspath, dirname, join, isdir
from os import curdir, makedirs
import logging

import torch
import torch.optim as optim

from utils import (save_checkpoints, load_model, return_loaders)

torch.backends.cudnn.benchmark = True
base = dirname(abspath(__file__))
sys.path.append(base)


def train(train_loader, net, optimizer, criterion, train_info, epoch, device):
    """ Perform single epoch of the training."""
    net.train()
    # # initialize variables that are augmented in every batch.
    train_loss, correct, total = 0, 0, 0
    start_time = time()
    for idx, data_dict in enumerate(train_loader):
        img, label = data_dict[0], data_dict[1]
        inputs, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred, label)
        assert not torch.isnan(loss), 'NaN loss.'
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum()
        if idx % train_info['display_interval'] == 0:
            m2 = ('Time: {:.04f}, Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(time() - start_time, epoch, idx, len(train_loader),
                            float(train_loss), float(correct) / total))
            start_time = time()
    return net


def test(net, test_loader, device='cuda'):
    """ Perform testing, i.e. run net on test_loader data
        and return the accuracy. """
    net.eval()
    correct, total = 0, 0
    if hasattr(net, 'is_training'):
        net.is_training = False
    for (idx, data) in enumerate(test_loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        img, label = data[0].to(device), data[1].to(device)
        with torch.no_grad():
             pred = net(img)
        _, predicted = pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    if hasattr(net, 'is_training'):
        net.is_training = True
    return correct / total


def main(seed=None, use_cuda=True):
    # # set the seed for all.
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # # set the cuda availability.
    cuda = torch.cuda.is_available() and use_cuda
    device = torch.device('cuda' if cuda else 'cpu')
    yml = yaml.safe_load(open('model_ncp.yml'))  # # file that includes the configuration.
    cur_path = abspath(curdir)
    # # define the output path
    out = join(cur_path, 'results_poly', '')
    if not isdir(out):
        makedirs(out)

    # # set the dataset options.
    train_loader, test_loader = return_loaders(**yml['dataset'])
    m1 = 'Current path: {}. Length of iters per epoch: {}. Length of testing batches: {}.'
    print(m1.format(cur_path, len(train_loader), len(test_loader)))
    # # load the model.
    modc = yml['model']
    net = load_model(modc['fn'], modc['name'], modc['args']).to(device)

    # # define the criterion and the optimizer.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    sub_params = [p for p in list(net.parameters()) if p.requires_grad]
    decay = yml['training_info']['weight_dec'] if 'weight_dec' in yml['training_info'].keys() else 5e-4
    optimizer = optim.SGD(sub_params, lr=yml['learning_rate'],
                          momentum=0.9, weight_decay=decay)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total params: {}'.format(total_params))

    # # get the milestones/gamma for the optimizer.
    tinfo = yml['training_info']
    mil = tinfo['lr_milestones'] if 'lr_milestones' in tinfo.keys() else [40, 60, 80, 100]
    gamma = tinfo['lr_gamma'] if 'lr_gamma' in tinfo.keys() else 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mil, gamma=gamma)
    best_acc, best_epoch, accuracies = 0, 0, []

    for epoch in range(1, tinfo['total_epochs'] + 1):
        scheduler.step()
        net = train(train_loader, net, optimizer, criterion, yml['training_info'], 
                    epoch, device)
        save_checkpoints(net, optimizer, epoch, out)
        # # testing mode to evaluate accuracy. 
        acc = test(net, test_loader, device=device)
        if acc > best_acc:
            out_path = join(out, 'net_best_1.pth')
            state = {'net': net.state_dict(), 'acc': acc, 
                     'epoch': epoch, 'n_params': total_params}
            torch.save(state, out_path)
            best_acc = acc
            best_epoch = epoch
        accuracies.append(float(acc))
        msg = 'Epoch:{}.\tAcc: {:.03f}.\t Best_Acc:{:.03f} (epoch: {}).'
        print(msg.format(epoch,  acc, best_acc, best_epoch))
        logging.info(msg.format(epoch, acc, best_acc, best_epoch))


if __name__ == '__main__':
    main()


