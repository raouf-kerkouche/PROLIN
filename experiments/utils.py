#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_noniid_final
from sampling import cifar_iid, cifar_noniid
import numpy as np
from torch.utils.data import DataLoader, Subset

f_pred = lambda x : x > 0  # same as sigmoid(x) > 0.5
f_err = lambda x, y: np.linalg.norm(x - y, ord = 1) / len(x)
class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True) + 1e-7

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

def train_test_split(dataset, idxs, batch_size, train_ratio=0.8):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    # split indexes for train, validation, and test (80, 10, 10)
    idxs_train = idxs[:int(train_ratio*len(idxs))]
    idxs_test = idxs[int(train_ratio*len(idxs)):]

    trainloader = DataLoader(Subset(dataset, idxs_train),
                                batch_size=batch_size, shuffle=True)
    testloader = DataLoader(Subset(dataset, idxs_test),
                            batch_size=int(len(idxs_test)/10), shuffle=True)
    return trainloader, testloader

def get_dataset(args, client_size, mia_ratio):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        #all_idxs = range(len(train_dataset))
        #mia_idxs = set(np.random.choice(all_idxs, int(mia_ratio * len(train_dataset)), replace=False))
        #train_idxs = list(set(all_idxs) - mia_idxs)

        # sample training data amongst users
        if args.iid:
            all_idxs = range(len(train_dataset))
            mia_idxs = set(np.random.choice(all_idxs, int(mia_ratio * len(train_dataset)), replace=False))
            train_idxs = list(set(all_idxs) - mia_idxs)
            # Sample IID user data from Mnist
            #user_groups = cifar_iid(train_dataset, args.num_users)
            user_groups = mnist_iid(train_dataset, args.num_users, train_idxs, client_size=client_size)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                #user_groups = cifar_noniid(train_dataset, args.num_users, train_idxs)
                all_idxs = range(len(train_dataset))
                user_groups, l_idx = cifar_noniid_final(train_dataset, args.num_users, all_idxs)
                mia_idxs = set(np.random.choice(l_idx, int(mia_ratio * len(train_dataset)), replace=False))

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        #all_idxs = range(len(train_dataset))
        #mia_idxs = set(np.random.choice(all_idxs, int(mia_ratio * len(train_dataset)), replace=False))
        #train_idxs = list(set(all_idxs) - mia_idxs)

        # sample training data amongst users
        if args.iid:
            all_idxs = range(len(train_dataset))
            mia_idxs = set(np.random.choice(all_idxs, int(mia_ratio * len(train_dataset)), replace=False))
            train_idxs = list(set(all_idxs) - mia_idxs)
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users, train_idxs, client_size=client_size)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users, train_idxs)
            else:
                # Chose euqal splits for every user
                #user_groups = mnist_noniid(train_dataset, args.num_users, train_idxs)
                all_idxs = range(len(train_dataset))
                user_groups, l_idx = cifar_noniid_final(train_dataset, args.num_users, all_idxs)
                mia_idxs = set(np.random.choice(l_idx, int(mia_ratio * len(train_dataset)), replace=False))

    return train_dataset, test_dataset, user_groups, mia_idxs

def reduce_weights(w, op='avg'):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if op == 'avg':
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def flatten_params(model):
    return torch.cat([torch.flatten(p.detach()) for p in model.parameters()])
    
def get_update(global_model,updated_model):
    w_g=copy.deepcopy(global_model.state_dict())
    w_u=copy.deepcopy(updated_model.state_dict())
    
    update=copy.deepcopy(global_model.state_dict())
    
    for key in list(w_g.keys()):
        update[key]=torch.subtract(copy.deepcopy(w_u).get(key),copy.deepcopy(w_g).get(key)) # modifie ici
    
    return update

def update(global_model,update):
    w_g=copy.deepcopy(global_model.state_dict())
        
    for key in list(w_g.keys()):
        w_g[key]=torch.add(copy.deepcopy(w_g).get(key),copy.deepcopy(update).get(key)) # modifie ici
    
    return w_g

def flatten_from_weights(weights):
    return torch.cat([torch.flatten(weights.get(key).detach()) for key in weights])
    
def norm_2(model):
    flat=flatten_from_weights(model)
    #print("shape flat:",flat.shape)
    norm=torch.norm(flat,2)#.detach().cpu().numpy()
    #print("Norm_2:",norm)
    return norm

def clip(model,desired_norm):
    norm=norm_2(model)
    #print("norm_clip:",norm)
    #print("desired_norm:",desired_norm)
    
    tmp_model=copy.deepcopy(model)
    
    for key in tmp_model:
        tmp_model[key]=torch.divide(tmp_model.get(key), torch.maximum(torch.tensor(1.0),torch.divide(norm,desired_norm)))
    
    return tmp_model

def min_max(weights):
    fla=flatten_from_weights(weights)
    print("min:",torch.min(fla).item())
    print("max:",torch.max(fla).item())

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Device     : {args.device}\n')
    
    print(f'MIA parameters:')
    print(f'    Training samples  : {args.mia_sample_num}')
    print(f'    Training epochs   : {args.attacker_model_ep}\n')

    print(f'Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users    : {args.frac}')
    print(f'    Local Batch size     : {args.local_bs}')
    print(f'    Local SGD iterations : {args.local_sgd_it}')
    print(f'    Local Epochs         : {args.local_ep}')
    print(f'    Number of clients    : {args.num_users}\n')
    return
