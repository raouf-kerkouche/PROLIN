#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import random

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, fixed_train_sample=None, train_ratio=0.8):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.batch_size = args.local_bs

        #train_num = int(train_ratio*len(idxs))
        train_num = len(idxs)
        if fixed_train_sample == None:
            idxs = list(idxs)
            self.idxs_train = idxs#[:train_num]
            #self.idxs_test = idxs[train_num:]
        else:
            self.idxs_train = random.sample(list(idxs - {fixed_train_sample}), train_num - 1) + [fixed_train_sample]
            random.shuffle(self.idxs_train)
            #self.idxs_test = list(idxs - set(self.idxs_train))

        self.idxs_test = self.idxs_train

        #self.trainloader, self.testloader = train_test_split(
        #    dataset, list(idxs), batch_size=args.local_bs)
        #self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.args.device)

    def update_weights(self, model, global_round):
        trainloader = DataLoader(Subset(self.dataset, self.idxs_train),
                                batch_size=self.batch_size, shuffle=True)

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)#,
                                         #weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(trainloader.dataset),
                        100. * batch_idx / len(trainloader.dataset), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        testloader = DataLoader(Subset(self.dataset, self.idxs_test),
                                batch_size=self.batch_size, shuffle=True)

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            #print (pred_labels, labels)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.NLLLoss().to(args.device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
