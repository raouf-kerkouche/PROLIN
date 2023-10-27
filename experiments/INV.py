from torch.utils.data.sampler import Sampler

import copy
import random
from torch.utils.data import DataLoader, Subset, random_split
import torch
from utils import TorchStandardScaler, flatten_params
from torch import nn
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection 
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
import gc
import numpy as np


class INV(object):
    def __init__(self, args, dataset, target_model, idxs, logger, lr=0.01):
        self.args = args
        self.logger = logger
        self.target_dataset = dataset

        # Default criterion set to BCE for logistic regression
        self.aga_criterion = nn.BCELoss().to(self.args.device)
        self.lr = lr
        self.idxs = idxs
        self.target_model = target_model

    def _get_train_data(self, data_size, target_idx=None):
        idxs_no_target = list(self.idxs - {target_idx})
        if target_idx == None:
            return Subset(self.target_dataset, random.sample(idxs_no_target, data_size))
        else:
            return Subset(self.target_dataset, random.sample(idxs_no_target, data_size - 1) + [target_idx])

    def _train_target_model(self, target_idx, local_sgd_it, sample_num=200):
        target_model = copy.deepcopy(self.target_model)
        target_model.to(self.args.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(target_model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(target_model.parameters(), lr=self.args.lr)#,
                                         #weight_decay=1e-4)

        criterion = nn.NLLLoss().to(self.args.device)
        
        def train_step_honest(images, labels):
            target_model.zero_grad()
            log_probs = target_model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

        inv_dataset = []    
   
        init_model = copy.deepcopy(target_model)
         
        # more local iterations decreases attack accuracy proportionally
        tsamples = tqdm(range(sample_num), unit="sample")
        #tsamples = tqdm(range(6000*6), unit="sample")
        for _ in tsamples:
            tsamples.set_description(f"Generating training data for INV")

            target_model.load_state_dict(init_model.state_dict())
            target_model.train()   

            # Sample IN (with target sample -> label = True)
            _train_data = self._get_train_data(self.args.local_bs * local_sgd_it, target_idx=None)
            for _ in range(self.args.local_ep):
                loader = DataLoader(_train_data, batch_size=self.args.local_bs, shuffle=True)
                for images, labels in loader:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    train_step_honest(images, labels)

            inv_dataset.append((-self.get_features(target_model, init_model), True))
          
            # Sample OUT (w/o target sample -> label = False)
            target_model.load_state_dict(init_model.state_dict())
            target_model.train()

            _train_data = self._get_train_data(self.args.local_bs * local_sgd_it, target_idx=None)
            for _ in range(self.args.local_ep):
                loader = DataLoader(_train_data, batch_size=self.args.local_bs, shuffle=True)
                for images, labels in loader:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    train_step_honest(images, labels)

            inv_dataset.append((self.get_features(target_model, init_model), False))
            
            #gc.collect()

            #torch.cuda.empty_cache()
          
        return inv_dataset

    def get_features(self, local_model, global_model, corr_factor = 1.0):
        return flatten_params(local_model) - torch.mul(corr_factor, flatten_params(global_model))

    def train(self, model, target_idx, batch_size = 16): #16
        # Generate training data for AGA
        inv_dataset = self._train_target_model(target_idx, self.args.local_sgd_it, sample_num=len(self.idxs))#self.args.mia_sample_num)
        
        print("HNA")

        self.train_data, self.test_data = random_split(inv_dataset, [int(0.8 * len(inv_dataset)) , len(inv_dataset) - int(0.8 * len(inv_dataset))])
        
        self.scaler = TorchStandardScaler()
        scaled_x_data = self.scaler.fit_transform(torch.stack([x[0] for x in self.train_data]))
        self.train_data = [(x, self.train_data[i][1]) for i, x in enumerate(scaled_x_data)]
        
        
        scaled_x_data = self.scaler.transform(torch.stack([x[0] for x in self.test_data]))
        self.test_data = [(x, self.test_data[i][1]) for i, x in enumerate(scaled_x_data)]
        
        #print("HNA")
        
        #print("OK")
        #print("size train:",len(self.train_data))
        #print("size test:",len(self.test_data))
        #print("OK")
        
        self.trainloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
    
        # Train AGA model
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        
        acc_max=0
        
        #print("HNA")
        
        for epoch in range(self.args.attacker_model_ep):
            batch_loss = []
            #cpt=0
            with tqdm(self.trainloader, unit="batch") as tepoch:
                #tepoch.set_description(f"Epoch {epoch}")
                for updates, labels in tepoch:
                    #cpt+=1
                    #print(cpt)
                    tepoch.set_description(f"Epoch {epoch}")
                    updates, labels = updates.to(self.args.device), labels.to(self.args.device)
                    #print (updates, labels)
                   
                    # Setting our stored gradients equal to zero
                    model.zero_grad()
                    outputs = torch.squeeze(model(updates))
                    
                    #print("Outputs:",outputs)
                    
                    if(self.args.attacker_model_reg=='l1'):
                        #Replaces pow(2.0) with abs() for L1 regularization
     
                        l1_lambda = self.args.penality
                        l1_norm = sum(p.abs().sum() for p in model.parameters())
                        add_reg=l1_lambda * l1_norm
                    
                    elif(self.args.attacker_model_reg=='l2'):
                        #Replaces pow(2.0) with abs() for L1 regularization
                        
                        l2_lambda = self.args.penality
                        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                        add_reg=l2_lambda * l2_norm
                    
                    loss = self.aga_criterion(outputs, labels.to(torch.float32)) + add_reg

                    predictions = outputs > 0.5
                
                    accuracy = torch.sum(torch.eq(predictions, labels)).item() / batch_size
                
                    # Computes the gradient of the given tensor w.r.t. graph leaves
                    loss.backward()
                    # Updates weights and biases with the optimizer (SGD)
                    optimizer.step()

                    tepoch.set_postfix(loss=loss.item(), accuracy="%.3f" % accuracy)
                    batch_loss.append(loss.item())
             
            model.eval()
            acc_ep= self.test_inference(model)
            model.train()

            if(acc_ep>acc_max):
                model_weights=copy.deepcopy(model.state_dict())
                acc_max=acc_ep
                ep_tmp=epoch
                        
                #print("Voila")
                #print("Cpt:",cpt)
                #print("len(self.trainloader)",len(self.trainloader))
                #print("Fin")
                
        print("Best_AGA_epoch:",ep_tmp)
        print("Best_AGA_acc:",acc_max)
        
        model.load_state_dict(model_weights)
        
        return model #, epoch_loss

    def linear(self, model, updates, corr_factor = 0):
        model.eval()

        # Scaling is not linear
        updates = self.scaler.transform(updates) + corr_factor * -self.scaler.mean / self.scaler.std

        # and Ax+b is not linear either
        return torch.squeeze(model.linear(updates) + corr_factor * model.linear.bias.detach())

    def predict(self, model, updates, labels, scale=False):
        model.eval()

        if scale:
            updates = self.scaler.transform(updates)

        outputs = torch.squeeze(model(updates))

        # Prediction
        predictions = outputs > 0.5

        return torch.sum(torch.eq(predictions.to(self.args.device), labels.to(self.args.device))).item(), predictions.to(self.args.device) # correct predictions

    def test_inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        total, correct = 0.0, 0.0

        self.testloader = DataLoader(self.test_data, batch_size=10, shuffle=True)

        for batch_idx, (updates, labels) in enumerate(self.testloader):
            updates, labels = updates.to(self.args.device), labels.to(self.args.device)

            correct += self.predict(model, updates, labels)[0]
            total += len(labels)
      
        accuracy = correct/total
        return accuracy
    
    def test_inference_bis(self, model, l_var, l_var_pos,l_var_neg,l_mean,l_mean_pos,l_mean_neg,l_median, l_median_pos, l_median_neg, l_mean_deviation_pos, l_mean_deviation_neg):
        """ Returns the inference accuracy and loss.
        """
        total, correct = 0.0, 0.0
        
        list_tmp=[]
        list_tmp_pos=[]
        list_tmp_neg=[]

        self.testloader = DataLoader(self.test_data, batch_size=10, shuffle=True)

        for batch_idx, (updates, labels) in enumerate(self.testloader):
            updates, labels = updates.to(self.args.device), labels.to(self.args.device)

            correct += self.predict(model, updates, labels)[0]
            total += len(labels)
            
            outputs = torch.squeeze(model.linear(updates)).to("cpu").detach().numpy()
            
            list_tmp=np.concatenate((list_tmp, outputs), axis=None)
            
            labels_bis=copy.deepcopy(labels).to("cpu").detach().numpy()
            
            ind_1=np.where(labels_bis==1)
            ind_0=np.where(labels_bis==0)
            
            list_tmp_pos=np.concatenate((list_tmp_pos, outputs[ind_1]), axis=None)
            list_tmp_neg=np.concatenate((list_tmp_neg, outputs[ind_0]), axis=None)
            
        print("length list_tmp:",len(list_tmp))
        print("length list_tmp_pos:",len(list_tmp_pos))
        print("length list_tmp_neg:",len(list_tmp_neg))
            
        l_var.append(np.var(list_tmp))
        l_var_pos.append(np.var(list_tmp_pos))
        l_var_neg.append(np.var(list_tmp_neg))
        
        l_mean.append(np.mean(list_tmp))
        l_mean_pos.append(np.mean(list_tmp_pos))
        l_mean_neg.append(np.mean(list_tmp_neg))
        
        l_median.append(np.median(list_tmp))
        l_median_pos.append(np.median(list_tmp_pos))
        l_median_neg.append(np.median(list_tmp_neg))
        
        l_mean_deviation_pos.append(np.mean(np.abs(np.subtract(np.array(list_tmp_pos),np.mean(list_tmp_pos)))))
        l_mean_deviation_neg.append(np.mean(np.abs(np.subtract(np.array(list_tmp_neg),np.mean(list_tmp_neg)))))
    
        accuracy = correct/total
        return accuracy, l_var, l_var_pos, l_var_neg, l_mean, l_mean_pos, l_mean_neg,l_median, l_median_pos, l_median_neg, l_mean_deviation_pos, l_mean_deviation_neg
    
    
    
    def test_inference_batch(self, model,Dic):
        """ Returns the inference accuracy and loss.
        """
        
        labels_total=[y for x,y in self.test_data]
        
        labels_total=np.array(labels_total).astype(int)
        
        #print(labels_total)
        
        
        ind_1=np.where(labels_total==1)[0]
        ind_0=np.where(labels_total==0)[0]
        
        #print("ind_1",ind_1)
        #print("ind_0",ind_0)
        
                
        for i in range(0,11):
            b_l=[]
            for _ in range(1000):
                
                if(i==0):
                    batch_ind=np.random.choice(ind_0,10-i,replace=False)
                elif(i==10):
                    batch_ind=np.random.choice(ind_1,i,replace=False)
                else:
                    batch_ind=np.random.choice(ind_1,i,replace=False)
                    batch_ind=np.concatenate((batch_ind, np.random.choice(ind_0,10-i,replace=False)), axis=None)
                
                
                #print(batch_ind)
                
                test_data_tmp=Subset(self.test_data, batch_ind)
                
                #print(test_data_tmp.data)
                            
                testloader = DataLoader(test_data_tmp, batch_size=10, shuffle=True)

                for batch_idx, (updates, labels) in enumerate(testloader):
                    
                    #print("labels",labels.shape)
                    
                    updates, labels = updates.to(self.args.device), labels.to(self.args.device)

                    labels_bis=copy.deepcopy(labels).to("cpu").detach().numpy()

                    outputs_agg = torch.sum(torch.squeeze(model.linear(updates))).item()
                    
                    #print(outputs_agg)
            
                    b_l.append(outputs_agg)
            
            Dic[i].append((np.mean(b_l),np.median(b_l),np.min(b_l),np.max(b_l),np.var(b_l)))
            
        print(Dic)
         
        return Dic