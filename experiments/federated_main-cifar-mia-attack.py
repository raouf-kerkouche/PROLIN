#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from scipy.linalg import lstsq
import pickle

import torch
from tensorboardX import SummaryWriter

from optionscifar import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LogisticRegression
from utils import get_dataset, reduce_weights, exp_details, flatten_params, f_pred, f_err
#from MIA import MIA
from MIA5 import MIA
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import statsmodels.api as sm
import cvxpy as cp

#from pympler import summary
#from memory_profiler import profile


def regression(p_A, p_b):
    # Define and solve the CVXPY problem.
    user_num = p_A.shape[1]
    x = cp.Variable((user_num,p_b.shape[1]))

    # Ridge regression (OLS with Thikonov regularization)
    # !!! Since there are only a few positive samples, false negatives should be more common than false positive (to be checked)!!!
    # Goes to negative (which is the case by assumption, most are negative), sampling can make it positive
    # we contract the solutions close to zero (regulation act as gaussian prior around 0 in a bayesian interpretation)
    # we consider those solutions that are close to 0
    # manipulating \lambda=2  corresponds to trading-off bias and variance
    # This gives a more intuitive interpretation for why Tikhonov regularization leads to a unique solution to the least-squares problem: there are infinitely many vectors
    # satisfying the constraints obtained from the data, but since we come to the problem with a prior belief that 
    # is normally distributed around the origin, we will end up choosing a solution with this constraint in mind.
    # Compared to ordinary least squares, ridge regression is not unbiased. It accepts little bias to reduce variance 
    # and the mean square error, and helps to improve the prediction accuracy. Thus, ridge estimator yields more 
    # stable solutions by shrinking coefficients but suffers from the lack of sensitivity to the data (i.e., gains generalizability).
    # \lambda trades of small training error with ‘simple’ solutions (will have similar solution but with some bias)

    # maybe if there is a false negative (positive who is reported as negative) can be corrected by forcing it to be around
    # 0 and use our sampling. though it would also increase false positive, since negatives can be reported as positive?
    # in any case, the idea was that errors can be fixed by sampling if the solutions are close to 0, hence we used regularization!
    # can be checked by confusion matrix!

    # we introduce bias to have smaller variance -> we will have more false negatives (since the data is biased towards
    # negatives). Then we fix the bias with sampling ??
    # What regularization does is gives more importance to only important parameters and ignore others, 
    # thus reducing the complexity of the model. -> hence we don't need weights per round (mia accuracy?)?
    
    #objective = cp.Minimize(cp.sum_squares(cp.multiply(cp.sqrt(weights), p_A @ x - p_b)) + cp.sum_squares(x))
    objective = cp.Minimize(cp.sum_squares(p_A @ x - p_b) + 2 * cp.sum_squares(x))
        
    prob = cp.Problem(objective)
    prob.solve()

    return x.value


#seed=0 #0

#torch.manual_seed(seed)
#random.seed(seed)
#np.random.seed(seed)

def main():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('./results')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = args.device
    
    
    ########################################################################
    
    list_local_it=[1]
    
    for local_it in list_local_it:
        #list_runs=[2,3]
        for run in range(1,4):
            
            #seed=0 #0

            #torch.manual_seed(seed)
            #random.seed(seed)
            #np.random.seed(seed)

            #args.attacker_model_reg=reg
            #args.penality=penality
    
            args.local_sgd_it=local_it

            # load dataset and user groups
            #args.local_sgd_it = 1
            client_size = args.local_bs * args.local_sgd_it

            print ("Records per client:", client_size)
            train_dataset, test_dataset, user_groups, mia_idxs = get_dataset(args, client_size, mia_ratio=0.1)
            print("len(train_dataset):",len(train_dataset))
            print("len(mia_idx):", len(mia_idxs))
            print("args.num_users:",args.num_users)
            print ("Max. client data size:", (len(train_dataset) - len(mia_idxs)) // args.num_users)

            assert client_size <= (len(train_dataset) - len(mia_idxs)) // args.num_users, "Not enough data!"

            # BUILD MODEL (TODO: Only mnist with IID is implemented, DON'T use other yet)
            if args.model == 'cnn':
                # Convolutional neural netork
                if args.dataset == 'mnist':
                    global_model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    global_model = CNNMnist(args=args) #CNNFashion_Mnist(args=args)
                elif args.dataset == 'cifar':
                    global_model = CNNCifar(args=args)

            elif args.model == 'mlp':
                # Multi-layer preceptron
                img_size = train_dataset[0][0].shape
                len_in = 1
                for x in img_size:
                    len_in *= x
                    global_model = MLP(dim_in=len_in, dim_hidden=64,
                                       dim_out=args.num_classes)
            else:
                exit('Error: unrecognized model')

            # Set the model to train and send it to device.
            global_model.to(device)
            global_model.train()
            print(global_model)

            ## MIA training

            # Select target sample randomly
            attacked_user = random.choice(list(user_groups.keys()))
            target_sample_idx = random.choice(list(user_groups[attacked_user]))

            attacked_users = set(random.sample(list(set(user_groups.keys()) - {attacked_user}), int(args.num_users * args.prop) - 1) \
                + [attacked_user])
            # replace an existing random sample in each attacked user's training set with the target sample
            for user in attacked_users:
                user_groups[user].pop()
                user_groups[user].add(target_sample_idx)

            print (f"--> Attacked users: {attacked_users}\n--> Target sample idx: {target_sample_idx}")

            total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
            print (f"Target model size: {total_params}\n")    

            # Training
            train_loss, train_accuracy, mia_accuracy, train_loss_test, train_accuracy_test, mia_accuracy_val = [], [], [], [], [], []
            expB_err, expC_err, expC_err_ols, expC_err_reg = [], [], [], []
            #val_acc_list = []
            #cv_loss, cv_acc = [], []
            #print_every = 2
            #val_loss_pre, counter = 0, 0

            tepoch = tqdm(range(args.epochs), unit="round")

            # For disaggregation
            A = np.zeros((args.epochs, args.num_users))
            X = np.zeros((args.epochs, args.num_users))
            b = np.zeros(args.epochs)
            b_grads = np.zeros((args.epochs, total_params))

            ground_truth = np.zeros(args.num_users)
            ground_truth[list(attacked_users)] = 1 
            # For plotting distribution of activation values
            linear_vals = defaultdict(list) 


            l_var=[]
            l_var_pos=[]
            l_var_neg=[]
            l_mean=[]
            l_mean_pos=[]
            l_mean_neg=[]
            l_median=[]
            l_median_pos=[]
            l_median_neg=[]
            l_mean_deviation_pos=[]
            l_mean_deviation_neg=[]

            Dic_batch= defaultdict(list)

            for epoch in tepoch:
                tepoch.set_description(f"Federated Training")
                local_weights, local_losses, mia_features = [], [], []

                global_model.train()
                # select clients randomly per round
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                # Participation matrix
                A[epoch, idxs_users] = 1

                # Train attacker model per round
                mia = MIA(args=args, dataset=train_dataset, target_model=global_model, 
                    idxs=mia_idxs, logger=logger, lr=0.01)

                attacker_model = LogisticRegression(input_dim=total_params, output_dim=1)
                attacker_model.to(device)
                attacker_model.train()

                mia.train(attacker_model, target_sample_idx)


                acc_val_mia, l_var, l_var_pos, l_var_neg, l_mean, l_mean_pos, l_mean_neg, l_median, l_median_pos, l_median_neg, l_mean_deviation_pos, l_mean_deviation_neg = mia.test_inference_bis(attacker_model,l_var,l_var_pos,l_var_neg,l_mean,l_mean_pos,l_mean_neg,l_median,l_median_pos,l_median_neg,l_mean_deviation_pos,l_mean_deviation_neg)

                #Dic_batch=mia.test_inference_batch(attacker_model,Dic_batch)

                #print("l_var:",l_var)
                #print("l_var_pos:",l_var_pos)
                #print("l_var_neg:",l_var_neg)

                #print("l_mean:",l_mean)
                #print("l_mean_pos:",l_mean_pos)
                #print("l_mean_neg:",l_mean_neg)

                #print("l_median:",l_median)
                #print("l_median_pos:",l_median_pos)
                #print("l_median_neg:",l_median_neg)

                #print("l_mean_deviation_pos:",l_mean_deviation_pos)
                #print("l_mean_deviation_neg:",l_mean_deviation_neg)

                #mia_accuracy_val.append(mia.test_inference(attacker_model))

                mia_accuracy_val.append(acc_val_mia)

                print ("MIA test accuracy:", mia_accuracy_val[-1])

                # only a single global model is maintained, and is used to train the local models
                # sequentially. LocalUpdate is just a dispatcher
                labels = []
                for idx in idxs_users:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,\
                                              idxs=user_groups[idx], logger=logger, fixed_train_sample=target_sample_idx if idx in attacked_users else None)

                    #print (f"Client {idx}, {idx in attacked_users}, {local_model.idxs_train}, {target_sample_idx}")
                    model, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=epoch)

                    local_weights.append(copy.deepcopy(model.state_dict()))
                    mia_features.append(mia.get_features(model, global_model))
                    labels.append(idx in attacked_users)
                    local_losses.append(copy.deepcopy(loss))

                # Launch MIA
                correct = mia.predict(attacker_model, torch.stack(mia_features), torch.tensor(labels), scale=True)[0]
                mia_accuracy.append((correct / len(labels), sum(labels), len(labels)))

                for u, v in zip(idxs_users, mia.linear(attacker_model, torch.stack(mia_features)).detach().cpu().numpy()):
                    X[epoch][u] = v
                    linear_vals[u].append(v)

                # update global weights
                global_weights = reduce_weights(local_weights)

                ref_model = copy.deepcopy(global_model)

                # since MIA feature is (model - global_model), we need to scale global model if model = sum(local_models) and we want to preserve linearity
                # Also, Ax+b as well as std.scaling are not additive either, we need some correction 
                # (NOTE: adversary can do these if it knows the number of participating clients in every federated round)
                global_model.load_state_dict(reduce_weights(local_weights, op='sum'))
                b[epoch] = mia.linear(attacker_model, mia.get_features(global_model, ref_model, corr_factor=len(local_weights)), \
                    corr_factor = len(local_weights) - 1)

                # this is the model holding the sum for expC
                b_grads[epoch] = flatten_params(global_model.cpu())

                # These two values should be identical after the above corrections (i.e., sum of linear activation vs. linear activation of the sum):
                #print ("Sum of linear activations:", np.dot(X[epoch], A[epoch]),  "Model applied on the sum of local weights:", b[epoch])

                # update global weights
                global_model.load_state_dict(global_weights)

                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all users at every epoch
                list_acc, list_loss = [], []
                global_model.eval()
                for _ in range(args.num_users):
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=global_model.to(args.device))
                    list_acc.append(acc)
                    list_loss.append(loss)

                train_accuracy.append(sum(list_acc)/len(list_acc))

                print("Avg accuracy:",train_accuracy)
                print("Avg loss:",train_loss)


                acc_test, loss_test = test_inference(args, global_model.to(args.device), test_dataset)

                train_loss_test.append(loss_test)
                train_accuracy_test.append(acc_test)

                print("Avg accuracy test:",train_accuracy_test)
                print("Avg loss test:",train_loss_test)

                #train_loss.append(sum(list_acc)/len(list_acc))
                #print ("Loss:", sum(list_loss)/len(list_loss))

                # print global training loss after every 'i' rounds
                #if (epoch+1) % print_every == 0:
                #    print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                #    print(f'Training Loss : {np.mean(np.array(train_loss))}')
                #    print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                #    print("MIA Accuracy: {:.2f}%".format(100 * np.mean([x[0] for x in mia_accuracy])))#

                tepoch.set_postfix(loss=np.mean(np.array(train_loss)), accuracy="%.3f" % train_accuracy[-1], mia=np.mean([x[0] for x in mia_accuracy]))
                
                """

                #correct = 0
                # Plotting the distribution of lin. activations per client
                #for cl, vals in linear_vals.items():
                #    pred = np.sum(np.array(vals) > 0) /  len(vals) > 0.5
                #    correct += pred == (cl in attacked_users) 
                #    sns.histplot(vals, stat='probability', kde=True, bins=np.arange(-30, 30))
                #    plt.savefig(f"client_{cl}_{cl in attacked_users}.png")
                #    plt.clf()
                
                # expC_err_ols:
                # x_init has a size of (user_num x model_params)
                x_init = lstsq(A[:epoch+1,:], b_grads[:epoch+1])[0]
                #weights = abs(b) ** 17
                #weights /= weights.sum()
                #weights=np.array(mia_accuracy_val) ** 2
                #weights /= weights.sum()

                #print("w:",weights.shape)
                #print("b_grad:",b_grads[:epoch].shape)
                #print("A:",A[:epoch,:].shape)

                #x_init = sm.WLS(b_grads[:epoch+1], A[:epoch+1,:], weights).fit().params
                print ("x init shape:", x_init.shape)

                mia_features = [(torch.from_numpy(x).float().to(args.device) - flatten_params(ref_model)) for x in x_init]
                correct, predictions = mia.predict(attacker_model, torch.stack(mia_features), torch.tensor(ground_truth), scale=True)
                print("expC error:", 1 - (correct / len(ground_truth)), "F1-score:", f1_score(ground_truth, predictions.cpu()))
                expC_err_ols.append((1 - (correct / len(ground_truth)) , predictions, ground_truth))
                
                
                # expC_err_reg:
                # x_init has a size of (user_num x model_params)
                x_init = regression(A[:epoch+1,:], b_grads[:epoch+1])
                #weights = abs(b) ** 17
                #weights /= weights.sum()
                #weights=np.array(mia_accuracy_val) ** 2
                #weights /= weights.sum()

                #print("w:",weights.shape)
                #print("b_grad:",b_grads[:epoch].shape)
                #print("A:",A[:epoch,:].shape)

                #x_init = sm.WLS(b_grads[:epoch+1], A[:epoch+1,:], weights).fit().params
                print ("x init shape:", x_init.shape)

                mia_features = [(torch.from_numpy(x).float().to(args.device) - flatten_params(ref_model)) for x in x_init]
                correct, predictions = mia.predict(attacker_model, torch.stack(mia_features), torch.tensor(ground_truth), scale=True)
                print("expC error:", 1 - (correct / len(ground_truth)), "F1-score:", f1_score(ground_truth, predictions.cpu()))
                expC_err_reg.append((1 - (correct / len(ground_truth)) , predictions, ground_truth))
                

                # expC:
                # x_init has a size of (user_num x model_params)
                #x_init = lstsq(A[:epoch,:], b_grads[:epoch])[0]
                #weights = abs(b) ** 17
                #weights /= weights.sum()
                weights=np.array(mia_accuracy_val) ** 2
                weights /= weights.sum()

                #print("w:",weights.shape)
                #print("b_grad:",b_grads[:epoch].shape)
                #print("A:",A[:epoch,:].shape)

                x_init = sm.WLS(b_grads[:epoch+1], A[:epoch+1,:], weights).fit().params
                print ("x init shape:", x_init.shape)

                mia_features = [(torch.from_numpy(x).float().to(args.device) - flatten_params(ref_model)) for x in x_init]
                correct, predictions = mia.predict(attacker_model, torch.stack(mia_features), torch.tensor(ground_truth), scale=True)
                print("expC error:", 1 - (correct / len(ground_truth)), "F1-score:", f1_score(ground_truth, predictions.cpu()))
                expC_err.append((1 - (correct / len(ground_truth)) , predictions, ground_truth))

                l_linear_C=[]
                for u, v in zip(np.arange(args.num_users), mia.linear(attacker_model,torch.stack(mia_features)).detach().cpu().numpy()):
                    l_linear_C.append(v)

                print("l_linear_C length:",len(l_linear_C))

                # expB:
                #x_init = lstsq(A[:epoch,:], b[:epoch])[0]
                #weights = abs(b) ** 17
                #weights /= weights.sum()
                weights=np.array(mia_accuracy_val) ** 2
                weights /= weights.sum()
                x_init = sm.WLS(b[:epoch+1], A[:epoch+1,:], weights).fit().params
                predictions = f_pred(x_init)
                print ("expB error:", f_err(ground_truth, predictions), "F1-score:", f1_score(ground_truth, predictions))

                expB_err.append((f_err(ground_truth, predictions), predictions, ground_truth))
                
                """
                
                torch.save(ref_model.state_dict(), f"./Presentation/ref_model_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{epoch}_{run}.pt")

                torch.save(attacker_model.state_dict(), f"./Presentation/attacker_model_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{epoch}_{run}.pt")
                pickle.dump(mia.scaler,open(f"./Presentation/scaler_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{epoch}_{run}.pkl",'wb'))

                    #pickle.dump(Dic_batch,open("Dic_batch.pickle",'wb'))

            #print ("Aggregated attack accuracy:", correct / len(linear_vals))

            # Test inference after completion of training
            # test_acc, test_loss = test_inference(args, global_model, test_dataset)

            print(f'\n=== Results after {args.epochs} global rounds of training ====')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Avg MIA Accuracy: {:.2f}%".format(100 * np.mean([x[0] for x in mia_accuracy])))
            print ("Losses:", train_loss)

            # Saving the objects train_loss and train_accuracy:
            #file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            #    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
            #           args.local_ep, args.local_bs)
            #with open(file_name, 'wb') as f:
            #    pickle.dump([train_loss, train_accuracy], f)

            # Saving for disaggregation (see disaggregation.py)
            np.save(f"./Presentation/X_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", X)
            np.save(f"./Presentation/A_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", A)
            np.save(f"./Presentation/b_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", b)
            np.save(f"./Presentation/b_grads_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", b_grads)
            np.save(f"./Presentation/ground_truth_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", ground_truth)
            np.save(f"./Presentation/MIA_accuracy_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", mia_accuracy)
            np.save(f"./Presentation/MIA_accuracy_val_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", mia_accuracy_val)
            np.save(f"./Presentation/train_loss_test_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", train_loss_test)
            np.save(f"./Presentation/train_accuracy_test_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", train_accuracy_test)
            np.save(f"./Presentation/l_var_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_var)
            np.save(f"./Presentation/l_var_pos_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_var_pos)
            np.save(f"./Presentation/l_var_neg_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_var_neg)
            np.save(f"./Presentation/l_mean_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean)
            np.save(f"./Presentation/l_mean_pos_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean_pos)
            np.save(f"./Presentation/l_mean_neg_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean_neg)
            np.save(f"./Presentation/l_median_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_median)
            np.save(f"./Presentation/l_median_pos_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_median_pos)
            np.save(f"./Presentation/l_median_neg_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_median_neg)
            np.save(f"./Presentation/l_mean_deviation_pos_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean_deviation_pos)
            np.save(f"./Presentation/l_mean_deviation_neg_{args.optimizer}_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean_deviation_neg)

            #pickle.dump((expB_err, expC_err,expC_err_ols,expC_err_reg),open("./Presentation/exp_err_weights_"+str(args.attacker_model_reg)+"_"+str(args.penality)+"_"+str(args.dataset)+"_"+str(args.local_bs)+"_"+str(args.local_sgd_it)+"_"+str(run)+".pickle",'wb'))
            
            #pickle.dump((expB_err, expC_err),open("exp_err_weights.pickle",'wb'))

            #pickle.dump(Dic_batch,open("Dic_batch.pickle",'wb'))

            print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))

            # PLOTTING (optional)
            # import matplotlib
            # import matplotlib.pyplot as plt
            # matplotlib.use('Agg')

            # Plot Loss curve
            # plt.figure()
            # plt.title('Training Loss vs Communication rounds')
            # plt.plot(range(len(train_loss)), train_loss, color='r')
            # plt.ylabel('Training loss')
            # plt.xlabel('Communication Rounds')
            # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
            #             format(args.dataset, args.model, args.epochs, args.frac,
            #                    args.iid, args.local_ep, args.local_bs))
            #
            # # Plot Average Accuracy vs Communication rounds
            # plt.figure()
            # plt.title('Average Accuracy vs Communication rounds')
            # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
            # plt.ylabel('Average Accuracy')
            # plt.xlabel('Communication Rounds')
            # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
            #             format(args.dataset, args.model, args.epochs, args.frac,
            #                    args.iid, args.local_ep, args.local_bs))

if __name__ == '__main__':
    main()