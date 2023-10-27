# The posterior of the adversary can be modelled with a distribution
# if the prior is only the least square constraint, then the OLS/WLS solution
# should return the mean which is only one solution! (with the least l2 norm) 
# but the real solution can be different from this! 
# Idea: if a client can be positive then we consider it as positive. is there
# solution where a particular client is positive? (closest to the OLS solution?)

# Other solution (idea, not implemented): transfer every client to positive (with optimization),
# record the cost for each (what is a cost? does it matter? can be random as we are interested in the relative distance?), 
# and cluster the costs. cost can be the Earth Mover distance from a what?
# is it the same as computing the confidence for a decision? or just decide with probability sigmoid(OLS)?

# Other solution (implemented): maximize the number of positive clients (that is, maximize ||x|| in Ax=b)
# A form of this is EM v2, but without explicit optimization. This seems to be worse than EM v1

# LASSO or compressive sensing could work, if there was a linear transformation (hinge loss does not work) which makes the input sparse
# Rather, we shift to the negative with heavy L2 regularization, and apply a sampling techniques so that values which 
# which are close to zero can be positive

import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans

from optionsfashionmnist import args_parser
import statsmodels.api as sm
from tqdm import tqdm
import torch
import cvxpy as cp
from MIA5test import MIA
from utils import flatten_params
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LogisticRegression
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
import math
from statistics import NormalDist
import random
import io
import copy
from mtadam import MTAdam

# Needed to load saved ref_model on non-cuda (CPU) device:
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0.01):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.last_val = np.inf

    def __call__(self, loss):
        if abs(loss - self.last_val) < self.min_delta or (loss - self.last_val)>0:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            
        
            if(self.last_val>loss):
                self.counter=0
                self.last_val = loss

EXP_NUM = 10 # number of EM iterations our EM (pred_em should converge)
SAMPLE_NUM = 1000 #1000 # number of samples used to approximate the loglikelihood in the E-step
START_ROUND = 10
INTERVAL_ROUND = 10 # perform reconstruction in every INTERVAL_ROUND
MIN_MIA_ACC = 0 # only those rounds are considered for reconstruction where the mia test accuracy is above this threshold

REG_PARAM_EM = 2
REG_PARAM_RIDGE = 5 # should work for all

# REG_PARAM_EM = 1 for local_sgd=1 MNIST, run=1           
# REG_PARAM_EM = 1 for local_sgd=2 MNIST, run=1
# REG_PARAM_EM = 1 for local_sgd=3 MNIST, run=1
# REG_PARAM_EM = 2 for local_sgd=3 FMNIST, run=3
# REG_PARAM_EM = 1 for local_sgd=2 FMNIST, run=3
# REG_PARAM_EM = 3 for local_sgd=1 FMNIST, run=3
# REG_PARAM_EM = 2 for local_sgd=1 CIFAR?, run=3

#args = args_parser()

#prefix = f"./results/save_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}/"
#prefix = "./Presentation_inv/"
#prefix = "./300_round_variance_2/"


list_attacks=["aga"] #["aga","inv","mia"]

list_datasets=["fmnist","cifar","mnist"]

for attack in list_attacks:
    for dataset in list_datasets:
        if(dataset=="cifar"):
            from optionscifar import args_parser
            extra="sgd_"
            list_local_sgd_it=[2,1]
        elif(dataset=="fmnist"):
            from optionsfashionmnist import args_parser
            extra=""
            list_local_sgd_it=[3,2,1]
        else:
            from options import args_parser
            extra=""
            list_local_sgd_it=[3,2,1]
        
        args = args_parser()
        
        #if(attack=="mia"):
        #    prefix = "./Presentation/"
        #else:
        #    prefix = "./Results_for_paper/Presentation_"+str(attack)+"/"
        
        if(attack=="mia"):
            prefix = "./Presentation/"
        else:
            prefix = "./Presentation_"+str(attack)+"/"       
        
        for sgd_it in list_local_sgd_it:
            
            args.local_sgd_it=copy.deepcopy(sgd_it)


            list_stack_f1_score=[]
            list_stack_f1_score_agg=[]
            list_stack_f1_score_std=[]
            list_stack_f1_score_agg_std=[]

            # For Raouf's script
            runs=3

            for run in range(1,runs+1):

                X_real = np.load(prefix + f"X_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                A = np.load(prefix + f"A_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                b = np.load(prefix + f"b_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                b_grads = np.load(prefix + f"b_grads_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                ground_truth = np.load(prefix + f"ground_truth_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                #np.load(f"./Presentation/MIA_accuracy_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", mia_accuracy)
                #np.load(f"./Presentation/MIA_accuracy_val_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", mia_accuracy_val)
                global_test_loss = np.load(prefix + f"train_loss_test_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy") 
                global_test_acc = np.load(prefix + f"train_accuracy_test_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                #l_var = np.load(f"./Presentation/l_var_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_var)
                pos_var = np.load(prefix + f"l_var_pos_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                neg_var = np.load(prefix + f"l_var_neg_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                #l_mean = np.load(f"./Presentation/l_mean_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean)
                pos_mean = np.load(prefix + f"l_mean_pos_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                neg_mean = np.load(prefix + f"l_mean_neg_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                #np.load(f"./Presentation/l_median_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_median)
                #np.load(f"./Presentation/l_median_pos_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_median_pos)
                #np.load(f"./Presentation/l_median_neg_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_median_neg)
                #np.load(f"./Presentation/l_mean_deviation_pos_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean_deviation_pos)
                #np.load(f"./Presentation/l_mean_deviation_neg_{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy", l_mean_deviation_neg)

                # we scale only for CIFAR
                #if args.dataset == "cifar":
                #    b /= 100

                logpdf = lambda X, mu, sigma2 : - ((2 * sigma2) ** -1) * (X - mu)**2 - np.log((2 * math.pi * sigma2)**0.5)       
                pdf_normal = lambda X, mu, sigma2 : ((2 * math.pi * sigma2) ** -0.5) * np.exp(-0.5*((X - mu)**2 / sigma2))

                # Find a solution for the participation probabilities
                def solve_tau(X, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, out_size=1):
                    # Define and solve the CVXPY problem.

                    factor = pdf_normal(X, pos_mean_view, pos_var_view) - pdf_normal(X, neg_mean_view, neg_var_view)
                    bias = pdf_normal(X, neg_mean_view, neg_var_view)

                    # x : user_num
                    x = cp.Variable(out_size)
                    factor = np.nan_to_num(factor)
                    bias = np.nan_to_num(factor)

                    objective = cp.Maximize(
                        cp.sum(cp.multiply(factor, x) + bias)
                        )

                    constraints = [x >= 0, x <= 1]
                    prob = cp.Problem(objective, constraints)
                    prob.solve() 
                    return x.value


                # Log-likelihood prediction for every method
                def get_pred(output, pos_var_view, pos_mean_view, neg_var_view, neg_mean_view, usr_rounds, method=None):
                    last_output = output[-1][1]
                    user_num = len(last_output)

                    preds = np.zeros(user_num)

                    if method == 'simple':
                        for user in range(user_num):
                            if len(usr_rounds[user]) == 0:
                                continue

                            #logl_pos = logpdf(output[user].repeat(len(usr_rounds[user])), pos_mean_view[usr_rounds[user]], pos_var_view[usr_rounds[user]])
                            #logl_neg = logpdf(output[user].repeat(len(usr_rounds[user])), neg_mean_view[usr_rounds[user]], neg_var_view[usr_rounds[user]])

                            #preds[user] = logl_pos.sum() > logl_neg.sum()

                            preds[user] = solve_tau(last_output[user].repeat(len(usr_rounds[user])), pos_mean_view[usr_rounds[user]], 
                                        pos_var_view[usr_rounds[user]], neg_mean_view[usr_rounds[user]], 
                                        neg_var_view[usr_rounds[user]]) 

                    else:
                        for user in range(user_num):
                            if len(usr_rounds[user]) == 0:
                                continue

                            #logl_pos = logpdf(output[user].repeat(len(usr_rounds[user])), pos_mean_view[usr_rounds[user]], pos_var_view[usr_rounds[user]])
                            #logl_neg = logpdf(output[user].repeat(len(usr_rounds[user])), neg_mean_view[usr_rounds[user]], neg_var_view[usr_rounds[user]])

                            #preds[user] = logl_pos.sum() > logl_neg.sum()

                            round_num = len(pos_mean_view)
                            means_usr = np.zeros(round_num)
                            prev = np.empty([])
                            for rounds, means in output:
                                new_rnds = np.setdiff1d(rounds[user], prev)
                                means_usr[new_rnds] = means[user].repeat(len(new_rnds))
                                prev = rounds[user]

                            preds[user] = solve_tau(means_usr[usr_rounds[user]], pos_mean_view[usr_rounds[user]], 
                                        pos_var_view[usr_rounds[user]], neg_mean_view[usr_rounds[user]], 
                                        neg_var_view[usr_rounds[user]]) 

                            '''
                            pos = 0
                            for r in usr_rounds[user]:
                                logl_pos = - ((2 * pos_var_view[r]) ** -1) * (output[user] - pos_mean_view[r])**2 - np.log(np.sqrt(pos_var_view[r]) * (2 * math.pi)**0.5)
                                logl_neg = - ((2 * neg_var_view[r]) ** -1) * (output[user] - neg_mean_view[r])**2 - np.log(np.sqrt(neg_var_view[r]) * (2 * math.pi)**0.5)

                                #pos += weights_view[r] * (logl_pos - logl_neg > 0)
                                pos += logl_pos - logl_neg > 0

                            #pos /= weights_view[usr_rounds[user]].sum()# len(usr_rounds[user])
                            pos /= len(usr_rounds[user])
                            preds[user] = pos > 0.5
                            '''


                    return preds

                # to compute the confidence of OLS/WLS
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                def agg_decide(results, weights_view):
                    #return (sum(x * y for x, y in zip(results, weights_view)) / weights_view.sum()) > 0.5
                    return (sum(results) / len(results)) > 0.5

                def ridge_regression(p_A, p_b, p_weights, lamb):
                    # Define and solve the CVXPY problem.
                    user_num = p_A.shape[1]
                    x = cp.Variable(user_num)

                    # Generalized Ridge Regression (weighted residual + weighted penalty values)

                    objective = cp.Minimize(cp.sum_squares(cp.multiply(cp.sqrt(p_weights), p_A @ x - p_b)) + cp.sum_squares(cp.multiply(cp.sqrt(lamb), x)))

                    prob = cp.Problem(objective)
                    prob.solve()

                    return x.value
                
                
                ######################
                def unbalanced_loss(output, target, loss_weights):
                    loss_array = []
                    for i in range(10):
                        one_class_imbalanced_weights = torch.zeros(10)
                        one_class_imbalanced_weights = one_class_imbalanced_weights.to("cuda") # TODO do it elegant
                        one_class_imbalanced_weights[i] = loss_weights[i]
                        loss = F.nll_loss(output, target, weight=one_class_imbalanced_weights)
                        loss_array.append(loss)

                    return loss_array

                def train(args, model, device, train_loader, optimizer, epoch, use_MTAdam, use_Unbalanced, loss_weights):
                    model.train()

                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        if use_MTAdam:
                            loss_array = unbalanced_loss(output, target, loss_weights)
                            ranks = [1]*10
                            optimizer.step(loss_array, ranks, None)

                            # if batch_idx % 1 == 0:
                            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            #         epoch, batch_idx * len(data), len(train_loader.dataset),
                            #         100. * batch_idx / len(train_loader), loss_array[0].item()))
                        else:
                            if use_Unbalanced:
                                loss = F.nll_loss(output, target, weight=loss_weights)
                            else:
                                loss = F.nll_loss(output, target)
                            loss.backward()
                            optimizer.step()
                            # if batch_idx % args.log_interval == 0:
                            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            #         epoch, batch_idx * len(data), len(train_loader.dataset),
                            #         100. * batch_idx / len(train_loader), loss.item()))
                
                ###################

                def report(ground_truth, preds):

                    msg = "*** F1-score,"
                    for method in preds:
                        msg += " " + method + ": %.4f;" % f1_score(ground_truth, preds[method])

                    print (msg, "\n")

                    for method in preds:
                        tn, fp, fn, tp = confusion_matrix(ground_truth, preds[method]).ravel()
                        print (f"{method}; FP = {fp}, FN = {fn}")

                def solve_complete2(_A, _b, p_weights, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0):

                    (round_num, user_num) = _A.shape

                    _pos_means = np.tile(pos_mean_view, (user_num, 1)).T
                    _neg_means = np.tile(neg_mean_view, (user_num, 1)).T

                    _pos_vars = np.tile(pos_var_view, (user_num, 1)).T
                    _neg_vars = np.tile(neg_var_view, (user_num, 1)).T

                    pos_means = torch.tensor(_pos_means, dtype=torch.float)
                    neg_means = torch.tensor(_neg_means, dtype=torch.float)
                    pos_vars = torch.tensor(_pos_vars, dtype=torch.float)
                    neg_vars = torch.tensor(_neg_vars, dtype=torch.float)
                    weights = torch.tensor(p_weights, dtype=torch.float)  

                    def pdf_normal_t(X, mu, sigma2): 
                        return ((2 * math.pi * sigma2) ** -0.5) * torch.exp(-0.5*((X - mu)**2 / sigma2))

                    b_orig = torch.tensor(_b, dtype=torch.float)
                    p_A = torch.tensor(_A, dtype=torch.float)

                    #x = torch.linalg.lstsq(p_A, b_orig).solution
                    x = torch.rand(user_num, dtype = torch.float)

                    # pytorch supports broadcasting
                    taus = torch.rand(user_num, dtype = torch.float)  #torch.tensor(tau_init, dtype=torch.float)
                    x.requires_grad = True
                    taus.requires_grad = True

                    optimizer = torch.optim.Adam([x, taus], lr = 0.001)
                    #sse_loss = torch.nn.MSELoss(reduction = 'sum')
                    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5)
                    early_stopping = EarlyStopping(tolerance=100, min_delta=1e-5)

                    for i in range(1000000):
                        optimizer.zero_grad()     

                        obj_lstsq = torch.sum(weights * (b_orig - p_A @ x)**2) 

                        # average of x should be close to the REG/OLS/WLS result
                        #obj_reg = torch.sum((torch.diag(p_A @ x)  - b_orig)**2)
                        obj_reg = 5 * torch.sum(x**2)

                        obj_mll = -torch.sum(taus * p_A * torch.nan_to_num(pdf_normal_t(x, pos_means, pos_vars)) + (1 - taus) * p_A * torch.nan_to_num(pdf_normal_t(x, neg_means, neg_vars)))

                        obj_func = obj_lstsq + obj_reg + obj_mll

                        obj_func.backward()

                        optimizer.step()

                        # projection
                        with torch.no_grad():
                            taus.clamp_(0, 1)

                        #print ("Lstsq loss:", obj_lstsq.item(), "Mll Loss:", obj_mll.item())
                        #print ("Reg loss:", obj_reg.item(), "Mll Loss:", obj_mll.item())
                        #scheduler.step(obj_func.item())

                        early_stopping(obj_func.item())

                        if early_stopping.early_stop:    
                            break

                    print ("Lstsq loss:", obj_lstsq.item(), "Reg loss:", obj_reg.item(), "Mll Loss:", obj_mll.item())

                    #print ("X:", p_A * x)
                    return taus.detach().numpy()
                    #return x.detach().numpy() 

                def solve_complete(_A, p_weight, _reg_solution, _b, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0):

                    (round_num, user_num) = _A.shape

                    _pos_means = np.tile(pos_mean_view, (user_num, 1)).T
                    _neg_means = np.tile(neg_mean_view, (user_num, 1)).T

                    _pos_vars = np.tile(pos_var_view, (user_num, 1)).T
                    _neg_vars = np.tile(neg_var_view, (user_num, 1)).T
                    _weights = _A * np.tile(p_weight, (user_num, 1)).T

                    pos_means = torch.tensor(_pos_means, dtype=torch.float)
                    neg_means = torch.tensor(_neg_means, dtype=torch.float)
                    pos_vars = torch.tensor(_pos_vars, dtype=torch.float)
                    neg_vars = torch.tensor(_neg_vars, dtype=torch.float)

                    weights = torch.tensor(p_weight, dtype=torch.float)
                    weights_mll = torch.tensor(_weights, dtype=torch.float)

                    def pdf_normal_t(X, mu, sigma2): 
                        return ((2 * math.pi * sigma2) ** -0.5) * torch.exp(-0.5*((X - mu)**2 / sigma2))

                    b_orig = torch.tensor(_b, dtype=torch.float)
                    p_A = torch.tensor(_A, dtype=torch.float)
                    reg_sol = torch.tensor(_reg_solution * _A.sum(axis=0), dtype=torch.float)

                    x = torch.rand((round_num, user_num), dtype = torch.float) 
                    # pytorch supports broadcasting
                    taus = torch.rand(user_num, dtype = torch.float)  #torch.tensor(tau_init, dtype=torch.float)
                    
                    #eta_1 = torch.rand(1, dtype = torch.float)  #torch.tensor(tau_init, dtype=torch.float)
                    #eta_2 = torch.rand(1, dtype = torch.float)
                    #eta_3 = torch.rand(1, dtype = torch.float)
                    
                    x.requires_grad = True
                    taus.requires_grad = True
                    #eta_1.requires_grad = True
                    #eta_2.requires_grad = True
                    #eta_3.requires_grad = True

                    #optimizer = torch.optim.Adam([x, taus,eta_2,eta_3], lr = 0.001)
                    lr=0.001
                    optimizer = MTAdam([x, taus], lr=lr)
                    #optimizer = torch.optim.SGD([x, taus], lr = 0.001)
                    #sse_loss = torch.nn.MSELoss(reduction = 'sum')
                    #scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5)
                    early_stopping = EarlyStopping(tolerance=100, min_delta=1e-5)
                    
                    loss_min=math.inf

                    for i in range(200000):
                        optimizer.zero_grad()     

                        obj_lstsq = torch.sum(weights * (b_orig - torch.diag(x @ p_A.t()))**2) 

                        # average of x should be close to the REG/OLS/WLS result

                        obj_reg = torch.sum((torch.diag(p_A.t() @ x)  - reg_sol)**2)

                        obj_mll = -torch.sum(torch.log1p(taus * p_A * pdf_normal_t(x, pos_means, pos_vars) + (1 - taus) * p_A * pdf_normal_t(x, neg_means, neg_vars)))
                        
                        #print([obj_mll, obj_lstsq, obj_reg])
                        
                        obj_func = [obj_mll, obj_lstsq, obj_reg]
                        
                        #sig=nn.Sigmoid()

                        #obj_func = [sig(obj_mll), sig(obj_lstsq), sig(obj_reg)]
                        
                        #print("obj_func:",obj_func)
                        
                        #total_loss = torch.Tensor(obj_func) #* torch.exp(-eta) + eta
                        
                        #total_loss = total_loss.sum()
                        
                        total_loss= obj_mll+ obj_lstsq+ obj_reg

                        #total_loss = torch.Tensor(obj_func) * torch.exp(-eta) + eta
                        
                        #print(total_loss)
                        
                        #print("ETA1:",eta_1)
                        #print("ETA2:",eta_2)
                        
                        #total_loss = 1/2 * (total_loss.sum())
                        
                        #print(total_loss)
                                                
                        #total_loss.backward()

                        #optimizer.step()
                        
                        ranks = [1]*3
                        optimizer.step(obj_func, ranks, None)

                        # projection
                        with torch.no_grad():
                            taus.clamp_(0, 1)
                            #eta_1.clamp_(0, 1)
                            #eta_2.clamp_(0, 1)
                            #eta3=1-eta_1-eta_2
                            
                        #print(taus)
                        
                        #print ("Lstsq loss:", obj_lstsq.item(), "Reg loss:", obj_reg.item(), "Mll Loss:", obj_mll.item())
                        #print ("Lstsq loss:", obj_lstsq.item(), "Mll Loss:", obj_mll.item())
                        #print ("Reg loss:", obj_reg.item(), "Mll Loss:", obj_mll.item())

                        #scheduler.step(obj_func.item())
                        
                        
                        if(total_loss.item()<loss_min):
                            loss_min=copy.deepcopy(total_loss.item())
                            taus_best=copy.deepcopy(taus)
                        

                        early_stopping(total_loss.item())

                        if early_stopping.early_stop:    
                            break

                    print("i:",i)        
                    
                    print ("Lstsq loss:", obj_lstsq.item(), "Reg loss:", obj_reg.item(), "MLL Loss:", obj_mll.item()) 
                    #print ("X:", p_A * x)
                    return taus_best.detach().numpy() 

                # Projection to acceptable states: Compute the disaggregated lin. act. values
                def solve_X(p_A, tau, b_orig, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0):    
                    # Define and solve the CVXPY problem.

                    (round_num, user_num) = p_A.shape

                    # weight_pos, neg, means: round_num x user_num
                    weight_pos = p_A * np.tile((2 * pos_var_view) ** -1, (user_num, 1)).T
                    weight_neg = p_A * np.tile((2 * neg_var_view) ** -1, (user_num, 1)).T

                    bias_pos = p_A * np.tile(np.log((2 * math.pi * pos_var_view) ** -0.5), (user_num, 1)).T
                    bias_neg = p_A * np.tile(np.log((2 * math.pi * neg_var_view) ** -0.5), (user_num, 1)).T

                    pos_means = p_A * np.tile(pos_mean_view, (user_num, 1)).T
                    neg_means = p_A * np.tile(neg_mean_view, (user_num, 1)).T

                    # tau: size of user_num
                    taus = p_A * np.tile(tau, (round_num, 1))

                    # x : round_num x user_num
                    x = cp.Variable((round_num, user_num))

                    objective = cp.Maximize(
                        cp.sum(cp.multiply(taus, cp.multiply(-cp.square(x - pos_means), weight_pos) + bias_pos) 
                            + cp.multiply(1 - taus, cp.multiply(-cp.square(x - neg_means), weight_neg) + bias_neg))
                        - reg_mu * cp.sum_squares(cp.multiply(p_A, x))
                        )

                    # least square constraint (it should be a valid solution of b)
                    constraints = [cp.abs(b_orig - cp.diag(x @ p_A.T)) <= 0.0001]
                    prob = cp.Problem(objective, constraints)
                    prob.solve() 
                    return x.value

                    # X : (rounds x user_num)

                assert X_real.shape == A.shape and len(ground_truth) == X_real.shape[1], "Dimension mismatch"
                # attacked_user_num = int(sum(ground_truth))

                user_num = len(ground_truth)
                round_num = A.shape[0]

                #b = np.array([np.dot(X_real[i],A[i]) for i in range(round_num)])

                ROUNDS = range(START_ROUND, round_num, INTERVAL_ROUND)

                agg_preds = defaultdict(list)
                agg_means = defaultdict(list)
                agg_f1_scores = defaultdict(list)

                # 1 - overlapping coefficient (OVL) as weights
                errors = []
                for pmean, pvar, nmean, nvar in zip(pos_mean, pos_var, neg_mean, neg_var):
                    errors.append(NormalDist(mu=pmean, sigma=pvar**0.5).overlap(NormalDist(mu=nmean, sigma=nvar**0.5)))

                weights = 1 - np.array(errors)

                for obs_num in ROUNDS:
                    b_view = b[:obs_num].copy()
                    b_grads_view = b_grads[:obs_num].copy()
                    A_view = A[:obs_num,:].copy()
                    weights_view = weights[:obs_num].copy()

                    pos_var_view = pos_var[:obs_num]
                    neg_var_view = neg_var[:obs_num]
                    pos_mean_view = pos_mean[:obs_num]
                    neg_mean_view = neg_mean[:obs_num]

                     # id of rounds where the client participates
                    usr_rounds = [np.where(A_view[:, user] > 0.0)[0] for user in range(user_num)]

                    print (f"Rounds: {obs_num}/{round_num}")

                    # Solution with OLS
                    omeans = lstsq(A_view, b_view)[0]

                    agg_means["OLS"].append((usr_rounds, omeans))
                    agg_preds["OLS_simple"].append(omeans>0.0)
                    agg_preds["OLS"].append(get_pred(agg_means["OLS"], pos_var_view, pos_mean_view, neg_var_view, neg_mean_view, usr_rounds, method='simple') > 0.5)

                    # Solution with WLS
                    wmeans = sm.WLS(b_view, A_view, weights_view).fit().params

                    agg_means["WLS"].append((usr_rounds, wmeans))
                    agg_preds["WLS_simple"].append(wmeans>0.0)
                    agg_preds["WLS"].append(get_pred(agg_means["WLS"], pos_var_view, pos_mean_view, neg_var_view, neg_mean_view, usr_rounds, method='simple') > 0.5)

                    # Solution with Ridge Regression (WLS + L2 regularization)
                    rmeans = ridge_regression(A_view, b_view, weights_view, np.full(user_num, REG_PARAM_RIDGE))

                    agg_means["REG"].append((usr_rounds, rmeans))
                    agg_preds["REG_simple"].append(rmeans>0.0)
                    pred_reg = get_pred(agg_means["REG"], pos_var_view, pos_mean_view, neg_var_view, neg_mean_view, usr_rounds, method='simple')
                    agg_preds["REG"].append(pred_reg > 0.5)

                    # Baseline: expC
                    # the attacker model trained in round i uses the global model trained in round (i - 1) as target 
                    # (the init target model with random weights is not saved!)
                    
                    # we need only the sum of model updates, not the sum of whole gradients
                    for j in range(obs_num): 
                        
                        if args.dataset == "fmnist":
                            ref_model = CNNMnist(args=args)
                        elif args.dataset == "mnist":
                            ref_model = CNNMnist(args=args)
                        elif args.dataset == "cifar":
                            ref_model = CNNCifar(args=args)
                        
                        # Since b_grads is the sum of models and not model updates !!!
                        ref_model.load_state_dict(torch.load(prefix + f"ref_model_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{j}_{run}.pt", map_location=torch.device('cpu')))
                        b_grads_view[j] = (b_grads_view[j] - A[j].sum() * flatten_params(ref_model).numpy())

                    # x_init has a size of (user_num x model_params)
                    x_init = lstsq(A_view, b_grads_view)[0]

                    # CAUTION: init ref_model is missing, we cannot compute in round 0!!!
                    preds_baseline = np.zeros((obs_num, user_num))
                    for j in tqdm(range(0, obs_num)):
                        if args.dataset == "fmnist":
                            ref_model = CNNMnist(args=args)
                        elif args.dataset == "mnist":
                            ref_model = CNNMnist(args=args)
                        elif args.dataset == "cifar":
                            ref_model = CNNCifar(args=args)

                        #ref_model.load_state_dict(torch.load(prefix + f"ref_model_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{j}_{run}.pt", map_location=torch.device('cpu')))
                        total_params = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
                        scaler = CPU_Unpickler(open(prefix + f"scaler_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{j}_{run}.pkl",'rb')).load()

                        attacker_model = LogisticRegression(input_dim=total_params, output_dim=1)
                        attacker_model.load_state_dict(torch.load(prefix + f"attacker_model_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{j}_{run}.pt", map_location=torch.device('cpu')))

                        mia_features = [torch.from_numpy(x).float() for x in x_init]
                        mia = MIA(args=args, scaler=scaler)
                        correct, predictions, output = mia.predict(attacker_model, torch.stack(mia_features), torch.tensor(ground_truth), scale=True)

                        preds_baseline[j] = predictions.detach().cpu().numpy()

                    pred_baseline = (preds_baseline.sum(axis=0) / preds_baseline.shape[0]) > 0.5
                    agg_preds["Baseline"].append(pred_baseline)

                    # EM solution

                    '''
                    preds_em = np.zeros((EXP_NUM, user_num))
                    #print ("Pred reg:", pred_reg)
                    preds_em[0] = pred_reg.copy() # latent variable to be estimated in EM, init EM solution to REG  

                    for i in range(1, EXP_NUM):

                        X = solve_X(A_view, preds_em[i - 1], b_view, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0)  

                        for user in range(user_num):         
                            if len(usr_rounds[user]) == 0:
                                continue

                            preds_em[i][user] = solve_tau(X[usr_rounds[user], user], pos_mean_view[usr_rounds[user]], 
                                pos_var_view[usr_rounds[user]], neg_mean_view[usr_rounds[user]], neg_var_view[usr_rounds[user]])

                        pred_em = preds_em[-1]
                        print (f"Round {i}, diff: {abs(preds_em[i] - preds_em[i-1]).sum()}")
                        #print (preds_em[i][:10])
                        #print (preds_em[i-1][:10])
                        #print (abs(preds_em[i-1] - preds_em[i]))

                    #print ("Ground truth:", ground_truth)
                    '''

                    pred_em = solve_complete(A_view, weights_view, rmeans, b_view, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0)
                    #rrmeans = solve_complete2(A_view, b_view, weights_view, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0)
                    #pred_em = solve_complete2(A_view, b_view, weights_view, pos_mean_view, pos_var_view, neg_mean_view, neg_var_view, reg_mu = 0)
                    #agg_means["REG2"].append((usr_rounds, rrmeans))
                    #pred_em = get_pred(agg_means["REG2"], pos_var_view, pos_mean_view, neg_var_view, neg_mean_view, usr_rounds)

                    print (pred_reg.copy())
                    print ("solve_complete pred:", pred_em)

                    agg_preds["EM"].append(pred_em > 0.5)
                    print ("Ground truth:", np.nonzero(ground_truth))

                    preds = {method: agg_preds[method][-1] for method in agg_preds}   
                    report(ground_truth, preds)

                    print ("\n-> Aggregated results:")
                    preds = {method: agg_decide(agg_preds[method], weights_view) for method in agg_preds}
                    for method in preds:
                        agg_f1_scores[method].append(f1_score(ground_truth, preds[method]))  

                    report(ground_truth, preds)

                # Plot results

                # F1 score
                f1_scores =  {method : [f1_score(ground_truth, x) for x in agg_preds[method]] for method in agg_preds}

                list_stack_f1_score.append(copy.deepcopy(f1_scores))
                list_stack_f1_score_agg.append(copy.deepcopy(agg_f1_scores))
                
                
                
            pickle.dump(list_stack_f1_score,open(f"./Results_CCS/list_stack_f1_score-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{REG_PARAM_RIDGE}_{REG_PARAM_EM}_"+str(attack)+"_check.pkl",'wb'))
            pickle.dump(list_stack_f1_score_agg,open(f"./Results_CCS/list_stack_f1_score_agg-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{REG_PARAM_RIDGE}_{REG_PARAM_EM}_"+str(attack)+"_check.pkl",'wb'))  
                
                
              

            f1_scores = defaultdict(list)
            agg_f1_scores = defaultdict(list)
            
            f1_scores_std = defaultdict(list)
            agg_f1_scores_std = defaultdict(list)

            for run in range(1,runs+1):
                dic_tmp=list_stack_f1_score[run-1]
                dic_tmp_agg=list_stack_f1_score_agg[run-1]
                for key in dic_tmp:
                    f1_scores[key].append(dic_tmp[key])
                    agg_f1_scores[key].append(dic_tmp_agg[key])

            for key in f1_scores:
                #print("SHAPE:",np.array(f1_scores[key]).shape)
                f1_scores[key]=np.mean(np.array(f1_scores[key]),axis=0)
                f1_scores_std[key]=np.std(np.array(f1_scores[key]),axis=0)
                agg_f1_scores[key]=np.mean(np.array(agg_f1_scores[key]),axis=0)
                agg_f1_scores_std[key]=np.std(np.array(agg_f1_scores[key]),axis=0)
                #print("LENGTH_1:",len(f1_scores[key]))
                #print("LENGTH_2:",len(agg_f1_scores[key]))

            #np.save("/home/raouf/Pytorch/Fine_Tunning/f1_scores_")

            #print("Length:",len(list(f1_scores.values())))

            pickle.dump(f1_scores,open(f"./Results_CCS/f1_scores-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{REG_PARAM_RIDGE}_{REG_PARAM_EM}_"+str(attack)+"_check.pkl",'wb'))
            pickle.dump(agg_f1_scores,open(f"./Results_CCS/agg_f1_scores-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{REG_PARAM_RIDGE}_{REG_PARAM_EM}_"+str(attack)+"_check.pkl",'wb'))

            pickle.dump(f1_scores_std,open(f"./Results_CCS/f1_scores_std-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{REG_PARAM_RIDGE}_{REG_PARAM_EM}_"+str(attack)+"_check.pkl",'wb'))
            pickle.dump(agg_f1_scores_std,open(f"./Results_CCS/agg_f1_scores_std-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{REG_PARAM_RIDGE}_{REG_PARAM_EM}_"+str(attack)+"_check.pkl",'wb'))

            """







            labels, data = zip(*f1_scores.items())

            # Reconstruction accuracy
            p = sns.lineplot(data=data, dashes=False, markers=True)

            p.set_xlabel("Epoch", fontsize = 15)
            p.set_ylabel("F1-score", fontsize = 15)
            plt.xlim(0)
            plt.ylim(0,1)
            p.set_xticks(range(len(ROUNDS))) # <--- set the ticks first
            p.set_xticklabels(["%d" % x for x in ROUNDS])
            for label in p.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)

            plt.legend(labels=labels)
            #plt.legend(labels=['Gradient disagg', 'Membership disagg (OLS)', 'Membership disagg (Ridge Reg.)', 'Membership disagg (Sampling)'])

            plt.savefig(f'f1-score-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}.png')

            plt.clf()

            # F1 score for AGG
            labels, data = zip(*agg_f1_scores.items())

            # Reconstruction accuracy
            p = sns.lineplot(data=data, dashes=False, markers=True)

            p.set_xlabel("Epoch", fontsize = 15)
            p.set_ylabel("F1-score", fontsize = 15)
            plt.xlim(0)
            plt.ylim(0,1)
            p.set_xticks(range(len(ROUNDS))) # <--- set the ticks first
            p.set_xticklabels(["%d" % x for x in ROUNDS])
            for label in p.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)

            plt.legend(labels=labels, loc='lower right')
            #plt.legend(labels=['Gradient disagg', 'Membership disagg (OLS)', 'Membership disagg (Ridge Reg.)', 'Membership disagg (Sampling)'])

            plt.savefig(f'agg-f1-score-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}.png')

            plt.clf()


            '''
            # Model accuracy
            p = sns.lineplot(data=[global_test_acc, mia_test_acc_orig], dashes=False, markers=True)

            p.set_xlabel("Epoch", fontsize = 15)
            p.set_ylabel("Accuracy", fontsize = 15)
            plt.xlim(0)
            plt.ylim(0,1)
            #p.set_xticks(range(args.epochs)) # <--- set the ticks first
            #p.set_xticklabels(["%d" % x for x in range(args.epochs)])
            #for label in p.xaxis.get_ticklabels()[::50]:
            #    label.set_visible(False)

            plt.legend(labels=['Target model', 'Membership inference'])
            plt.savefig(f'model-mia-acc-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}.png')

            plt.clf()
            '''

            # Plot the distribution of lin. act. values (original vs. reconstructed) for each client

            attacked_users = np.where(ground_truth > 0)[0]

            px = 1.0 / plt.rcParams['figure.dpi']  # pixel in inches
            for user in range(user_num):
                fig, (ax1, ax2) = plt.subplots(1,2,figsize=(2.5*600*px,600*px))

                X_usr_orig = X_real[usr_rounds[user], user]
                ax = ax1
                ax.set_xlim(-30,30)
                fig.suptitle("Client %d, Groundtruth: %s, Round: %d" % (user, user in attacked_users, obs_num))
                ax.set_title("Original")
                sns.histplot(X_usr_orig, stat='probability', ax=ax, kde=True, bins=np.arange(-30, 30))
                # Draw vertical line at the reconstructed values 
                ax.axvline(X_usr_orig.mean(), color="purple")
                ax.axvline(rmeans[user], color="blue")
                ax.axvline(omeans[user], color="green")

                # Plot the solution of the last experiment of EM
                ax = ax2
                ax.set_xlim(-30,30)
                #ax.set_title("EM Decision: %s (%.2f), OLS decision: %s (%.2f), WLS decision: %s (%.2f)" % 
                #    (pred_em_v2[user], conf_em[user], pred_ols[user], conf_ols[user], pred_wls[user], conf_wls[user]))
                ax.set_title("A sampled solution")
                X_usr = X[usr_rounds[user], user]
                sns.histplot(X_usr, stat='probability', ax=ax, kde=True, bins=np.arange(-30, 30))
                # Draw vertical line at the reconstructed value
                ax.axvline(rmeans[user], color="blue")
                #ax.axvline(wmeans[user], color="red")
                ax.axvline(X_usr.mean(), color="red")
                ax.axvline(X_usr_orig.mean(), color="purple")

                fig.savefig(f"client_{user}_{user in attacked_users}.png")
                plt.close(fig)
                plt.clf()

            """