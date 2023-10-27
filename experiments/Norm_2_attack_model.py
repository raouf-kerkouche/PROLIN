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
import itertools

from optionsfashionmnist import args_parser
import statsmodels.api as sm
from tqdm import tqdm
import torch
import cvxpy as cp
#from MIA5test import MIA
from MIA5testbaslin import MIA
from utils import flatten_params, norm_2
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LogisticRegression
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
import math
from statistics import NormalDist
import random
import io
import copy
from mtadam import MTAdam
from matplotlib import rc,rcParams


# choose style for plots
sns.set_style("darkgrid")

rc('font', weight='bold')


plt.rcParams.update({
    #'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
    'font.size' : 15,                   # Set font size to 11pt
    'axes.labelsize': 15,               # -> axis labels
    'legend.fontsize': 15,              # -> legends
    'xtick.labelsize':15,
    'ytick.labelsize':15,
    'font.family': 'fontenc',
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',
    'text.usetex': True,
    'axes.titlesize': 15.0,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage[T1]{fontenc}'
        # ... more packages if needed
    )
})


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


list_attacks=["aga","inv","mia"]

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
        
        if(attack=="mia"):
            prefix = "./Results_for_paper/Presentation/"
        else:
            prefix = "./Results_for_paper/Presentation_"+str(attack)+"/"
        
        
        for sgd_it in list_local_sgd_it:
            
            args.local_sgd_it=copy.deepcopy(sgd_it)


            list_stack_norm_2=[]

            # For Raouf's script
            runs=3

            for run in range(1,runs+1):
                
                print("Attack:",attack)
                print("Dataset:",dataset)
                print("local_sgd_it:",args.local_sgd_it)
                print("run:",run)

                X_real = np.load(prefix + f"X_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                A = np.load(prefix + f"A_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                b = np.load(prefix + f"b_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                b_grads = np.load(prefix + f"b_grads_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                ground_truth = np.load(prefix + f"ground_truth_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")

                global_test_loss = np.load(prefix + f"train_loss_test_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy") 
                global_test_acc = np.load(prefix + f"train_accuracy_test_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                pos_var = np.load(prefix + f"l_var_pos_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                neg_var = np.load(prefix + f"l_var_neg_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")

                pos_mean = np.load(prefix + f"l_mean_pos_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                neg_mean = np.load(prefix + f"l_mean_neg_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")


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
                

                def report(ground_truth, preds):

                    msg = "*** F1-score,"
                    for method in preds:
                        msg += " " + method + ": %.4f;" % f1_score(ground_truth, preds[method])

                    print (msg, "\n")

                    for method in preds:
                        tn, fp, fn, tp = confusion_matrix(ground_truth, preds[method]).ravel()
                        print (f"{method}; FP = {fp}, FN = {fn}")

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
                
                list_norm_2=[]

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

                    # Baseline: expC
                    # the attacker model trained in round i uses the global model trained in round (i - 1) as target 
                    # (the init target model with random weights is not saved!)
                    

                    weights_alpha_attack_model=torch.load(prefix + f"attacker_model_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{obs_num}_{run}.pt", map_location=torch.device('cpu'))

                    norm_2_tmp=norm_2(weights_alpha_attack_model)

                    list_norm_2.append(norm_2_tmp.item())
                    
                list_stack_norm_2.append(np.array(list_norm_2))
                
            list_norm_2=np.mean(np.array(list_stack_norm_2).reshape(3,list_stack_norm_2[0].shape[0]),axis=0)

            START_ROUND = 10
            INTERVAL_ROUND = 10 # perform reconstruction in every INTERVAL_ROUND

            #round_num = len(list_norm_2)
            ROUNDS = np.arange(10, 300, 10)

            marker = itertools.cycle(('o', '+', 'x', '*', '.', 'X'))

            plt.plot(ROUNDS, list_norm_2, label="Norm_2",  marker = next(marker))

            #p = sns.lineplot(data=data, dashes=False, markers=True)
            ax = plt.gca()

            plt.xlim(0)
            #plt.ylim(0,1)

            ax = plt.gca()
            ax.set_xlabel("Round", fontsize = 17, labelpad=2)
            ax.tick_params(axis='both', which='major', labelsize=14)

            plt.legend(loc='upper right', fontsize=13)

            #plt.savefig(f'./Results_CCS/Norms_figures/Norm_2_attack_model_weights-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{attack}.png')
            
            
            plt.tight_layout()

            #plt.savefig('MIA.pdf',bbox_inches='tight')
            
            plt.savefig(f'./Results_CCS/Norms_figures/Norm_2_attack_model_weights-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{attack}.pdf')

            plt.clf()