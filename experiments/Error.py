from statistics import NormalDist
import matplotlib 
import itertools  
import numpy as np
import matplotlib.pyplot as plt
import copy
#from options import args_parser




# https://blog.finxter.com/matplotlib-subplots/
# https://www.edureka.co/community/68584/how-to-make-single-legend-for-many-subplots-with-matplotlib
# https://www.codegrepper.com/code-examples/whatever/matplotlib+one+legend+for+all+subplots
# https://www.delftstack.com/fr/howto/matplotlib/how-to-make-a-single-legend-for-all-subplots-in-matplotlib/
# color: https://www.geeksforgeeks.org/how-to-create-a-single-legend-for-all-subplots-in-matplotlib/

# Import necessary modules and (optionally) set Seaborn style
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
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






list_attacks=["aga","inv","mia"]

list_datasets=["fmnist","cifar","mnist"]

for attack in list_attacks:
    for dataset in list_datasets:
        
        print(dataset,attack)
        
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


            list_stack_error=[]

            # For Raouf's script
            runs=3

            for run in range(1,runs+1):
                
                
                pos_var = np.load(prefix + f"l_var_pos_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                neg_var = np.load(prefix + f"l_var_neg_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")

                pos_mean = np.load(prefix + f"l_mean_pos_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")
                neg_mean = np.load(prefix + f"l_mean_neg_{extra}{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.local_bs}_{args.local_sgd_it}_{args.attacker_model_reg}_{args.penality}_{args.prop}_weights_{run}.npy")


                errors = []
                for pmean, pvar, nmean, nvar in zip(pos_mean, pos_var, neg_mean, neg_var):
                    errors.append(NormalDist(mu=pmean, sigma=pvar**0.5).overlap(NormalDist(mu=nmean, sigma=nvar**0.5)))

                errors = np.array(errors)
                
                list_stack_error.append(errors)
                
            print("list_stack_error[0].shape[0]:",list_stack_error[0].shape[0])
            print("shape:",np.array(list_stack_error).shape)

            errors=np.mean(np.array(list_stack_error).reshape(3,list_stack_error[0].shape[0]),axis=0)

            START_ROUND = 10
            INTERVAL_ROUND = 10 # perform reconstruction in every INTERVAL_ROUND

            round_num = len(errors)
            ROUNDS = np.arange(START_ROUND, round_num, INTERVAL_ROUND)

            marker = itertools.cycle(('o', '+', 'x', '*', '.', 'X'))

            plt.plot(ROUNDS, errors[ROUNDS], label="OVL",  marker = next(marker))

            #p = sns.lineplot(data=data, dashes=False, markers=True)
            ax = plt.gca()

            plt.xlim(0)
            plt.ylim(0,0.6)

            ax = plt.gca()
            ax.set_xlabel("Round", fontsize = 17, labelpad=2)
            ax.tick_params(axis='both', which='major', labelsize=14)

            plt.legend(loc='upper left', fontsize=13)

            #plt.savefig(f'./Results_CCS/Errors_figures/OVL-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{attack}.png')
            
            
            plt.tight_layout()

            #plt.savefig('Impact_N.pdf',bbox_inches='tight')
            
            plt.savefig(f'./Results_CCS/Errors_figures/OVL-{args.dataset}_{args.epochs}_{args.num_users}_{args.frac}_{args.local_bs * args.local_sgd_it}_{args.mia_sample_num}_{attack}.pdf')

            plt.clf()