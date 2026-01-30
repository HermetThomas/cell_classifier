print('\nLoading packages')
import pandas as pd
import numpy as np
import math
from scipy import interpolate
import scanpy as sc; sc.settings.verbosity = 0
from scanpy.external.pp import hashsolo
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import os
import shutil
import anndata as ad
import random
import argparse
import time
from itertools import product
from functools import reduce
from scipy.sparse import issparse
from bioservices import KEGG
from CRISPR_functions import *

import importlib

#Choose :
    # Projection you desire : l1,1 / l1,infinity
    # Network you desire : netBio [Barlaud M et al, 2021] / LeNet [LeCun Y. et al, 1998]
    
#network_name = "proj_l11_netBio"
#network_name = "proj_l11_LeNet"
network_name = "proj_infinity_netBio"
#network_name = "proj_infinity_LeNet"

if 'infinity' in network_name :
    default_ETA = 0.25; list_ETA = [0.25, 0.5, 1, 2]
elif 'l11' in network_name :
    default_ETA = 25; list_ETA = [10, 25, 50, 100]

n_seeds = 1
seeds = [random.randint(1,100) for _ in range(n_seeds)]

#Hashsolo priors to use for [Negatives, Singlets, Doublets]
#Default parameters = [0.01, 0.8, 0.9]
priors=[0.01,0.8,0.19]

module = importlib.import_module("autoencoder.Run_SSAE_alldata")
run_SSAE = getattr(module, network_name)

start = time.time()

def CRISPR_demux() :

    """
    Load, normalize, merge and demultiplex CRISPR counts matrices
    Find discriminant features between perturbed and control with a Supervised AutoEncoder 

    Required inputs :
    
        - adata object containing cells to classify
        - Conditions to test
        - Condition to use as control

    Outputs :
    - Classification for each cell
    - AutoEncoder scores (Precision, Recall, F1...)
    - Ranking of most discriminant features
    - Pathways associated to most discriminant features
    """    

    parser = argparse.ArgumentParser(
        prog = 'demux', 
        formatter_class=argparse.MetavarTypeHelpFormatter,
        description = 'Discriminant genes discovery by AutoEncoder')
    
    parser.add_argument('-adata', type = file_path, help = 'Path/to/adata.h5ad', required = True)
    parser.add_argument('-obs', type = str, help = 'Name of .obs column to use for classification') 
    parser.add_argument('-tests', nargs = '+', type =str, help = 'Name of conditions to test', default = None)
    parser.add_argument('-neg', type =str, help = 'Name of negative control', default = None)
    parser.add_argument('-runs', type = int, help = 'Number of random samplings and AutoEncoder runs to perform', default = 1)
    parser.add_argument('-eta', action='store_true', help = 'Test multiple eta values to get highest accuracy', default = False)
    parser.add_argument('-pathways', action='store_true', help = 'Add -pathways if you want to find pathways associated to the top genes from the KEGG database', default = False)
    
    args = parser.parse_args()

    adata = args.adata
    
    results_dir = create_results_folder()

    print('\nLoading counts matrix')
    #Load the counts matrix as an AnnData object
    counts_adata = sc.read_h5ad(adata)

    tests = args.tests
    neg = args.neg
    column = args.obs

    #Make the matrix dense if it is sparse
    gene_counts_array = counts_adata.X.A if issparse(counts_adata.X) else counts_adata.X
    gene_counts = gene_counts_array.sum(axis=0)
    gene_counts_array_1d = gene_counts.A1 if issparse(gene_counts) else gene_counts.flatten()

    #make a Pandas dataframe containing the genes names and their counts sum
    gene_expression_df = pd.DataFrame({
        'Gene': counts_adata.var_names,
        'ExpressionSum': gene_counts_array_1d
    })
    
    #Sort the genes by descending counts sum
    sorted_genes = gene_expression_df.sort_values(by='ExpressionSum', ascending=False)
    #Store the names of the 10,000 most expressed genes in a list 
    top_10k_genes = sorted_genes.head(10000)['Gene'].tolist()
    #Select the rows corresponding to the 10,000 most expressed genes from the counts matrix
    top10k =counts_adata[:, counts_adata.var_names.isin(top_10k_genes)].copy()    

    #dictionary that contains the most differentially expressed genes for each condition and their rank
    allresults = {}
    #dictionary that contains the log(fold change) of the studied genes between the perturbed cells and control cells 
    expression = {}

    for condition in tests :
        #create a list for each condition in each dictionary
        allresults[condition] = []
        expression[condition] = []
    
    #Create a dataframe in which the accuracies by eta will be stored for each condition
    acc_df = pd.DataFrame(columns=[f'eta_{ETA}' for ETA in list_ETA])
  
    #Take the cells with non-targeting gRNA as negative control
    Neg = top10k[top10k.obs[column] == neg].to_df().T
    #Add the label 0 for 'control cell' for each cell in the dataframe
    Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
    Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

    if args.eta :
        #Create a dataframe in which the accuracies by eta will be stored for each condition
        acc_df = pd.DataFrame(columns=[f'eta_{ETA}' for ETA in list_ETA])
        classif_df = pd.DataFrame(columns=['condition', 'True_positive', 'False_negative'])
        best_ETA = pd.DataFrame(columns=['ETA', 'accuracy'])

        for condition in tests : 
            condition_data = top10k[top10k.obs[column] == condition].to_df().T
            #Add the label 1 for 'perturbed' for each cell in the dataframe
            condition_data.loc['Label'] = pd.Series(np.ones(len(condition_data.columns)), index=condition_data.columns)
            condition_data = pd.concat([condition_data.loc[['Label']], condition_data.drop('Label')])
            if len(condition_data.columns) >= 10 and len(Neg.columns) >= 10:    
                #Take a random sample of both datasets to have a matching number of control and perturbed cells
                Neg_cut = Neg.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=0)
                condition_data_cut = condition_data.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=0)
                #Concatenate the negative control and the perturbed cells counts
                dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                
                #create a list that will contain the accuracy for each eta parameter tested
                list_acc = []
                for ETA in list_ETA :
                    print(f'\nProcessing data for {condition},  ETA = {ETA}\n')

                    run_SSAE(condition, dataset, results_dir, eta=ETA, seeds=seeds)
                    
                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                    perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                    control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                    #Get the number of true perturbed and false perturbed cells from the classification results

                    if control and perturbed :
                        Neg_cut = Neg.loc[:, control]
                        condition_data_cut = condition_data.loc[:, perturbed]
                        Neg_cut = Neg_cut.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                        condition_data_cut = condition_data.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                        dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                        
                        if len(dataset.columns) >=10 :
                            run_SSAE(condition, dataset, results_dir, eta=ETA, seeds=seeds)

                            #Add the accuracy of the run to the list of accuracies
                            accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                            accuracy = pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, index_col=0, sep=';').Global.loc['Mean']
                            if ETA == list_ETA[0] :
                                best_ETA.loc[condition] = [ETA, accuracy]
                            else :
                                if accuracy > best_ETA['accuracy'].loc[condition] :
                                    best_ETA.loc[condition] = [ETA, accuracy]
                            best_ETA.to_csv(f'{results_dir}/best_ETA.csv')
                        
                    else :
                        pass

                #Final run with the optimal ETA parameter 
                Neg_cut = Neg.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=0)
                condition_data_cut = condition_data.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=0)
                dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)

                run_SSAE(condition, dataset, results_dir, eta=best_ETA['ETA'].loc[condition])

                classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                #Get the number of true perturbed and false perturbed cells from the classification results

                if len(control) > 10 and len(perturbed) > 10 :
                    Neg_cut = Neg.loc[:, control]
                    condition_data_cut = condition_data.loc[:, perturbed]
                    Neg_cut = Neg_cut.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                    condition_data_cut = condition_data.sample(n=min(len(condition_data.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                    dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                    
                    run_SSAE(condition, dataset, results_dir, eta=best_ETA['ETA'].loc[condition])

                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                    distrib_classif(results_dir, condition, classif, run=2)

                if list_acc and top_genes :
                    #Concatenate the top genes dataframes 
                    final_df = features_df(top_genes)
                    #Save the dataframe after each condition to be able to check the advancement of the script
                    acc_df.to_csv(f'{results_dir}/accuracies.csv')
                else :
                    pass
            else :
                pass

        
        
        #Add a row containing the number of times each eta parameter has the max accuracy
        acc_df = pd.concat([acc_df, pd.DataFrame(acc_df.idxmax(axis=1).value_counts()).T])
        acc_df.to_csv(f'{results_dir}/accuracies.csv')
        if len(classif_dict) > 1 :
            for name, df in classif_dict.items() :
                plot_classif(results_dir, df, tests, eta=name.split('_')[-1]) 

    if args.runs>1 :

        #dictionary that contains the clasification score metrics for each run and each condition
        metrics_dict = {}
        #Same dictionary for the second run of autoencoder
        metrics_dict2 = {}

        #dictionary that contains the number of true positives and dalse negatives per condition per ETA tested
        classif_dict = {}
        #Same dictionary for the second run of autoencoder
        classif_dict2 = {}

        for run in range(args.runs) :

            classif_dict[f'run_{run}'] = pd.DataFrame(columns=['nb_cells', 'True_positive', 'False_negative'])
            metrics_dict[f'run_{run}'] = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
            classif_dict2[f'run_{run}'] = pd.DataFrame(columns=['nb_cells', 'True_positive', 'False_negative'])
            metrics_dict2[f'run_{run}'] = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])

            for condition in tests : 

                condition_data = top10k[top10k.obs[column] == condition].to_df().T
                #Add the label 1 for 'perturbded' for each cell in the dataframe
                condition_data.loc['Label'] = pd.Series(np.ones(len(condition_data.columns)), index=condition_data.columns)
                condition_data = pd.concat([condition_data.loc[['Label']], condition_data.drop('Label')])
                if len(condition_data.columns) > 10 and len(Neg.columns) > 10:    
                    #Take a random sample of both datasets to have a matching number of control and perturbed cells
                    Neg_cut = Neg.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=run)
                    condition_data_cut = condition_data.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=run)
                    #Concatenate the negative control and the perturbed cells counts
                    dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                    print(f'\First run of AutoEncoder for {condition},  run n°{run+1}\n')

                    if args.eta :
                        run_SSAE(condition, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                    else :
                        run_SSAE(condition, dataset, results_dir, eta=default_ETA, seeds=seeds)

                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                    perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                    control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                    #Plot the distribution of perturbed and control cells from the classification results
                    distrib_classif(results_dir, condition, classif, run=1)
                    tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(condition_data_cut.columns) * 100
                    fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(condition_data_cut.columns) * 100
                    classif_dict[f'run_{run}'].loc[condition] = [len(dataset.columns), tp, fn]

                    accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                    metrics_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'auctest' in file), None)

                    metrics_dict[f'run_{run}'].loc[condition] = [len(dataset.columns), pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, sep=';', 
                                                index_col=0)['Global'].loc['Mean']] + list(pd.read_csv(f'{results_dir}/{condition}/{metrics_file}', header=0, sep=';', 
                                                                            index_col=0).loc['Mean', ['AUC', 'Precision', 'Recall', 'F1 score']])

                    if len(control) >= 5 and len(perturbed) >= 5 :
                        Neg_cut = Neg.loc[:, control]
                        condition_data_cut = condition_data.loc[:, perturbed]
                        Neg_cut = Neg_cut.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1, 50))
                        condition_data_cut = condition_data_cut.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1, 50))
                        
                        #create a dataframe that contains the expression of each gene in perturbed and control cells for the current run
                        expression_df = pd.DataFrame({'Gene' : Neg_cut.index.to_list()[1:], 
                                        'Perturbed_expression' : condition_data_cut.iloc[1:, :].mean(axis=1), 
                                        'Control_expression' : Neg_cut.iloc[1:, :].mean(axis=1)})
                        #Add a column that contains the log fold change of each gene between control and perturbed cells
                        expression_df['log2_FC'] = np.log2(expression_df['Perturbed_expression'] / expression_df['Control_expression'])
                        expression_df.index = expression_df['Gene']
                        expression[condition].append(expression_df)

                        dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                        
                        print(f'\nSecond run of AutoEncoder for {condition},  run n°{run+1}\n')
                        if args.eta :
                            run_SSAE(condition, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                        else :
                            run_SSAE(condition, dataset, results_dir, eta=default_ETA)

                        #Store the most differentially expressed genes, their weight and their rank in a dataframe
                        top_genes_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'Mean_Captum' in file), None)
                        scores = pd.read_csv(f'{results_dir}/{condition}/{top_genes_file}', header=0, sep=';')[['Features', 'Mean']]
                        scores['Rank'] = scores.index + 1
                        #Add the dataframe to the list of dataframes corresponding to the current condition
                        allresults[condition].append(scores)

                        classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', sep=';', header=0)
                        #Plot the distribution of perturbed and control cells from the classification results
                        distrib_classif(results_dir, condition, classif, run=2)
                        tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(condition_data_cut.columns) * 100
                        fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(condition_data_cut.columns) * 100
                        classif_dict2[f'run_{run}'].loc[condition] = [len(dataset.columns), tp, fn]

                        accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                        metrics_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'auctest' in file), None)

                        metrics_dict2[f'run_{run}'].loc[condition] = [len(dataset.columns), pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, sep=';', 
                                                    index_col=0)['Global'].loc['Mean']] + list(pd.read_csv(f'{results_dir}/{condition}/{metrics_file}', header=0, sep=';', 
                                                                                index_col=0).loc['Mean', ['AUC', 'Precision', 'Recall', 'F1 score']])
                                                    
                        metrics_dict2[f'run_{run}'].to_csv(f'{results_dir}/{condition}/run_{run}.csv', index = True)
                    
                    else:
                        pass
        
        #Dataframes that will contain the mean of the classification results for each each condition 
        results_df = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])

        # Compiling Classification results for 
        for condition in tqdm(tests) :

            #For each condition, make a dataframe that contains the mean weight, weight std, mean rank, rank std and log2 fold change of each gene

            list_nbcells, list_acc, list_auc, list_precision, list_recall, list_f1, list_tp, list_fn = [], [], [], [], [], [], [], []

            for run in range(args.runs) :
                
                if condition in metrics_dict[f'run_{run}'].index :
                    nbcells, acc, auc, precision, recall, f1 = metrics_dict[f'run_{run}'].loc[condition].values
                    tp, fn = classif_dict[f'run_{run}'].loc[condition, ['True_positive', 'False_negative']]
                    list_nbcells.append(nbcells)
                    list_acc.append(acc)
                    list_auc.append(auc)
                    list_precision.append(precision)
                    list_recall.append(recall)
                    list_f1.append(f1)
                    list_tp.append(tp)
                    list_fn.append(fn)
                else :
                    pass
            
            if list_nbcells and  list_acc and  list_auc and  list_precision and  list_recall and  list_f1 and  list_tp and  list_fn :

                nbcells = sum(list_nbcells)/len(list_nbcells)
                acc = sum(list_acc)/len(list_acc)
                auc = sum(list_auc)/len(list_auc)
                precision = sum(list_precision)/len(list_precision)
                recall = sum(list_recall)/len(list_recall)
                f1 = sum(list_f1)/len(list_f1)
                tp = sum(list_tp)/len(list_tp)
                fn = sum(list_fn)/len(list_fn)

                results_df.loc[condition] = [nbcells, acc, auc, precision, recall, f1]
                classif_df.loc[condition] = [nbcells, tp, fn]
        
        classif_df.to_csv(f'{results_dir}/cell_classification_allruns.csv', index=True)
        results_df.to_csv(f'{results_dir}/metrics_allruns.csv', index=True)


        #Dataframes that will contain the mean of the classification results for each each condition 
        results_df2 = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df2 = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])

    
    elif args.runs == 1 : 

        results_df = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])
        results_df2 = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df2 = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])

        for condition in tests : 

            condition_data = top10k[top10k.obs[column] == condition].to_df().T
            #Add the label 1 for 'perturbded' for each cell in the dataframe
            condition_data.loc['Label'] = pd.Series(np.ones(len(condition_data.columns)), index=condition_data.columns)
            condition_data = pd.concat([condition_data.loc[['Label']], condition_data.drop('Label')])
            if len(condition_data.columns) >= 10 and len(Neg.columns) >= 10:    
                
                Neg_cut = Neg.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=42)
                condition_data_cut = condition_data.sample(n=min(len(condition_data.columns), len(Neg.columns)), axis=1, random_state=42)
                #Concatenate the negative control and the perturbed cells counts
                dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                print(f'\nProcessing data for {condition}\n')
            
                if args.eta :
                    run_SSAE(condition, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                else :
                    run_SSAE(condition, dataset, results_dir, eta=default_ETA, seeds=seeds)
                
                classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                #Plot the distribution of perturbed and control cells from the classification results
                distrib_classif(results_dir, condition, classif, run=1)
                
                tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(condition_data_cut.columns) * 100
                fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(condition_data_cut.columns) * 100
                classif_df.loc[condition] = [len(dataset.columns), tp, fn]
                classif_df.to_csv(f'{results_dir}/cell_classification_run1.csv')
                accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                metrics_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'auctest' in file), None)
                results_df.loc[condition] = [len(dataset.columns), pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, sep=';', 
                                            index_col=0)['Global'].loc['Mean']] + list(pd.read_csv(f'{results_dir}/{condition}/{metrics_file}', header=0, sep=';', 
                                                                        index_col=0).loc['Mean', ['AUC', 'Precision', 'Recall', 'F1 score']])
                results_df = results_df.sort_index()
                results_df.to_csv(f'{results_dir}/metrics_run1.csv')


                if len(control) >= 5 and len(perturbed) >= 5 :
                    Neg_cut = Neg.loc[:, control]
                    condition_data_cut = condition_data.loc[:, perturbed]
                    Neg_cut = Neg_cut.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1,10))
                    condition_data_cut = condition_data_cut.sample(n=min(len(condition_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1,10))
                    dataset = pd.concat([Neg_cut, condition_data_cut], axis=1)
                    expression_df = pd.DataFrame({'Gene' : Neg_cut.index.to_list()[1:], 
                                        'Perturbed_expression' : condition_data_cut.iloc[1:, :].mean(axis=1), 
                                        'Control_expression' : Neg_cut.iloc[1:, :].mean(axis=1)})
                    #Add a column that contains the log fold change of each gene between control and perturbed cells
                    expression_df['log2_FC'] = np.log2(expression_df['Perturbed_expression'] / expression_df['Control_expression'])
                    expression_df.index = expression_df['Gene']
                        
                    print(f'\n Second run for {condition}\n')
                    if args.eta :
                        run_SSAE(condition, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                    else :
                        run_SSAE(condition, dataset, results_dir, eta=default_ETA, seeds=seeds)

                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', sep=';', header=0)
                    #Plot the distribution of perturbed and control cells from the classification results
                    distrib_classif(results_dir, condition, classif, run=2)
                    tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(condition_data_cut.columns) * 100
                    fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(condition_data_cut.columns) * 100
                    classif_df2.loc[condition] = [len(dataset.columns), tp, fn]
                    classif_df2.to_csv(f'{results_dir}/cell_classification_run2.csv')
                    accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                    metrics_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'auctest' in file), None)

                    results_df2.loc[condition] = [len(dataset.columns), pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, sep=';', 
                                                index_col=0)['Global'].loc['Mean']] + list(pd.read_csv(f'{results_dir}/{condition}/{metrics_file}', header=0, sep=';', 
                                                                            index_col=0).loc['Mean', ['AUC', 'Precision', 'Recall', 'F1 score']])
                    results_df2 = results_df2.sort_index()
                    results_df2.to_csv(f'{results_dir}/metrics_run2.csv')
                    weight_expression(results_dir, condition, expression_df)
                    
                else :
                    pass 
    
    top_genes(results_dir, tests, pathways = args.pathways)   

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) 

if __name__ == "__main__" :
    CRISPR_demux()  
