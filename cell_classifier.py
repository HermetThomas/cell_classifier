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
    
network_name = "proj_l11_netBio"
#network_name = "proj_l11_LeNet"
#network_name = "proj_infinity_netBio"
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
        
        - Counts library
        - Negative control guides as bash inputs or python inputs
        
    Optional inputs :
        - Separate gRNA library
        - Separate HTO library
        - Number of paired libraries to concatenate and treat simultaneously
        - Priors for gRNA demultiplexing

    Outputs :
    - Classification for each cell
    - AutoEncoder scores (Precision, Recall, F1...)
    - Ranking of most discriminant features
    - Pathways associated to most discriminant features
    """    

    parser = argparse.ArgumentParser(
        prog = 'demux', 
        formatter_class=argparse.MetavarTypeHelpFormatter,
        description = 'Detects HTO and gRNA present in each cell and creates a classification dataframe')
    
    parser.add_argument('-libs', type = int, help = 'Number of libraries to treat at the same time', default = 1)
    parser.add_argument('-counts', type = dir_path, help = 'Path/to/counts/library_1/', required = True)
    parser.add_argument('-hto', type = dir_path, help = 'Path/to/hto/library_1/') 
    parser.add_argument('-neg', nargs = '+', type =str, help = 'Name of negative control gRNAs', default = None)
    parser.add_argument('-runs', type = int, help = 'Number of random samplings and AutoEncoder runs to perform', default = 1)
    parser.add_argument('-eta', action='store_true', help = 'Test multiple eta values to get highest accuracy', default = False)
    parser.add_argument('-plot', action='store_true', help = 'Add -plot to save demultiplexing distribution plots', default = False)
    parser.add_argument('-pathways', action='store_true', help = 'Add -pathways if you want to find pathways associated to the top genes from the KEGG database', default = False)
    
    args = parser.parse_args()

    cfolder = args.counts

    if args.hto :
        hfolder = args.hto
        #Find and HTO names file
        features_file = next((file for file in os.listdir(hfolder) if 'features' in file), None)
        hto_names = list(pd.read_csv(hfolder + features_file, sep = '\t', names = ['Names']).Names)
        #Remove the nucleotide sequence from the name 
        hto_names = [hto.split('-')[0] for hto in hto_names]

    if args.libs > 1 :
        #If multiple datasets, store them in a dictionary
        counts_matrices = {}
        for i in range(args.libs) :
            i+=1
            print(f"\nLoading counts matrix n°{i}")
            #Change the directory name to iteratively load the counts matrices
            folder = replace_digit(cfolder, i)
            #Find the prefix before 'matrix.mtx.gz', 'barcode.tsv.gz' and 'features.tsv.gz'
            prefix = get_prefix(folder)
            #Load the counts matrix as an AnnData object with cell barcodes in .obs and genes names in .var
            matrix = sc.read_10x_mtx(folder, prefix=prefix, cache_compression='gzip', gex_only=False)
            #Remove the nucleotide sequence from the HTO names
            matrix.obs_names = [barcode.split('-')[0] + f"-{i}" for barcode in matrix.obs_names]
            #CPM normalization on the gene counts
            #sc.pp.normalize_total(matrix, target_sum=1e6)
            matrix.X = round(matrix.X)
            
            #Add the current matrix to the matruces dictionary
            counts_matrices[f"matrix_{i}"] = matrix
        
    elif args.libs == 1 :
        print('\nLoading counts matrix')
        #Find the prefix before 'matrix.mtx.gz', 'barcode.tsv.gz' and 'features.tsv.gz'
        prefix = get_prefix(cfolder)
        #Load the counts matrix as an AnnData object
        counts_adata = sc.read_10x_mtx(cfolder, prefix=prefix, cache_compression='gzip', gex_only=False)
        #CPM normlaization on the gene counts
        #sc.pp.normalize_total(counts_adata, target_sum=1e6)
        counts_adata.X = round(counts_adata.X)
        # Remove the nucleoide sequence from the HTO names
        counts_adata.obs.index = [barcode.split('-')[0] for barcode in counts_adata.obs.index]

    neg = args.neg

    print('Loading HTO counts')
    
    if args.libs > 1 :
        i+=1
        for i in range(args.libs) :
            #If the HTO counts are in a separate matrix
            
            if args.hto :
                print(f"\nLoading HTO matrix n°{i}")

                #Change the directory name to iteratively load the HTO matrices
                folder = replace_digit(hfolder, i)
                #Find the HTO counts matrix in the directory
                matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                #Find the cell barcodes file in the directory
                barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                matrix_name = f"matrix_{i}"
                #Load the HTO counts matrix as an AnnData object 
                matrix = sc.read_mtx(folder + matrix_file).T
                #CPM normalization on HTO counts 
                #sc.pp.normalize_total(matrix, target_sum = 1e6)
                matrix.X = round(matrix.X)
                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                #Replace the barcode index to differrenciate the libraries
                matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                #Store the HTO counts as integers in .obs for demultiplexing 
                matrix.obs = matrix.to_df().astype(int)
                #Rename the columns as the HTO names
                matrix.obs.columns = hto_names
                #Remove the cells that have 0 counts for all HTO
                matrix = matrix[matrix.obs.sum(axis=1) > 0]

                #Demultiplex the HTO using Hashsolo from Scanpy (Bernstein, et al. 2020)
                hashsolo(matrix, hto_names, priors=priors)

                #Store the HTO labels in the main counts matrix
                counts_matrices[matrix_name].obs['Classif_HTO'] = matrix.obs['Classification']

            else :
                #If the HTO counts are part of the main counts matrix

                #Find the rows corresponding to HTO in the matrix
                hto_rows = find_HTOs(counts_matrices[f"matrix_{i}"])
                #Store these rows names as hto_names
                hto_names = [counts_matrices[f"matrix_{i}"].var_names[row] for row in hto_rows]

                #Store each HTO row as a column in .obs
                for row, name in zip(hto_rows, hto_names) :
                    if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                    else :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                #Remove the HTO counts from the matrix
                counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(hto_names)]
                #Remove the cells that have 0 counts for all HTO
                counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                #Demultiplex the HTO using Hashsolo from Scanpy (Bernstein, et al. 2020)
                hashsolo(counts_matrices[f"matrix_{i}"], hto_names, priors=priors)

                #Remove the columns other than the gRNA classification from .obs
                counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace = True)
        
        #Concatenate the AnnData objects as a single objet
        counts_adata = ad.concat(list(counts_matrices.values()), label = 'Library')
    
    else :
        #Same processes with a single library 

        if args.hto :
            matrix_file = next((file for file in os.listdir(hfolder) if 'matrix' in file), None)
            barcodes_file = next((file for file in os.listdir(hfolder) if 'barcodes' in file), None)
            hto_adata = sc.read_mtx(hfolder + matrix_file).T
            #sc.pp.normalize_total(hto_adata, target_sum = 1e6)
            grna_adata.X = round(hto_adata.X)
            barcodes = list(pd.read_csv(hfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
            hto_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]

            hto_adata.obs = hto_adata.to_df()
            hto_adata.obs.columns = hto_names

            hto_adata = hto_adata[hto_adata.obs.sum(axis=1) > 0]

            hashsolo(hto_adata, hto_names, priors=priors)

            counts_adata.obs['Classif_HTO'] = hto_adata.obs['Classification']  

        else :
            hto_rows = find_HTOs(counts_adata)
            hto_names = [counts_adata.var_names[row] for row in hto_rows]
            for row, name in zip(hto_rows, hto_names) :
                if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                    counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                else :
                    counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

            counts_adata = counts_adata[:, ~counts_adata.var_names.isin(hto_names)]
            counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

            hashsolo(counts_adata, hto_names, priors=priors)

            counts_adata.obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace = True)
            
    if args.plot :

        distrib_dir = f"{results_dir}/Distribution_plots/"

        #Check if the plots directory already exists
        if not os.path.exists(distrib_dir):
            os.makedirs(distrib_dir)
        else :
            print(f"Directory '{distrib_dir}' already exists.")

        #Put the HTO classification results in a Pandas dataframe for plotting 
        hashed_hto = pd.DataFrame(counts_adata.obs)

        plt.figure(figsize=(25, 9))

        if args.libs > 1 :
            sns.countplot(data = hashed_hto, x = 'Classif_HTO',
                    hue = 'Library', palette = 'Set2')
            plt.legend(title='Libraries')
        elif args.libs == 1 :
            sns.countplot(data = hashed_hto, x = 'Classif_HTO', palette = 'Set2')   
        
        plt.xlabel('HTO')
        plt.ylabel('Count')
        plt.title('Distribution of HTO classification by Hashsolo')
        plt.savefig(f"{distrib_dir}HTO_distribution.png")

    #Remove the doublets, negatives and unmapped reads
    counts_adata = counts_adata[counts_adata.obs.Classification.isin(hto_names)]
    counts_adata.obs.index = counts_adata.obs.Barcode
    #Keep only the classification in .obs
    counts_adata.obs = counts_adata.obs[['Classif_HTO']]

    print('\nSelecting 10k most expressed genes')
    #Select the 10,000 most expressed genes in the counts matrix to reduce the computational cost

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

    for condition in targets :
        #create a list for each condition in each dictionary
        allresults[condition] = []
        expression[condition] = []
    
    #Create a dataframe in which the accuracies by eta will be stored for each condition
    acc_df = pd.DataFrame(columns=[f'eta_{ETA}' for ETA in list_ETA])
    
    hto_names.remove(neg)

    #Take the cells with non-targeting gRNA as negative control
    Neg = top10k[top10k.obs['Classif_HTO'].str.contains(neg)].to_df().T
    #Add the label 0 for 'control cell' for each cell in the dataframe
    Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
    Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

    if args.eta :
        #Create a dataframe in which the accuracies by eta will be stored for each condition
        acc_df = pd.DataFrame(columns=[f'eta_{ETA}' for ETA in list_ETA])
        classif_df = pd.DataFrame(columns=['condition', 'True_positive', 'False_negative'])
        best_ETA = pd.DataFrame(columns=['ETA', 'accuracy'])

        for HTO in hto_names : 
            condition = HTO
            HTO_data = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)].to_df().T
            #Add the label 1 for 'perturbded' for each cell in the dataframe
            HTO_data.loc['Label'] = pd.Series(np.ones(len(HTO_data.columns)), index=HTO_data.columns)
            HTO_data = pd.concat([HTO_data.loc[['Label']], HTO_data.drop('Label')])
            if len(HTO_data.columns) >= 10 and len(Neg.columns) >= 10:    
                #Take a random sample of both datasets to have a matching number of control and perturbed cells
                Neg_cut = Neg.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=0)
                HTO_data_cut = HTO_data.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=0)
                #Concatenate the negative control and the perturbed cells counts
                dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                
                #create a list that will contain the accuracy for each eta parameter tested
                list_acc = []
                for ETA in list_ETA :
                    print(f'\nProcessing data for {HTO},  ETA = {ETA}\n')

                    run_SSAE(HTO, dataset, results_dir, eta=ETA, seeds=seeds)
                    
                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                    perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                    control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                    #Get the number of true perturbed and false perturbed cells from the classification results

                    if control and perturbed :
                        Neg_cut = Neg.loc[:, control]
                        HTO_data_cut = HTO_data.loc[:, perturbed]
                        Neg_cut = Neg_cut.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                        HTO_data_cut = HTO_data.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                        dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                        
                        if len(dataset.columns) >=10 :
                            run_SSAE(HTO, dataset, results_dir, eta=ETA, seeds=seeds)

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
                Neg_cut = Neg.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=0)
                HTO_data_cut = HTO_data.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=0)
                dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)

                run_SSAE(HTO, dataset, results_dir, eta=best_ETA['ETA'].loc[condition])

                classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                #Get the number of true perturbed and false perturbed cells from the classification results

                if len(control) > 10 and len(perturbed) > 10 :
                    Neg_cut = Neg.loc[:, control]
                    HTO_data_cut = HTO_data.loc[:, perturbed]
                    Neg_cut = Neg_cut.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                    HTO_data_cut = HTO_data.sample(n=min(len(HTO_data.columns), len(Neg_cut.columns)), axis=1, random_state=0)
                    dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                    
                    run_SSAE(HTO, dataset, results_dir, eta=best_ETA['ETA'].loc[condition])

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
                plot_classif(results_dir, df, hto_names, eta=name.split('_')[-1]) 

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

            for HTO in hto_names : 

                condition = HTO
                HTO_data = top10k[top10k.obs['Classif_gRNA'].str.contains(HTO)].to_df().T
                #Add the label 1 for 'perturbded' for each cell in the dataframe
                HTO_data.loc['Label'] = pd.Series(np.ones(len(HTO_data.columns)), index=HTO_data.columns)
                HTO_data = pd.concat([HTO_data.loc[['Label']], HTO_data.drop('Label')])
                if len(HTO_data.columns) > 10 and len(Neg.columns) > 10:    
                    #Take a random sample of both datasets to have a matching number of control and perturbed cells
                    Neg_cut = Neg.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=run)
                    HTO_data_cut = HTO_data.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=run)
                    #Concatenate the negative control and the perturbed cells counts
                    dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                    print(f'\First run of AutoEncoder for {HTO},  run n°{run+1}\n')

                    if args.eta :
                        run_SSAE(HTO, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                    else :
                        run_SSAE(HTO, dataset, results_dir, eta=default_ETA, seeds=seeds)

                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                    perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                    control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                    #Plot the distribution of perturbed and control cells from the classification results
                    distrib_classif(results_dir, condition, classif, run=1)
                    tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(HTO_data_cut.columns) * 100
                    fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(HTO_data_cut.columns) * 100
                    classif_dict[f'run_{run}'].loc[condition] = [len(dataset.columns), tp, fn]

                    accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                    metrics_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'auctest' in file), None)

                    metrics_dict[f'run_{run}'].loc[condition] = [len(dataset.columns), pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, sep=';', 
                                                index_col=0)['Global'].loc['Mean']] + list(pd.read_csv(f'{results_dir}/{condition}/{metrics_file}', header=0, sep=';', 
                                                                            index_col=0).loc['Mean', ['AUC', 'Precision', 'Recall', 'F1 score']])

                    if len(control) >= 5 and len(perturbed) >= 5 :
                        Neg_cut = Neg.loc[:, control]
                        HTO_data_cut = HTO_data.loc[:, perturbed]
                        Neg_cut = Neg_cut.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1, 50))
                        HTO_data_cut = HTO_data_cut.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1, 50))
                        
                        #create a dataframe that contains the expression of each gene in perturbed and control cells for the current run
                        expression_df = pd.DataFrame({'Gene' : Neg_cut.index.to_list()[1:], 
                                        'Perturbed_expression' : HTO_data_cut.iloc[1:, :].mean(axis=1), 
                                        'Control_expression' : Neg_cut.iloc[1:, :].mean(axis=1)})
                        #Add a column that contains the log fold change of each gene between control and perturbed cells
                        expression_df['log2_FC'] = np.log2(expression_df['Perturbed_expression'] / expression_df['Control_expression'])
                        expression_df.index = expression_df['Gene']
                        expression[condition].append(expression_df)

                        dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                        
                        print(f'\nSecond run of AutoEncoder for {HTO},  run n°{run+1}\n')
                        if args.eta :
                            run_SSAE(HTO, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                        else :
                            run_SSAE(HTO, dataset, results_dir, eta=default_ETA)

                        #Store the most differentially expressed genes, their weight and their rank in a dataframe
                        top_genes_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'Mean_Captum' in file), None)
                        scores = pd.read_csv(f'{results_dir}/{condition}/{top_genes_file}', header=0, sep=';')[['Features', 'Mean']]
                        scores['Rank'] = scores.index + 1
                        #Add the dataframe to the list of dataframes corresponding to the current condition
                        allresults[condition].append(scores)

                        classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', sep=';', header=0)
                        #Plot the distribution of perturbed and control cells from the classification results
                        distrib_classif(results_dir, condition, classif, run=2)
                        tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(HTO_data_cut.columns) * 100
                        fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(HTO_data_cut.columns) * 100
                        classif_dict2[f'run_{run}'].loc[condition] = [len(dataset.columns), tp, fn]

                        accuracy_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'acc' in file), None)
                        metrics_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'auctest' in file), None)

                        metrics_dict2[f'run_{run}'].loc[condition] = [len(dataset.columns), pd.read_csv(f'{results_dir}/{condition}/{accuracy_file}', header=0, sep=';', 
                                                    index_col=0)['Global'].loc['Mean']] + list(pd.read_csv(f'{results_dir}/{condition}/{metrics_file}', header=0, sep=';', 
                                                                                index_col=0).loc['Mean', ['AUC', 'Precision', 'Recall', 'F1 score']])
                    
                    else:
                        pass
        
        #Dataframes that will contain the mean of the classification results for each each condition 
        results_df = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])

        # Compiling Classification results for 
        for condition in tqdm(hto_names) :

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
        
        classif_df.to_csv(f'{results_dir}/cell_classification_run1.csv', index=True)
        results_df.to_csv(f'{results_dir}/metrics_run1.csv', index=True)


        #Dataframes that will contain the mean of the classification results for each each condition 
        results_df2 = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df2 = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])

        for condition in tqdm(targets) :

            #For each condition, make a dataframe that contains the mean weight, weight std, mean rank, rank std and log2 fold change of each gene

            list_nbcells, list_acc, list_auc, list_precision, list_recall, list_f1, list_tp, list_fn = [], [], [], [], [], [], [], []

            for run in range(args.runs) :
                
                if condition in metrics_dict2[f'run_{run}'].index :
                    nbcells, acc, auc, precision, recall, f1 = metrics_dict2[f'run_{run}'].loc[condition].values
                    tp, fn = classif_dict2[f'run_{run}'].loc[condition, ['True_positive', 'False_negative']]
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

                results_df2.loc[condition] = [nbcells, acc, auc, precision, recall, f1]
                classif_df2.loc[condition] = [nbcells, tp, fn]

            if allresults[condition] and expression[condition]:
                results1=allresults[condition][0]

                for idx, results in enumerate(allresults[condition]) :
                    results.index.name = 'Features'
                    if idx != 0 :
                        results = results.reindex(allresults[condition][0].index)                        
                    
                df = pd.DataFrame({'Gene' : results1['Features'], 
                                'Weight' : pd.concat(allresults[condition], axis=1)['Mean'].mean(axis=1),
                                'Weight_Std' : pd.concat(allresults[condition], axis=1)['Mean'].std(axis=1),
                                'Rank' : pd.concat(allresults[condition], axis=1)['Rank'].mean(axis=1),
                                'Rank_Std' : pd.concat(allresults[condition], axis=1)['Rank'].std(axis=1)})
                df = df.sort_values(by='Mean_Rank')
                df.index = df['Gene']

                df2 = pd.DataFrame({'Gene' : expression[condition][0]['Gene'].to_list(), 
                                    'log2_FC' : pd.concat(expression[condition], axis=1)['log2_FC'].mean(axis=1)})
                df2.index = df2['Gene']
                df2 = df2.reindex(df.index)
                df['log2_FC'] = df2['log2_FC']
                df['color'] = df['log2_FC'].apply(lambda x: determine_color(x))
                df.to_csv(f'{results_dir}/{condition}/weight_expression.csv', index = False)    

            else :
                pass 
        
        classif_df2.to_csv(f'{results_dir}/cell_classification_run2.csv', index=True)
        results_df2.to_csv(f'{results_dir}/metrics_run2.csv', index=True)
    
    elif args.runs == 1 : 

        results_df = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])
        results_df2 = pd.DataFrame(columns=['nb_cells', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1_score'])
        classif_df2 = pd.DataFrame(columns = ['nb_cells', 'True_positive', 'False_negative'])

        for HTO in hto_names : 
            condition = HTO
            HTO_data = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)].to_df().T
            #Add the label 1 for 'perturbded' for each cell in the dataframe
            HTO_data.loc['Label'] = pd.Series(np.ones(len(HTO_data.columns)), index=HTO_data.columns)
            HTO_data = pd.concat([HTO_data.loc[['Label']], HTO_data.drop('Label')])
            if len(HTO_data.columns) >= 10 and len(Neg.columns) >= 10:    
                
                Neg_cut = Neg.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=42)
                HTO_data_cut = HTO_data.sample(n=min(len(HTO_data.columns), len(Neg.columns)), axis=1, random_state=42)
                #Concatenate the negative control and the perturbed cells counts
                dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                print(f'\nProcessing data for {HTO}\n')
            
                if args.eta :
                    run_SSAE(HTO, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                else :
                    run_SSAE(HTO, dataset, results_dir, eta=default_ETA, seeds=seeds)
                
                classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', header=0, sep=';')
                perturbed = classif[(classif['Labels'] == 1) & (classif['Proba_class1']>=0.5)].Name.to_list()
                control = classif[(classif['Labels'] == 0) & (classif['Proba_class0']>0.5)].Name.to_list()
                #Plot the distribution of perturbed and control cells from the classification results
                distrib_classif(results_dir, condition, classif, run=1)
                
                tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(target_data_cut.columns) * 100
                fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(target_data_cut.columns) * 100
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
                    HTO_data_cut = HTO_data.loc[:, perturbed]
                    Neg_cut = Neg_cut.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1,10))
                    HTO_data_cut = HTO_data_cut.sample(n=min(len(HTO_data_cut.columns), len(Neg_cut.columns)), axis=1, random_state=random.randint(1,10))
                    dataset = pd.concat([Neg_cut, HTO_data_cut], axis=1)
                    expression_df = pd.DataFrame({'Gene' : Neg_cut.index.to_list()[1:], 
                                        'Perturbed_expression' : HTO_data_cut.iloc[1:, :].mean(axis=1), 
                                        'Control_expression' : Neg_cut.iloc[1:, :].mean(axis=1)})
                    #Add a column that contains the log fold change of each gene between control and perturbed cells
                    expression_df['log2_FC'] = np.log2(expression_df['Perturbed_expression'] / expression_df['Control_expression'])
                    expression_df.index = expression_df['Gene']
                        
                    print(f'\n Second run for {HTO}\n')
                    if args.eta :
                        run_SSAE(HTO, dataset, results_dir, eta=best_ETA['ETA'].loc[condition], seeds=seeds)
                    else :
                        run_SSAE(HTO, dataset, results_dir, eta=default_ETA, seeds=seeds)

                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', sep=';', header=0)
                    #Plot the distribution of perturbed and control cells from the classification results
                    distrib_classif(results_dir, condition, classif, run=2)
                    tp = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(HTO_data_cut.columns) * 100
                    fn = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(HTO_data_cut.columns) * 100
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

    top_genes(results_dir, hto_names, pathways = args.pathways)   
    plot_classif(results_dir, classif_df2, hto_names)  

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) 
