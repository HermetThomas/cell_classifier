import pandas as pd
import numpy as np
import random
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm, trange
import os
import math
import time
from itertools import product
from scipy import interpolate
from bioservices import KEGG


def dir_path(path : str):

        if os.path.isdir(path):
            return path
        else:
            raise NotADirectoryError(path)

def create_results_folder(base_folder="autoencoder/results_stat/"):
    numerical_suffix = max([int(folder[-3:]) for folder in os.listdir(base_folder) if folder.startswith("results_") and folder[-3:].isdigit()], default=0) + 1
    new_folder_name = f"results_{numerical_suffix:03d}"
    new_folder_path = os.path.join(base_folder, new_folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

def get_prefix(path) :
    parts = next((file for file in os.listdir(path) if 'matrix' in file), None).split('matrix', 1)
    if len(parts) > 1 :
        return parts[0].strip()
    else :
        return ""

def replace_digit(path, i):
    # Find the index of 'Lib' in the path
    lib_index = path.find('Lib')

    # Check if 'Lib' is present in the path
    if lib_index != -1:
        # Find the digit next to 'Lib'
        digit_index = lib_index + 3
        if digit_index < len(path) and path[digit_index].isdigit():
            # Replace the digit with the variable i
            modified_path = path[:digit_index] + str(i) + path[digit_index + 1:]
            return modified_path
        else:
            raise ValueError("No digit found next to 'Lib'")
    else:
        raise ValueError("'Lib' not found in the folder name.")

def get_pathways(gene, organism="hsa"):
    try:
        k = KEGG()
        pathways = k.get_pathway_by_gene(gene, organism=organism)
        return list(pathways.keys())[:3]
    except Exception :
        return None
    
def get_pathway_info(pathway_id):
    try:
        k = KEGG()
        pathway_info = k.parse(k.get(pathway_id))
        pathway_name = pathway_info.get("NAME", "No name available")
        return pathway_name[0].split(' - ')[0]
    except Exception :
        return 'No pathway found'

def find_guides(adata) :
    grna_rows = [index for index, value in enumerate(adata.var.feature_types) if value == 'CRISPR Guide Capture']
    return [row for row in grna_rows]

def find_HTOs(adata) :
    HTO_rows = [index for index, value in enumerate(adata.var.feature_types) if value == 'Antibody Capture']
    return [row for row in HTO_rows]

def check_duplicates(input_list):
    seen = {}
    result = []

    for item in input_list:
        if item in seen:
            seen[item] += 1
            new_item = f"{item}-{seen[item] + 1}"
            result.append(new_item)
        else:
            seen[item] = 0
            result.append(item + '-1')

    return result

def clean_guides(guides_list, neg) :
    guides_list = [guide.strip() for guide in guides_list]
    guides_list = [guide.split('-')[0] for guide in guides_list]
    guides_list = [guide.split('_')[0] for guide in guides_list]

    #Remove poly-A tail sequence in the guides names
    guides_list = check_duplicates(guides_list)

    if neg == None :
        for index, item in enumerate(guides_list) :
            print(index, item)
        neg = input('index of the negative controls (separated by a space) :')
        for idx in neg.split() :
            idx = int(idx)
            guides_list[idx] = f"Neg-sg{idx}"
    
    else :
        for neg in neg :
            for i in range(len(guides_list)) :
                if neg in guides_list[i] : 
                    guides_list[i] = f"Neg-sg{i}"

    targets = [guide.split('-')[0] for guide in guides_list]
    targets = [target for target in targets if 'Neg' not in target]
    targets = [target for target in targets if 'unmapped' not in target]
    targets = list(set(targets)) 

    return guides_list, targets

def determine_color(value):
    if value <0 :
        return 'royalblue'
    elif value >0 :
        return 'indianred'
    elif value == 0 :
        return 'darkgray'

def weight_expression(results_dir, condition, expression_df) :
    top_genes_file = next((file for file in os.listdir(f'{results_dir}/{condition}') if 'Mean_Captum' in file), None)
    df = pd.read_csv(f"{results_dir}/{condition}/{top_genes_file}", sep = ';', header = 0)[['Features', 'Mean']]
    df.rename(columns={'Features' : 'Gene', 'Mean' : 'Weight'}, inplace = True)
    df.index = df['Gene']
    expression_df = expression_df.reindex(df.index)
    df['log2_FC'] = expression_df['log2_FC']
    df['color'] = df['log2_FC'].apply(lambda x: determine_color(x))
    df.to_csv(f'{results_dir}/{condition}/weight_expression.csv', index=False)

def top_genes(results_dir, targets_names, hto_names=None, pathways = False) :
    
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(len(hto_names))]
    
    targets = [target for target in targets_names if target in os.listdir(results_dir)]

    if hto_names != None :

        for target in tqdm(targets) :
            
            plt.figure()
            htos = [hto for hto in hto_names if hto in os.listdir(f'{results_dir}/{target}')]
            _, axes = plt.subplots(nrows=1, ncols=len(htos), figsize=(21, 11))

            for HTO_idx,HTO in enumerate(htos):   
                if os.path.isfile(f'{results_dir}/{target}/{HTO}/weight_expression.csv') : 
                    top = pd.read_csv(f'{results_dir}/{target}/{HTO}/weight_expression.csv', header=0).nlargest(30, 'Weight')   
                    acc_file = next((file for file in os.listdir(f'{results_dir}/{target}/{HTO}') if 'acc' in file), None)      
                    acc = round(pd.read_csv(f'{results_dir}/{target}/{HTO}/{acc_file}', header=0, index_col=0, sep=';').Global.loc['Mean'], 2)    
                    axes[HTO_idx].barh(top['Gene'], top['Weight'], color = colors[HTO_idx])
                    axes[HTO_idx].set_xlabel('Mean weight')
                    axes[HTO_idx].set_ylabel('Gene')
                    axes[HTO_idx].set_title(f'{HTO} - accuracy : {acc}')
                    axes[HTO_idx].invert_yaxis()
                    for i, tick in enumerate(axes[HTO_idx].get_yticklabels()):  # Loop through tick labels
                        tick.set_color(top['color'].loc[i])


                    if pathways == True :
                        
                        genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

                        genes_info.Gene = top.nlargest(20, 'Mean').Features

                        print(f"Searching pathways for {target}_{HTO} top genes")

                        for gene in trange(len(genes_info)) :
                            pathways_info = get_pathways(genes_info.Gene.loc[gene])

                            pathways_list = []

                            if pathways_info:
                                for pathway in pathways_info:
                                    pathway_description = get_pathway_info(pathway)
                                    pathways_list.append(pathway_description)
                                    
                            else:
                                genes_info.Pathways.loc[gene] = 'No pathways found' 

                            genes_info.Pathways.loc[gene] = ' & '.join(pathways_list)

                        genes_info.to_csv(f"{results_dir}/{target}/{HTO}/topGenes_pathways.csv", index = False, sep = ';')
                else :
                    pass    
            plt.suptitle(f"Most discriminant Features for {target}", fontsize = 30)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{target}/top_genes.png")
            plt.close()
            

    else :

        for target in tqdm(targets) :
            if os.path.isfile(f'{results_dir}/{target}/weight_expression.csv') :
                top = pd.read_csv(f'{results_dir}/{target}/weight_expression.csv', header=0).nlargest(30, 'Weight') 
                acc_file = next((file for file in os.listdir(f'{results_dir}/{target}/{HTO}') if 'acc' in file), None)      
                acc = round(pd.read_csv(f'{results_dir}/{target}/{HTO}/{acc_file}', header=0, index_col=0, sep=';').Global.loc['Mean'], 2)
                plt.figure()

                bars = plt.barh(top['Gene'], top['Weight'], color = "royalblue")
                for i, tick in enumerate(bars.get_yticklabels()):  # Loop through tick labels
                        tick.set_color(top['color'].loc[i])
                plt.xlabel("Mean weight")
                plt.ylabel("Gene")
                plt.title(f'{target} - accuracy : {acc}')
                plt.gca().invert_yaxis()
                
                plt.savefig(f"{results_dir}{target}/top_genes.png")

                if pathways == True :
                    genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

                    genes_info.Gene = top.nlargest(20, 'Weight').Features

                    print(f"Searching for pathways for {target} top genes")
                    for gene in trange(len(genes_info)) :
                        pathways = get_pathways(genes_info.Gene.loc[gene])

                        pathways_list = []

                        if pathways:
                            for pathway in pathways:
                                pathway_description = get_pathway_info(pathway)
                                pathways_list.append(pathway_description)
                                
                        else:
                            genes_info.Pathways.loc[gene] = 'No pathways found' 

                        genes_info.Pathways.loc[gene] = ' & '.join(pathways_list)
                    
                    genes_info.to_csv(f"{results_dir}/{target}/topGenes_Pathways.csv", index = False)


def getcloser(x, yfit, xnew):
    idx = (np.abs(xnew - x)).argmin()
    return yfit[idx]

def make_pchip_graph(x, y, npoints=300):
    pchip = interpolate.PchipInterpolator(x, y)
    xnew = np.linspace(min(x), max(x), num=npoints)
    yfit = pchip(xnew)
    plt.plot(xnew, yfit)
    return (xnew, yfit)

def eta_plot(list_acc, list_eta, results_dir, condition) :

    plt.figure()

    accuracy = np.array([acc for acc in list_acc])
    RadiusC = np.array(list_eta)

    radiusToUse= RadiusC
    accToUse=accuracy

    xnew, yfit = make_pchip_graph(radiusToUse,accToUse)

    plt.title(condition)  # titre général
    plt.xlabel("Parameter $\eta$")                         # abcisses
    plt.ylabel("Accuracy")                      # ordonnées

    a = min(radiusToUse)
    b = max(radiusToUse)  #  
    tol = 0.01  # impact the execution time
    r= 0.5*(3-math.sqrt(5))

    start_time = time.time()

    while (b-a > tol):
        c = a + r*(b-a);
        d = b - r*(b-a);
        if(getcloser(c, yfit,xnew) > getcloser(d, yfit,xnew)):
            b = d
        else:
            a = c

    parameter = getcloser(c,xnew, xnew)

    end_time = time.time()

    # Calculate and print the execution time
    execution_time = (end_time - start_time)*1000

    print(f"Execution time: {execution_time} ms")


    print("Golden Section Optimal parameter", parameter)
    print("Golden section Maximum accuracy ", getcloser(c,yfit,xnew))


    plt.axvline(x=parameter, color='g', linestyle='--', label='Optimal Parameter')

    # Display the legend
    plt.legend()
    parts = condition.split('/')
    # Show the plot
    plt.savefig(f'{results_dir}/{parts[0]}/{parts[1]}_ETA_curve.png')
    plt.close()

def features_df(top_genes) :

    final_df = pd.DataFrame()
    # Iterate through each dataframe in the list
    for df in top_genes:
        # Extract 'Features' and 'Weights' columns
        features = df['Features'].reset_index(drop=True)
        weights = df['Mean'].reset_index(drop=True)
        
        # Concatenate 'Features' and 'Weights' columns alternately
        df_concat = pd.concat([features, weights], axis=1)
        
        # Append the concatenated dataframe to the final dataframe
        final_df = pd.concat([final_df, df_concat], axis=1)
    
    return final_df

def sortDataframe(df):
    nbColumnToSort= int((df.shape[1]/2) -1)
    sorter = df["Features1"].tolist()
    for i in range(nbColumnToSort):
        df2 = df.iloc[:, [2*i+2, 2*i+3]]
        sort2= df2.iloc[:,0].tolist()
        for item in sort2:
            if item not in sorter:
                sorter.append(item)
        df2= df2.rename(columns={df2.columns[0]: 'Features'})
        df2.Features = df2.Features.astype("category")
        df2.Features = df2.Features.cat.set_categories(sorter)
        df2= df2.sort_values(["Features"])
        df.iloc[:, [2*i+2, 2*i+3]] = df2.iloc[:, :2]
    
    return df

def plotValues(df, figdir):
    nbColumn= int((df.shape[1]/2))
    plt.figure(figsize=(10,7))
    for i in range(nbColumn):
        df2 = df.iloc[:, [2*i, 2*i+1]]
        df2=df2.dropna(how='any') 
        plt.plot(df2.iloc[:, 0],df2.iloc[:, 1] , label=df2.columns[1] ) 
        
    plt.title("Feature ranking as a function of the weight")  # titre général
    plt.xlabel("Feature name")                         # abcisses
    plt.ylabel("Feature's weight")                      # ordonnées
    
    #plt.legend(["Bilevel$\ell_{1.\infty}$","$\ell_{1.\infty}$","$\ell_{1.1}$"])
    plt.legend(loc="upper right")
    plt.xticks(rotation=90)
    plt.savefig(f'{figdir}_ETA_weight.png') 

def eta_weight_plot(df, figdir, list_eta) :
    
    Nbr= 20 #Number of features to keep in the plot
    
    listfeat=[]
    for idx, eta in enumerate(list_eta) :
        listfeat+=[f'Features{idx}', f"$\eta={eta}$"]
    df.columns = listfeat
    for idx, eta in enumerate(list_eta) :
        df[f"$\eta={eta}$"] = df[f"$\eta={eta}$"].apply(float)
        df[f'Features{idx}'] = df[f'Features{idx}'].apply(str)
    
    df= df.head(Nbr)
    
    df=sortDataframe(df)
    
    plotValues(df, figdir)

def plot_classif(results_dir, df, targets, hto_names=None, eta='', run = '') :

    hto_list = hto_names
    grna_list = targets
    if hto_names :

        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(len(hto_list))]
        # Common color for the second bar in each plot
        common_color = 'bisque'

        # Create a figure and subplots for each combination of HTO and gRNA
        fig, axes = plt.subplots(len(hto_list), len(grna_list), figsize=(len(grna_list)*2, len(hto_list)*4), sharex='col', sharey='row')

        for i, hto in enumerate(hto_list):
            for j, grna in enumerate(grna_list):
                condition = f'{grna}/{hto}'
                if condition in df.index :
                    y1, y2 = df.loc[condition, ['True_positive', 'False_negative']].values
                    # Plot data
                    ax = axes[i, j]
                    ax.bar('A', y1, width=0.5, label=f'{hto} - {grna}', color=colors[i], alpha=1)  # Use different colors for each HTO
                    ax.bar('A', y2, width=0.5, label=f'{hto} - {grna} (Second)', color=common_color, alpha=0.73, bottom=y1)  # Common for the second bar, stacked on top
                    ax.get_xaxis().set_visible(False)
                    ax.set_title(grna, fontdict={'family':'sans-serif','color':'black','weight':'normal','size':14})
                elif 'Labelspred_softmax.csv' in os.listdir(f'{results_dir}/{condition}') :
                    classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', sep=';', header=0)
                    y1 = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(classif[classif['Labels'] == 1]) * 100
                    y2 = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(classif[classif['Labels'] == 1]) * 100
                    # Plot data
                    ax = axes[i, j]
                    ax.bar('A', y1, width=0.5, label=f'{hto} - {grna}', color=colors[i], alpha=1)  # Use different colors for each HTO
                    ax.bar('A', y2, width=0.5, label=f'{hto} - {grna} (Second)', color=common_color, alpha=0.73, bottom=y1)  # Common for the second bar, stacked on top
                    ax.get_xaxis().set_visible(False)
                    ax.set_title(grna, fontdict={'family':'sans-serif','color':'black','weight':'normal','size':14})
        # Add legend with color patches for each HTO and the common color
        for i, hto in enumerate(hto_list):
            legend_ax = fig.add_subplot(len(hto_list), 1, i+1)
            legend_patches = [patches.Patch(color=colors[i], label='Perturbed cells %')]

            # Add color patch for the common color
            legend_patches.append(patches.Patch(color=common_color, label='Non perturbed cells %'))

            legend_ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title=hto, fontsize='large', title_fontsize='x-large')
            legend_ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{results_dir}/perturbed_cells.png')
        plt.close()
    
    else :        
        # Common color for the second bar in each plot
        common_color = 'bisque'

        grna_list = targets
        # Determine the number of rows and columns for the layout
        num_grna = len(grna_list)
        num_cols = min(8, num_grna)
        num_rows = (num_grna - 1) // num_cols + 1

        # Create a figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 15), sharex='col', sharey='row')

        # Generate example data and plot for each grna
        for i, condition in enumerate(grna_list):
            # Calculate the row and column index
            row_index = i // num_cols
            col_index = i % num_cols
            if condition in df.index :
                y1, y2 = df.loc[condition, ['True_positive', 'False_negative']].values
                # Plot data
                ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]
                ax.bar('A', y1, width=0.5, label=f'{grna}', color='mediumslateblue', alpha=1)  # Use blue color for all bars
                ax.bar('A', y2, width=0.5, label=f'{grna} (Second)', color=common_color, alpha=0.63, bottom=y1)  # Same color for the second bar, stacked on top
                # Set title with a rectangle around it
                ax.set_title(f'{grna}', fontdict={'fontsize': 12, 'fontweight': 'normal', 'fontfamily': 'sans-serif'})
                ax.get_xaxis().set_visible(False)
                # Hide the y-axis
                ax.get_yaxis().set_visible(False)
            
            elif 'Labelspred_softmax.csv' in os.listdir(f'{results_dir}/{condition}') :
                classif = pd.read_csv(f'{results_dir}/{condition}/Labelspred_softmax.csv', sep=';', header=0)
                y1 = len(classif[classif['Labels'] == 1][classif['Proba_class1'] >= 0.5]) / len(classif[classif['Labels'] == 1]) * 100
                y2 = len(classif[classif['Labels'] == 1][classif['Proba_class1'] < 0.5]) / len(classif[classif['Labels'] == 1]) * 100
                ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]
                ax.bar('A', y1, width=0.5, label=f'{grna}', color='mediumslateblue', alpha=1)  # Use blue color for all bars
                ax.bar('A', y2, width=0.5, label=f'{grna} (Second)', color=common_color, alpha=0.63, bottom=y1)  # Same color for the second bar, stacked on top
                # Set title with a rectangle around it
                ax.set_title(f'{grna}', fontdict={'fontsize': 12, 'fontweight': 'normal', 'fontfamily': 'sans-serif'})
                ax.get_xaxis().set_visible(False)
                # Hide the y-axis
                ax.get_yaxis().set_visible(False)

        # Create legend outside of subplots
        legend_patches = [patches.Patch(color='mediumslateblue', label='Perturbed %'), patches.Patch(color=common_color, label='Non perturbed cells %')]
        plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        # Adjust layout to accommodate legends and color descriptions
        plt.tight_layout()
        plt.savefig(f'{results_dir}/perturbed_cells.png')
        plt.close()

def plot_weight_FC(results_dir, targets, hto_names = None) :
    if hto_names :
        for hto in hto_names :
            for target in targets :
                df = pd.read_csv(f'{results_dir}/{target}/{hto}/weight_expression.csv', header=0, index_col=0)
                df = df.nlargest(30, 'Weight')
                _, ax = plt.subplots(figsize = (9,7))
                weights = list(df['Weight'])
                fc = list(df['log2_FC'])
                colors = list(df['color'])
                names=list(df.index)
                ax.scatter(x=fc, y=weights, c=colors)
                ax.set_xlabel('log(Fold Change)')
                ax.set_ylabel('Gene weight')
                ax.set_title(f'{target} - {hto}')
                for i, txt in enumerate(names):
                    ax.annotate(txt, (fc[i], weights[i]), ha='center')
                plt.tight_layout()
                plt.savefig(f'{results_dir}/{target}/{hto}/weight_logFC.png')

def distrib_classif(results_dir, condition, df, eta='', run='') :
    

    df['Classification'] = df['Labels'].replace({0: "control", 1: "targeted"})
    plt.figure()
    sns.kdeplot(data=df, x='Proba_class1', hue='Classification', alpha=.3, fill=True)
    plt.xlabel('Perturbation score')
    plt.tight_layout()
    if eta and run :
        plt.savefig(f'{results_dir}/{condition}/Perturbation_score_eta{eta}_run{run}.png')
    elif eta :
        plt.savefig(f'{results_dir}/{condition}/Perturbation_score_eta{eta}.png')
    elif run :
        plt.savefig(f'{results_dir}/{condition}/Perturbation_score_run{run}.png')
    else :
        plt.savefig(f'{results_dir}/{condition}/Perturbation_score.png')    
    plt.close()

def accuracy_heatmap(results_dir, targets, hto_names) :
    df2 = pd.DataFrame(columns=hto_names)
    for HTO in hto_names :
        for target in targets :
            if HTO in os.listdir(f'{results_dir}/{target}') :
                if 'Perturbation_score_run2.png' in os.listdir(f'{results_dir}/{target}/{HTO}') :
                    acc_file = next((file for file in os.listdir(f'{results_dir}/{target}/{HTO}') if 'acc' in file), None)      
                    df2.loc[target, HTO] = pd.read_csv(f'{results_dir}/{target}/{HTO}/{acc_file}', sep=';', index_col=0, header=0).loc['Mean', 'Global']

    df = df2.apply(pd.to_numeric, errors='coerce')

    df_filled = df.fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_filled, annot=True, cmap='RdYlGn', linewidths=.4)
    plt.title('Heatmap of Accuracy per condition')
    plt.savefig(f'{results_dir}/accuracy_heatmap.png')
