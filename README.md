# cell_classifier

This repository contains a Python script used to exctract discriminating genes between two conditions in your dataset.

The script takes as input a pre-processed single-cell anndata object (.h5ad format)

We use a supervised AutoEncoder to find top discriminant features between a condition of interest and a control condition. 
We added an option to fetch in the KEGG database the biological pathways in which these discriminant genes are implicated, with the use of the BioServices package.

The AutoEncoder used to perform the classification was developed by Barlaud M. and Guyard F. and published in the following papers :

Michel Barlaud, Frédéric Guyard : *Learning sparse deep neural networks using efficient structured projections on convex constraints for green ai.* ICPR 2020 Milan Italy (2020) doi : 10.1109/ICPR48806.2021.9412162

and 

David Chardin, Cyprien Gille, Thierry Pourcher and Michel Barlaud : *Accurate Diagnosis with a confidence score using the latent space of a new Supervised Autoencoder for clinical metabolomic studies.* BMC Informatics 2022 doi: 10.1186/s12859-022-04900-x

# Table of contents 

1. [Repository Content](repository-content)
2. [Installation](#installation)
3. [Input format](#input-format)
4. [Usage](#usage)


## **Repository Content**
|File/Folder|Description|
|:-:|:-:|
|cell_classifier.py|Script to run for analysis|
|classifier_functions.py|Various functions used in main script|
|autoencoder/|Contains the AutoEncoder sript and functions it calls|
|requirements.txt|Python packages required to run the script|

## **Installation**

```{bash}
git clone https://github.com/HermetThomas/cell_classifier.git
```

## **Input format**

a single .h5ad AnnData object with pre-filtered counts and raw counts in .X

### Required arguments

-adata _path/to/AnnData/object.h5ad_
-obs _Name of the column in .obs to use for classification_
-tests _Name(s) of the condition(s) to test_
-neg _Name of the negative control (needs to be in the classification column)_


### **If you want to perform multiple runs on every dataset**

Add '-runs {number of runs to compute}' to the command line

```{bash}
python3 CRISR_demux.py
   -counts /path/to/first/counts_library/
   -neg Neg_control1 Neg_control2
   -runs 3
```

### **If you want to search for  pathways associated to the discriminant genes**

Add '-pathways' to the command line

```{bash}
python3 CRISR_demux.py
   -counts /path/to/first/counts_library/
   -neg Neg_control1 Neg_control2
   -runs 3
   -pathways
```

### **If you want to use the optimal projection radius for each subdataset**

_Not necessary, can be used to get a better accuracy depending on the dataset_
Add '-eta' to the command line

```{bash}
python3 CRISR_demux.py
   -counts /path/to/first/counts_library/
   -neg Neg_control
   -eta
```

[!WARNING] The computation time will be multiplied by the number of tested parameters
