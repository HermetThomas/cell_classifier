# cell_classifier
Derivation of the CRISPR_demux repository to analyze datasets using HTO only 

This repository contains a Python script used to exctract discriminating genes between the various conditions in your dataset.

The script uses Hashsolo by Scanpy to load Cellranger counts matrices and perform the demultiplexing of the HTOs in the dataset.
It then separates the dataset into X*Y subsets according to the HTO in each cell.
The top discriminating features between each condition and the negative control are found through the use of a supervised AutoEncoder and stored in a dataframe. The biological pathways in which these genes are implicated are fetched in the KEGG database with the BioServices package.

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
|cell_classifier.py|Script to launch|
|autoencoder/|Contains the AutoEncoder sript and functions it calls|
|requirements.txt|Python packages required to run the script|

## **Installation**

```{bash}
git clone https://github.com/HermetThomas/cell_classifier.git
```

## **Input format**

If you have multiple libraries, homologs are in the same directory with identical names except their index from 1 to n &rarr; Lib1, Lib2, ... Libn

> [!WARNING]
> Only the digit directly next to 'Lib' will be automatically modified.

Example :
* data/
   * Counts_Lib1/
   * Counts_Lib2/
   * HTO_Lib1_03-04-2024_01/
   * HTO_Lib2_03-04-2024_02/

The different libraries need to contain the same files as the following :

* Counts_Lib1/
   * matrix.mtx.gz  
   * barcodes.tsv.gz
   * features.tsv.gz

* Counts_Lib2/
   * matrix.mtx.gz  
   * barcodes.tsv.gz
   * features.tsv.gz


### **Matrix.mtx**

Counts matrix &rarr; **X cells * Y genes**

### **barcodes.tsv**

.tsv file containing the cell barcode associated to each cell in the counts matrix
&rarr; **X rows**

### **features.tsv**

.tsv file containing the genes names and types &rarr; **Y rows**

## **Usage**



### **If the HTO counts and gRNA counts are in the main counts matrix**

The HTO are found by using the gene types in the features.tsv.gz file.

'Antibody Capture' &rarr; HTO 

Add -neg {Name of the negative control guides}

```{bash}
python3 CRISPR_demux.py 
   -libs number_of_libraries
   -counts /path/to/first/counts_library/
   -neg Neg_control1 Neg_control2
```

### **If the HTO counts are in a separate counts matrix**

Add the path to the first library of HTO counts / gRNA counts / both

```{bash}
python3 CRISPR_demux.py 
   -counts /path/to/first/counts_library/
   -hto /path/to/first/HTO_library/
   -neg Neg_control1 Neg_control2
```

### **If you want to plot the distribution of the HTOs among the cells**

Add '-plot' to the command line

```{bash}
python3 CRISR_demux.py
   -counts /path/to/first/counts_library/
   -neg Neg_control1 Neg_control2
   -plot
```


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

Add '-eta' to the command line

```{bash}
python3 CRISR_demux.py
   -counts /path/to/first/counts_library/
   -neg Neg_control
   -eta
```

[!WARNING] The computation time will be multiplied by the number of tested parameters
