# Supervised Autoencoder

This is the code from : *Accurate Diagnosis with a confidence score using the latent space of a new Supervised Autoencoder for clinical metabolomic studies.*

In this repository, you will find the code to replicate the statistical study described in the paper.
  
When using this code, please cite:

> Barlaud, M., Guyard, F.: Learning sparse deep neural networks using efficient structured projections on convex constraints for green ai. ICPR 2020 Milan Italy (2020)

and 

> David Chardin, Cyprien Gille, Thierry Pourcher and Michel Barlaud : Accurate Diagnosis with a confidence score using the latent space of a new Supervised Autoencoder for clinical metabolomic studies.


## Table of Contents
***
1. [Repository Contents](repository-contents)
2. [Installation and Use](#installation-and-use)
  
### **Repository Contents**
|File/Folder | Description |
|:---|:---:|
|`Run_SSAE_alldata.py`|Main script to train and evaluate the SAE|
|`functions`|Contains fuctions called in the main script|
    
### **Installation and Use** 
---

The autencoder is automatically run when running the CRISPR_demux.py script
The dependencies are already specified in the requirements.txt of the main branch

To change the number of iterations on each dataset change X in Seed = [X] at line 70 in Run_SSAE_alldata.py 

---
