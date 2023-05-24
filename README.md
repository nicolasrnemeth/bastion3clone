# Bastion 3 Clone

Binary classification for predicting whether proteins are secreted by the bacterial Type III secretion system based on 
combinations of three models trained on three different feature groups. This software is a reimplementation of the existing
software Bastion 3, which provides open access to the source code and is used as for a direct comparison with EffectiveT3.


## Install dependencies using the following command 
### Note: they will be automatically installed via pip using the install command below
> pip install -r requirements.txt


## Install using pip

install from PyPI:    

> pip install bastion3clone

install locally:      

> pip install .

## Commands available by the program (check help description by running below commands)

> bastion3clone --help

> trainmodel --help

The 1. command is used for prediction. 
The 2. command is used for training a model.


## Configure training parameters used for training script

Inside the file 'training_config.yaml' you can change the existing training parameters.
If configuration file cannot be loaded due to an error or there are missspellings for certain keys,
then the default values will be taken. In this case you will be informed via console-output.

## Short guide and support in how to obtain PSSM-profiles of proteins

The pssm-profile should be in ascii-format obtained by the ´psiblast´-command from the ncbi blast + suite:

Check details here on how to obtain pssm-profiles in ascii format: https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html

The commands goes like this (you need to download the uniref50 protein database as fasta-file before)

Create database from uniref50.fasta file

> makeblastdb -in uniref50.fasta -dbtype prot -out uniref50_db

Compute pssm-profile for a single protein sequence given a single fasta file parameters inlcuding the curly braces {params} by their actual values

> psiblast -query {single_protein_fasta_file_name}.fasta -db {uniref50_database_prepared_using_makeblastdb_command} -num_iterations 2 -save_pssm_after_last_round -out_ascii_pssm {path_where_to_save_pssm_file}.txt -num_threads {number_of_threads_to_use_for_computing_pssm_profile}

## LICENSE

see LICENSE.txt

### Original paper:

Jiawei Wang and others, Bastion3: a two-layer ensemble predictor of type III secreted effectors, 
Bioinformatics, Volume 35, Issue 12, June 2019, Pages 2017–2028, 
https://doi.org/10.1093/bioinformatics/bty914
