## --------------------------------------------------------------------
## Original code copyright Nicolas Nemeth 2023
## Covered by original MIT license
## --------------------------------------------------------------------


import numpy as np
import os
from typing import Tuple, Union, List


SELECTED_FEATURES = {aa: "ARNDCQEGHILKMFPSTWYV" for aa in "ACDEFGHIKLMNPQRSTVWY"}



def parse_pssm_matrix(filename: str, seq_range: Tuple[int, int]=None) -> List:
    """
        Computes Position-Specific-Scoring-Matrix Composition Encoding
        given a text-file containing the PSSM-profile of a given protein.
            
        Args:
            filename (str): the .txt file containing the PSSM-profile
            seq_range (Tuple[int, int]): the sequence region to compute the encoding for
            
        Returns:
            400-sized list containing the PSSM-composition encoding for a given protein sequence
    """
    lines = list()
    sequences = list()
    
    with open(filename, 'r') as ifile:
        file_content = ifile.read()
        file_lines = [l.strip() for l in file_content.split('\n') if l.strip() != ""]
        for i in range(len(file_lines)-1):
            if file_lines[i+1][0].isdigit():
                PSSM_columns = np.array(file_lines[i].split()[:20])
                break
            
    for l in file_lines:
        if l[0].isdigit():
            sequences.append(l.split()[1])
            lines.append(l.split()[2:22])
           
    selected_idxs = dict()
    for aa_seq, aa_cols in SELECTED_FEATURES.items():
        selected_idxs[aa_seq] = list()
        for aa in aa_cols:
            selected_idxs[aa_seq].append(np.where(PSSM_columns == aa)[0][0])
    for aa in selected_idxs:
        selected_idxs[aa] = np.array(selected_idxs[aa])
        
        
    sequences = np.array(sequences)
    pssm_matrix = np.array(lines, dtype=np.float64)
    if seq_range is not None and len(pssm_matrix) > seq_range[1]:
        pssm_matrix = pssm_matrix[seq_range[0]:seq_range[1]]
        sequences = sequences[seq_range[0]:seq_range[1]]
        
    
    first = True
    pssm_comp = list()
    for aa in SELECTED_FEATURES.keys():
        rows = np.where(sequences == aa)[0]
        if first and rows.size != 0:
            pssm_comp.extend(np.sum(pssm_matrix[rows,:][:,selected_idxs[aa]], axis=0))
            first = False
            continue
        if rows.size == 0:
            pssm_comp.extend(np.zeros(len(SELECTED_FEATURES[aa])))
        else:
            pssm_comp.extend(np.sum(pssm_matrix[rows,:][:,selected_idxs[aa]], axis=0))
    ###
    return pssm_comp


def PSSM(fileName: str, idx_range: Union[Tuple[int, int], List[int]], seq_range: Tuple[int, int]=None) -> np.ndarray:
    """
        Computes the total PSSM composition encoding matrix for a list of protein sequences
        
        Args:
            fileName (str): base file name of the .txt files containing the PSSM-profile for the proteins
            idx_range (Tuple[int, int]): range in-between which .txt files are numbered, e.g.
                
                pssm_0.txt
                pssm_1.txt
                ...
                pssm_49.txt
                
                -> we have 50 protein sequences and 50 .txt files with pssm-profile where
                'pssm_' is the ´fileName´ and (0, 50) is the index range ´idx_range´
            seq_range (Tuple[int, int]): sequence region to compute the PSSM encoding for
            
        Returns:
            n x 400 dimensional np.ndarray where n is the number of steps in the index range
            or the number of entries in the list idx_range
    """
    
    features = [ parse_pssm_matrix(str(fileName)+str(idx_range[0])+".txt",seq_range=seq_range) ]
    
    if type(idx_range) is tuple and len(idx_range) == 2:
        for id_ in range(idx_range[0]+1, idx_range[1]):
            features.append( parse_pssm_matrix(fileName+str(id_)+".txt",seq_range=seq_range) )
    else:
        for id_ in idx_range[1:]:
            features.append( parse_pssm_matrix(fileName+str(id_)+".txt",seq_range=seq_range) )
    return np.array(features)