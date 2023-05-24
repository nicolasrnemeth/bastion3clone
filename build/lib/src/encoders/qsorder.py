## --------------------------------------------------------------------
## Original code copyright Zhen Chen, Xuhan Liu, Pei Zhao, Chen Li, Yanan Wang, Fuyi Li, Tatsuya Akutsu, Chris Bain, Robin B Gasser, Junzhou Li, Zuoren Yang, Xin Gao, Lukasz Kurgan, Jiangning Song 2022
## Covered by original MIT license
##
## Modifications copyright Nicolas Nemeth 2023
## Modifications licensed under the MIT License
## --------------------------------------------------------------------



import sys, platform, os, re
import numpy as np
from typing import Tuple


pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)



def minSequenceLengthWithNormalAA(fastas: np.ndarray) -> int:
    """
        Computes the minimum sequence length.
        
        Args:
            fastas (np.ndarray): array of List[str, str] -> [protein identifier, protein sequence]
            
        Returns:
            int: minimum sequence length
    """
    return min([len(re.sub('-', '', sequence)) for _, sequence in fastas])


def QSOrder(fastas: np.ndarray, nlag: int=30, w: float=0.1, seq_range: Tuple[int, int]=None) -> np.ndarray:
    """
        Quasi Sequence Order
    
        QSO (Chou, 2000) explores a protein’s order effect to generate 
        a 100-dimensional feature vector, by measuring the physicochemical 
        distance between amino acids within the sequence using the physicochemical
        distance matrices by Schneider-Wrede (1994) and Grantham (1974)
        
        Args:
            fastas (np.ndarray): array containing List[str, str] -> List[protein identifier, protein sequence]
            nlag (int): maximum lag + 1, must be smaller than the protein sequence length
            w (float): weight for the sequence order coupling number
            seq_range (Tuple[int, int]): sequence region to compute the quasi-sequence order for
            
        Returns:
            n x 100 dimensional np.ndarray where is the number of protein sequences and
            100 only holds true when nlag is equal to 30, because the feature dimension
            depends on the parameter nlag
    """
    if seq_range is not None:
        for idx in range(len(fastas)):
            if len(fastas[idx][1]) > seq_range[1]:
                fastas[idx][1] = fastas[idx][1][seq_range[0]:seq_range[1]] 
    
    if minSequenceLengthWithNormalAA(fastas) < nlag + 1:
        print('Error: all sequences should be longer than the nlag+1: ' + str(nlag + 1) + '\n\n')
        return 0

    dataFile = os.path.join("src", "encoders", "data", "Schneider-Wrede.txt")
    dataFile1 = os.path.join("src", "encoders", "data", "Grantham.txt")

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {}
    for i in range(len(AA)):
        DictAA[AA[i]] = i

    DictAA1 = {}
    for i in range(len(AA1)):
        DictAA1[AA1[i]] = i

    with open(dataFile) as f:
        records = f.readlines()[1:]
    AADistance = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance.append(array)
    AADistance = np.array(
        [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

    with open(dataFile1) as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance1.append(array)
    AADistance1 = np.array(
        [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
        (20, 20))

    features = list()
    for name, sequence in fastas:
        sequence = re.sub('-', '', sequence)
        arraySW = []
        arrayGM = []
        for n in range(1, nlag + 1):
            arraySW.append(
                sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
            arrayGM.append(sum(
                [AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
        myDict = {}
        code = list()
        for aa in AA1:
            myDict[aa] = sequence.count(aa)
        for aa in AA1:
            code.append(myDict[aa] / (1 + w * sum(arraySW)))
        for aa in AA1:
            code.append(myDict[aa] / (1 + w * sum(arrayGM)))
        for num in arraySW:
            code.append((w * num) / (1 + w * sum(arraySW)))
        for num in arrayGM:
            code.append((w * num) / (1 + w * sum(arrayGM)))
        features.append(code)
    return np.array(features)