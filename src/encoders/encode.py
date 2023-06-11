import numpy as np
from typing import Tuple
from .aac import AAC
from .ctdc import CTDC
from .ctdt import CTDT
from .dpc import DPC
from .pssm import PSSM
from .qsorder import QSOrder


def encode(fastas: np.ndarray, pssm_folder: str, seq_range: Tuple[int, int] = None,
           model_: str = "G1G2G3") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Encode protein sequences using different features groups

        Args:
            fastas (np.ndarray): array containing the protein sequences
            pssm_folder (str): folder containing the pssm .txt files
            seq_range (Tuple[int, int]): sequence range to use for prediction (defaults to full-length)
            model_ (str): single model combination to use for prediction
            process_id (str): id of the process, in case this task is executed in parallel
            result_dict (dict): dictionary to add the results to

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] containing the protein identifiers
            and the encoded features of all input protein sequence for each feature group
    """

    # Only perform sequence region extraction once to improve computation time
    if seq_range is not None:
        for idx in range(len(fastas)):
            if len(fastas[idx][1]) > seq_range[1]-1:
                fastas[idx][1] = fastas[idx][1][seq_range[0]:seq_range[1]]

    # Sequence composition (Group 1)
    g1_feats, g2_feats, g3_feats = [None] * 3
    if "G1" in model_:
        aac = AAC(fastas, seq_range=None)
        dpc = DPC(fastas, seq_range=None)
        qsorder = QSOrder(fastas, seq_range=None)
        g1_feats = np.hstack((aac, dpc, qsorder))
    # Amino acid properties (Group 2)
    if "G2" in model_:
        ctdc = CTDC(fastas, seq_range=None)
        ctdt = CTDT(fastas, seq_range=None)
        g2_feats = np.hstack((ctdc, ctdt))
    # PSSM composition (Group 3)
    if "G3" in model_:
        g3_feats = PSSM(pssm_folder, (0, len(fastas)), seq_range=None)
    # names, sequence features, amino acid property features, pssm features
    return fastas[:, 0],  g1_feats, g2_feats,  g3_feats
