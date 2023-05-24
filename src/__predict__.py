import os
import sys
import time
import itertools
from .predictor import predictor
from multiprocessing import cpu_count

from argparse import ArgumentParser, RawTextHelpFormatter
from .__init__ import __version__

## CONSTANTS

CPU_COUNT = cpu_count()

# Description strings for command line parameters
DESCRIPTION = """Classifies whether bacterial proteins are secreted by the Type III secretion system, 
based on information contained in the protein sequence. There are three groups of features which 
have been used to train three different models.

\n    - GROUP 1: sequence features (single amino acid, dipeptide composition and quasi order structure)
\n    - GROUP 2: amino acid property features (based on hydrophobicity, normalized van der vaals volume, polarity, charge, secondary structure and solvent accessibility)
\n    - GROUP 3: evolutionary information obtained from the position-specific scoring matrix."""

PSSM_HELP = """(Required) Path to the folder containing the Position Specific Scoring Matrices.
Except for an additional appended index corresponding to the order of the protein sequences 
in the input fasta file, the file names and file type extensions should be equal. 
Therefore, please also append the file name to the path. 
Example: you have 100 protein sequences that you want to predict. The value you provide to the
'--path' command line argument reads 'your_folder/pssm_files/', then you should have 100 files like:

    - 0.txt
    - 1.txt
    ...
    - 99.txt

in your folder 'pssm_files'. This is not required if your model selection does not include the 'G3'-model.
In this case simply pass some string to --path, e.g. --path some_string

The pssm-profile should be in ascii-format obtained by the ´psiblast´-command from the ncbi blast + suite:

check details here on how to obtain pssm-profiles in ascii format: https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html
"""

# Possible model combinations
AVAILABLE_MODELS = [''.join(subset) for size in range(3) for subset in
                    itertools.combinations(['G1', 'G2', 'G3'], size + 1)]

MODEL_HELP = """(Optional) Choose any combination of the three single models.
    - G1 = Protein-sequence-based prediction
    - G2 = Amino-acid-property-based prediction
    - G3 = PSSM-composition-based prediction
Available models: """ + ' | '.join(AVAILABLE_MODELS)


CPU_CORES_HELP = """(Optional) The number of CPU-cores that you want to use for prediction. By default all available CPU cores are used.
If your selected number of cores is above the available number of cores you will be provided with the
number of accessible CPU-cores on your operating system, to provide a valid choice the this parameter."""


TRUE_LABELS_HELP = """"Path to the file containing the comma-separated true labels encoded as integers (0 for False (not-secreted) and 1 for True (secreted)).
The labels must be in the same order as the protein sequences contained in the input fasta-file. If this command line parameter is set, i.e. not None
then an additional file ("{output_file_name_as_set_in_the_command_line_argument}_metrics.json") containing all kinds of evaluation metrics for the prediction will be saved.
So the json-file containing the evaluation metrics will be saved inside the same path as the outputfile name containing only the predictions but with "_metrics.json"
concatenated to the filename.
"""

##


def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    # Fasta file containing sequences to predict
    parser.add_argument('-f', '--file', required=True, type=str,
                        help="(Required) Path to the input fasta-file, e.g. 'your_folder/your_file.fasta'")

    # Path to directory containing PSSM files
    parser.add_argument('-p', '--path', required=True, type=str,
                        help=PSSM_HELP)

    # Output file path
    parser.add_argument('-o', '--ofile', required=False, type=str, default='results.txt',
                        help='(Required) Provide the file path for the output file. --ofile path/{file_name}.json to '
                        + 'save it in json-format or path/{file_name}.txt to save it in txt-format')

    # Model selection
    parser.add_argument('-m', '--model', choices=AVAILABLE_MODELS, required=False, type=str, default='G1G2G3',
                        help=MODEL_HELP)
    
    # Number of cores to use for prediction
    parser.add_argument('-c', '--cores', choices=list(range(1, CPU_COUNT+1)), required=False, type=int, default=CPU_COUNT,
                        help=CPU_CORES_HELP)
    
    # True labels
    parser.add_argument('-l', '--truelabels', required=False, type=str, default=None,
                        help=TRUE_LABELS_HELP)

    # Program version
    parser.add_argument('-v', '--version', action='version', version='bastion3clone ' + __version__,
                        help="(Optional) Show program's version number and exit")

    return parser.parse_args()



def convert_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m {seconds}s"



def start(pargs):
    start = time.time()
    predictor(fasta_file=pargs.file, pssm_folder=pargs.path,
              num_cores=pargs.cores, ofile_path=pargs.ofile, 
              model_=pargs.model, seq_range=None, true_labels_file_name=pargs.truelabels)
    print(f"\nPrediction took {convert_seconds(time.time() - start)}\n")


def main():
    try:
        # Start program
        args = parse_args()
        start(args)
        file_path = os.path.join(os.getcwd(), args.ofile)
        print('Successful execution of the program!')
        print('\n--> Please find the results here: ' + file_path)
        sys.exit(0)
    except Exception as e:
        print('Program ran into an error: ', str(e))
        sys.exit(0)



if __name__ == '__main__':
    main()