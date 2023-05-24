import os
import sys
import json
import time
import pickle
import warnings
from typing import Any
from .trainer import Trainer
from argparse import ArgumentParser, RawTextHelpFormatter

"""
    Once the training has finished add model_G1.bin, model_G2.bin and model_G3.bin
    inside the folder 'SAVED_MODELS_FOLDER' (see variable below) inside src/models.
    src/models contains the models that will be used by the prediction script.
"""

## Constants

# Folder where to save models and parameters
SAVED_MODELS_FOLDER = "src/training/models_and_parameters"

DESCRIPTION = """Trains the model and saves the optimized hyperparameters and the model 
to the folder 'src/training/models_and_parameters/'."""

##



def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    parser.add_argument('-p', '--pos', required=True, type=str,
                        help="Path to the fasta-file containing positive protein sequences.")

    parser.add_argument('-n', '--neg', required=True, type=str,
                        help="Path to the fasta-file containing negative protein sequences.")

    return parser.parse_args()



def convert_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m {seconds}s"



def save_model(model_path: str, param_path: str, model: Any, parameters: dict) -> None:
    with open(model_path, 'wb') as ofile:
        pickle.dump(model, ofile)
    with open(param_path, 'w') as ofile:
        json.dump(parameters, ofile)



def start(pargs: dict) -> None:
    start = time.time()
    print("\nLoading models and computing encodings ...\n")
    trainerG1 = Trainer(pargs.pos, pargs.neg, feature_group='G1', seq_range=None)
    trainerG2 = Trainer(pargs.pos, pargs.neg, feature_group='G2', seq_range=None)
    trainerG3 = Trainer(pargs.pos, pargs.neg, feature_group='G3', seq_range=None)
    print("Training Model G1 ...\n")
    model_G1, parameters_G1 = trainerG1.train()
    save_model(SAVED_MODELS_FOLDER+"/model_G1.bin", SAVED_MODELS_FOLDER+"/parameters_G1.json", 
               model_G1, parameters_G1)
    print("Done! Model G1 and parameters saved!")
    print("Training Model G2 ...\n")
    model_G2, parameters_G2 = trainerG2.train()
    save_model(SAVED_MODELS_FOLDER+"/model_G2.bin", SAVED_MODELS_FOLDER+"/parameters_G2.json", 
               model_G2, parameters_G2)
    print("Done! Model G2 and parameters saved!")
    print("Training Model G3 ...\n")
    model_G3, parameters_G3 = trainerG3.train()
    save_model(SAVED_MODELS_FOLDER+"/model_G3.bin", SAVED_MODELS_FOLDER+"/parameters_G3.json", 
               model_G3, parameters_G3)
    print("Done! Model G3 and parameters saved!")
    print(f"\nTraining took {convert_seconds(time.time() - start)}\n")
    
    

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        # Start the training program
        start(args)
        folder_path = os.path.join(os.getcwd(), "src/training/models_and_parameters") 
        destination_path = os.path.join(os.getcwd(), "models")
        print('\nSuccessful execution of training!')
        print('\n--> Please find the saved models and optimized hyperparameters here: ' + folder_path)
        print('\n\n move the models model_G1.bin, model_G2.bin, model_G3.bin from this folder into ',
              destination_path, "if you want to use the newly trained models for prediction.")
        sys.exit(0)
    except Exception as e:
        print('Program ran into an error: ', str(e))
        sys.exit(0)


        
if __name__ == '__main__':
    main()