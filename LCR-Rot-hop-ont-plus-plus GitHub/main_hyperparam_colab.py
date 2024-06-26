import argparse
import json
import os
import pickle
from typing import Optional

import torch
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split


class HyperOptManager:
    """A class that performs hyperparameter optimization and stores the best states as checkpoints."""

    def __init__(self, year: int, val_ont_hops: Optional[int]):
        self.year = year
        self.n_epochs = 20
        self.val_ont_hops = val_ont_hops

        self.eval_num = 0
        self.best_loss = None
        self.best_hyperparams = None
        self.best_state_dict = None
        self.no_improvement_count = 0
        self.max_evals = 100
        self.trials = Trials()

        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else 'cpu')

        self.__checkpoint_dir = f"data/checkpoints/{year}_epochs{self.n_epochs}"
        if os.path.isdir(self.__checkpoint_dir):
            try:
                self.best_state_dict = torch.load(f"{self.__checkpoint_dir}/state_dict.pt")
                with open(f"{self.__checkpoint_dir}/hyperparams.json", "r") as f:
                    self.best_hyperparams = json.load(f)
                with open(f"{self.__checkpoint_dir}/trials.pkl", "rb") as f:
                    self.trials = pickle.load(f)
                    self.eval_num = len(self.trials)
                with open(f"{self.__checkpoint_dir}/loss.txt", "r") as f:
                    self.best_loss = float(f.read())
                print(f"Resuming from previous checkpoint {self.__checkpoint_dir} with best loss {self.best_loss}")
            except IOError:
                raise ValueError(f"Checkpoint {self.__checkpoint_dir} is incomplete, please remove this directory")
        else:
            print("Starting from scratch")

    def run(self):
        space = [
            hp.choice('learning_rate', [0.005, 0.001, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
            hp.quniform('dropout_rate', 0.25, 0.75, 0.1),
            hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
            hp.choice('weight_decay', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
            hp.choice('lcr_hops', [3]), 
            hp.choice('gamma', [0])
        ]
        best = fmin(self.objective, space=space, algo=tpe.suggest, trials=self.trials, max_evals=self.max_evals, show_progressbar=False)

    def objective(self, hyperparams):
        if self.no_improvement_count >= 30 or self.eval_num >= self.max_evals:
            return {'loss': 0.0, 'status': STATUS_OK}  # Simulate a complete optimization run

        self.eval_num += 1
        print(f"\n\nEval {self.eval_num} with hyperparams {hyperparams}")

        # [Setup and training logic remains unchanged]

        validation_accuracy = val_n_correct / val_n

        if best_accuracy is None or validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check and update the best loss
        if self.best_loss is None or validation_accuracy > -self.best_loss:
            self.best_loss = -validation_accuracy
            self.best_hyperparams = hyperparams
            self.best_state_dict = (model.state_dict(), optimizer.state_dict())

        return {
            'loss': -validation_accuracy,  # Objective to minimize (negate because Hyperopt minimizes)
            'status': STATUS_OK,
            'space': hyperparams
        }

    # [Rest of your class code remains unchanged]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2014, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--val-ont-hops", default=None, type=int, required=False,
                        help="The number of hops to use in the validation phase")
    args = parser.parse_args()

    opt = HyperOptManager(args.year, args.val_ont_hops)
    opt.run()

if __name__ == "__main__":
    main()
