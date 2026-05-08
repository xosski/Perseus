'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
'''

import argparse

class CommonArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Training Arguments."
        )
        self.add_core_args()

    def get_parser(self) -> argparse.ArgumentParser:
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Common Arguments")
        self.parser.add_argument(
            "--savedir",
            required=True,
            type=str,
            help="Top-level directory to save training results.",
        )
        self.parser.add_argument(
            "--datadir",
            required=True,
            type=str,
            help="Top-level directory containing the dataset.",
        )
        self.parser.add_argument(
            "--split-file",
            default=None,
            type=str,
            help="Path to train-val-test split file (.npz format).",
        )
        self.parser.add_argument(
            "--total-energy",
            action="store_true",
            help="Train on total energy instead of normalized energy.",
        )
        self.parser.add_argument(
            "--quantile", 
            action="store_true", 
            help="Train a model with the quantile loss function for the 95th and 5th energy quantiles. NB: Currently only implemented for MACE models."
        )
        self.parser.add_argument(
            "--train-fraction",
            default=1.0,
            type=float,
            help="Fraction of training set to use each batch.",
        )
        self.parser.add_argument(
            "--train-forces",
            action="store_true",
            help="Include forces in loss function.",
        )
        self.parser.add_argument(
            "--batch-size",
            default=32,
            type=int,
            help="Number of samples per batch.",
        )
        self.parser.add_argument(
            "--dynamic-batch",
            action="store_true",
            help="Use dynamic batching. If applied, progress bar will be disabled. Currently only available for single GPU training.",
        )
        self.parser.add_argument(
            "--max-epochs", 
            default=500,
            type=int,
            help="Maximum number of epochs for training.",
        )
        self.parser.add_argument(
            "--min-epochs", 
            default=1,
            type=int,
            help="Minimum number of epochs for training.",
        )
        self.parser.add_argument(
            "--max-time", 
            default=None,
            type=str,
            help="Maximum amount of time for training (format 00:12:00:00).",
        )
        self.parser.add_argument(
            "--lr",
            default=0.001,
            type=float,
            help="Initial learning rate.",
        )
        self.parser.add_argument(
            "--lr-patience",
            default=10,
            type=int,
            help="Learning rate paticence: number of epochs before decreasing learning rate.",
        )
        self.parser.add_argument(
            "--es-patience",
            default=25,
            type=int,
            help="Early stopping patience: number of epochs without improvement in val loss before stopping.",
        )
        self.parser.add_argument(
            "--swa", 
            action="store_true",
            help="Apply Stochastic Weight Averaging.",
        )
        self.parser.add_argument(
            "--clip", 
            default=200, 
            type=int, 
            help="Gradient clipping value.",
        )
        self.parser.add_argument(
            "--progress-bar", 
            action="store_true",
            help="Display progress bar during training.",
        )
        self.parser.add_argument(
            "--seed", 
            default=42, 
            type=int, 
            help="Seed for torch, cuda, numpy"
        )
        self.parser.add_argument(
            "--full-reproducibility",
            action="store_true", 
            help="Torch Lightning flag for full reproducibility. NB: Applying flag slows training."
        )
        self.parser.add_argument(
            "--amp", 
            action="store_true", 
            help="Use mixed-precision training. Force scaling will be applied."
        )
        self.parser.add_argument(
            "--umap", 
            action="store_true", 
            help="Train umap and collect embeddings."
        )
        self.parser.add_argument(
            "--for-lammps", 
            action="store_true", 
            help="Compile trained model into LAMMPS-compatible format."
        )

args = CommonArgs()
