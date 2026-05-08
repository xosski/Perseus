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

import os
import json
import torch
from torch_geometric.data import Data, Batch
import numpy as np
import sys
import logging
import glob

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator 
from ase.data import atomic_numbers

from mace.tools.scripts_utils import get_config_type_weights
from mace.data import load_from_xyz
from mace.data.neighborhood import get_neighborhood
from mace.data.utils import compute_average_E0s
from mace.tools import utils, AtomicNumberTable, to_one_hot, atomic_numbers_to_indices
from mace.calculators import mace_mp

from .data import XYZ, XYZLarge, read_stats


def check_stats(datadir, model='medium'):
    for ddir in glob.glob(os.path.join(datadir,'raw','*')):
        if os.path.isfile(os.path.join(ddir, 'statistics.json')):
            logging.info(f'...statistics file present for {ddir.split("/")[-1]}.')
        else:
            logging.info(f'...statistics file being computed for {ddir.split("/")[-1]}.')
            compute_atomic_energy_references(ddir, model_type=model)

def compute_atomic_energy_references(datadir, model_type='medium'):
    """Compute per-atom references to normalize energy values.
    NB: This step can take a long time if total number of structures is large.
    
    Args:
        datadir (str): Directory containing all extxyz files to process.
        model_type (str): Size of MACE foundational model ("small", "medium", or "large").
    """

    # Extract z_table from pre-trained model.
    model = mace_mp(model=model_type, default_dtype="float64", device="cpu")
    z_table = utils.AtomicNumberTable([int(z) for z in model.models[0].atomic_numbers])

    # Concatenate all files in folder for read-in by MACE code
    os.system(f"cat {os.path.join(datadir, '*xyz')} > {os.path.join(datadir, 'temp.xyz')}")
    
    atomic_energies_dict, all_train_configs = load_from_xyz(
            file_path=os.path.join(datadir, 'temp.xyz'),
            config_type_weights=get_config_type_weights('{"Default":1.0}'),
            energy_key="energy",
            forces_key="forces",
            extract_atomic_energies=True,
            keep_isolated_atoms=True,
        )
    
    atoms_list=compute_average_E0s(all_train_configs, z_table)

    # Save to file following MACE formatting
    d = {"atomic_energies": str(atoms_list), "atomic_numbers": str(list(atoms_list.keys()))}
    with open(os.path.join(datadir, 'statistics.json'), 'w') as f:
        json.dump(d, f)

    # Delete combined file
    os.remove(os.path.join(datadir, 'temp.xyz'))


class XYZ4MACEMP(XYZ):
    """Data class for MACE models processed from extxyz files.
    
    Args:
        root (str): Root directory where the processed dataset should be saved.
        model (str): Size of MACE foundational model ("small", "medium", or "large").

        **kwargs from XYZ base class
        total_energy (bool): Flag denoting whether to use total energy (True) or 
            reference energy against per-atom contributions (False).
    """ 

    def __init__(self, root: str, 
                 model: str = 'medium', 
                 total_energy: bool = False,
                 **kwargs):

        self.root = root
        self.model_type = model
        self._extract_model_properties()

        if not total_energy:
            self._get_statistics()

        super().__init__(self.root, **kwargs)
        
    def _extract_model_properties(self):
        model = mace_mp(model=self.model_type, default_dtype="float64", device="cpu")
        self.z_table = utils.AtomicNumberTable([int(z) for z in model.models[0].atomic_numbers])
        self.r_max = model.models[0].r_max.item()
        del model

    def _get_statistics(self):
        # Get atomic energy refs if not already present
        check_stats(self.root, model=self.model_type)

    def _extract_properties(self, mol, i, atomic_energies, dtype=torch.float32):     
        # extract graph info using MACE functions
        edge_index, shifts, unit_shifts, cell = get_neighborhood(positions=mol.get_positions(), 
                                                           cutoff=self.r_max, 
                                                           pbc=mol.pbc, 
                                                           cell=mol.cell
                                                          )

        indices = atomic_numbers_to_indices(mol.get_atomic_numbers(), z_table=self.z_table)
        one_hot = to_one_hot(torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
                             num_classes=len(self.z_table),
                            )

        # reference against atom energies
        energy = mol.get_potential_energy()

        # reference against atom energies
        reference_energy = sum([atomic_energies['atomic_energies'][elem] for elem in mol.get_atomic_numbers()])
        if not self.total_energy:
            energy -= reference_energy

        return Data(edge_index=torch.tensor(edge_index, dtype=torch.long),
                    node_attrs=one_hot.type(dtype),
                    z=torch.tensor(mol.get_atomic_numbers(), dtype=torch.long),
                    positions=torch.Tensor(mol.get_positions()).type(dtype),
                    shifts=torch.Tensor(shifts).type(dtype),
                    unit_shifts=torch.Tensor(unit_shifts).type(dtype),
                    cell=torch.Tensor(np.array(cell)).type(dtype),
                    forces=torch.Tensor(mol.get_forces()).type(dtype),
                    energy=torch.Tensor([energy]).type(dtype),
                    reference_energy=torch.Tensor([reference_energy]).type(dtype),
                    n_atoms=torch.Tensor([len(mol)]).type(dtype),
                    idx=torch.IntTensor([i]),
                )

