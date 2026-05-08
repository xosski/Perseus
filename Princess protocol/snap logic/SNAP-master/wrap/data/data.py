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
import glob
import json
import h5py
import warnings 
import numpy as np
from tqdm import tqdm
from ase.io import read
import logging
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.data.summary import Summary

def read_stats(path):
    """Extract per-atom energy references."""
    if not os.path.isfile(path):
        return {}
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    try:
        # when written as numbers
        data['atomic_energies']={int(k):v for k,v in data['atomic_energies'].items()}
    except:
        # when written as a string
        data['atomic_energies']={int(i.split(':')[0]):float(i.split(':')[1]) for i in data['atomic_energies'][1:-1].split(',')}
        data['atomic_numbers']=[int(i) for i in data['atomic_numbers'][1:-1].split(',')]

    return data

class XYZ(InMemoryDataset):
    """Dataset class for datasets that can fit into CPU memory.
    
    Args:
        root (str): 
            Root directory where the processed dataset should be saved.
        total_energy (bool): 
            Flag denoting whether to use total energy (True) or 
            reference energy against per-atom contributions (False).
    """ 

    def __init__(self, root: str, total_energy: bool = False):
        # set up file name tag
        self.tag = 'data'

        # specify whether total energy needs to be referenced
        self.total_energy = total_energy
       
        if not self.total_energy:
            self.tag += '-ref'

        super().__init__(root)
        self._data = torch.load(self.processed_paths[0])
        self.summary()

    def summary(self):
        """Compute summary of graph sizes."""
        dataset_summary = Summary.from_dataset(self._data)
        logging.info(dataset_summary)
        self.max_num_nodes = int(dataset_summary.num_nodes.max)
        self.max_num_edges = int(dataset_summary.num_edges.max)
        
    @property
    def processed_file_names(self):
        return f'{self.tag}.pt'
        
    @property
    def raw_file_names(self):
        return '*xyz'

    def _extract_properties(self, mol, i, refs):     
        """Extract properties used by your model of choice."""

        # Reference against per-atom energies if desired
        energy = mol.get_potential_energy()
        if not self.total_energy:
            reference_energy = sum([refs['atomic_energies'][elem] for elem in mol.get_atomic_numbers()])
            energy -= reference_energy
        else:
            reference_energy = 0

        # Keys in Data object must have same name as those used in your model of choice
        return Data(z=torch.IntTensor(mol.get_atomic_numbers()),
                    pos=torch.Tensor(mol.get_positions()).type(torch.float32),
                    y=torch.Tensor([energy]).type(torch.float32),
                    pbc=torch.Tensor(np.array(mol.pbc)),
                    cell=torch.Tensor(np.array(mol.cell)).type(torch.float32),
                    f=torch.Tensor(mol.get_forces()).type(torch.float32),
                    reference_energy=torch.Tensor([reference_energy]).type(torch.float32),
                    n_atoms=torch.Tensor([len(mol)]).type(torch.float32),
                    idx=torch.IntTensor([i]),
                    name=mol.symbols.get_chemical_formula(),
                   )

            
    def _read_atom_references(self):
        self.atom_refs = {}
        sample_dirs = next(os.walk(self.raw_dir))[1]
        
        for key in sample_dirs:
            if os.path.isfile(os.path.join(self.raw_dir,key,'statistics.json')):
                self.atom_refs[key] = read_stats(os.path.join(self.raw_dir,key,'statistics.json'))

            
    def process(self):
        """Read in extxyz files and get properties."""
        
        self.xyz_paths = glob.glob(os.path.join(self.raw_dir, '*', self.raw_file_names))

        # Read in precomputed per-atom references
        if not self.total_energy:
            self._read_atom_references()
        else:
            sample_set = 'None'
            self.atom_refs = {'None':'None'}
        
        m=0
        data_list=[]
        for p in tqdm(self.xyz_paths):
            if not self.total_energy:
                sample_set = p.split('/')[-2]
                if sample_set not in self.atom_refs.keys():
                    warnings.warn(f"{sample_set} does not have associated per-atom reference values -- skipping sample")
                    continue

            for r, mol in enumerate(read(p, index=":")):
                newsample=self._extract_properties(mol, m, self.atom_refs[sample_set])
                data_list.append(newsample)
                m+=1

        # Save as .pt file
        torch.save(data_list, self.processed_paths[0])

    def generate_split(self, filename="split.npz"):
        """Generate randomized 80:10:10 train:val:test split."""
        data = torch.load(self.processed_paths[0])
        n_samples = len(data)

        indices = np.arange(0, n_samples, 1, dtype=int)
        pred_indices = indices.copy()
        
        np.random.shuffle(indices)

        n_train = int(np.floor(n_samples*0.8))
        n_val = int(np.floor(n_samples*0.1))

        np.savez(file=filename,
                 train_idx=indices[:n_train],
                 val_idx=indices[n_train:n_train+n_val],
                 test_idx=indices[n_train+n_val:],
                 pred_idx=pred_indices,
                )

