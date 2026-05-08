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
from torch_geometric.data import Data

import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data, download_url, extract_tar
import numpy as np
import lzma
from ase.io import iread

import glob
import sys

from .data import XYZ


urls = {'200k': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar',
        '2M': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar',
        'val_id': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar',
        'val_ood_ads_traj': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_ads_trajectories.tar',
        'val_ood_cat_traj': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_cat_trajectories.tar',
        'val_ood_both_traj': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_both_trajectories.tar',
       }

subdirs = {'200k': 's2ef_train_200K/s2ef_train_200K',
           '2M': 's2ef_train_2M/s2ef_train_2M',
           'val_id': 's2ef_val_id/s2ef_val_id',
           'val_ood_ads_traj': 'is2res_val_ood_ads_trajectories',
           'val_ood_cat_traj': 'is2res_val_ood_cat_trajectories',
           'val_ood_both_traj': 'is2res_val_ood_both_trajectories',
          }

class XYZ4OCP(XYZ):
    """Data class for OCP models processed from extxyz files.
    
    Args:
        root (str): Root directory where the processed dataset should be saved.
        dataset (str): OCP dataset to download.
        **kwargs from XYZ base class
        total_energy (bool): Flag denoting whether to use total energy (True) or 
            reference energy against per-atom contributions (False).
    """ 

    def __init__(self, root: str, 
                 dataset: str = "200k",
                 **kwargs):

        self.dataset_id = dataset
        self.download_url = urls[self.dataset_id]
        self.raw_subdir = subdirs[self.dataset_id]
        
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return f'OC20-{self.dataset_id}.pt'

    def _download(self):
        if not os.path.isdir(os.path.join(self.raw_dir, self.raw_subdir)):
            path = download_url(self.download_url, self.raw_dir)
            extract_tar(path, os.path.join(self.raw_dir), mode='r')
            os.unlink(path)
            
    def _read_reference_energy(self, txt_path):
        with lzma.open(txt_path, mode='rt', encoding='utf-8') as fid:
            system_id, frame_number, reference_energy = [],[],[]
            for line in fid:
                system_id += [line.split(',')[0]]
                frame_number += [line.split(',')[1]]
                reference_energy += [line.split(',')[2]]
        df = pd.DataFrame({'system_id':system_id, 'frame_number':frame_number, 'reference_energy':reference_energy})
        df['reference_energy']=pd.to_numeric(df['reference_energy'])
        return df

    def _read_reference_energy_traj(self, txt_path):
        df=pd.read_csv(txt_path, names=['system_id','reference_energy'], dtype={'system_id':'str', 'reference_energy':'float32'})
        return df
        
    def _extract_properties(self, mol, i, ref):

        energy = mol.get_potential_energy()

        if self.total_energy:
            y = energy
        else:
            y = energy-ref['reference_energy']


        # set fixed atoms for relaxation
        fixed = np.zeros(len(mol))
        fixed[mol.constraints[0].todict()['kwargs']['indices']]=1

        data = Data(
            z=torch.IntTensor(mol.get_atomic_numbers()),
            pos=torch.Tensor(mol.get_positions()),
            y=torch.Tensor([y]),
            f=torch.Tensor(mol.get_forces()),
            cell=torch.Tensor(np.array(mol.cell))[None,...],
            tags=torch.IntTensor(mol.get_tags()),
            fixed=torch.IntTensor(fixed),
            e_total=torch.Tensor([energy]),
            free_energy=torch.Tensor([mol.info['free_energy']]),
            e_ref=torch.Tensor([ref['reference_energy']]),
            name=mol.symbols.get_chemical_formula(),
            system_id=ref['system_id'],
            idx=torch.IntTensor([i]),
            natoms=torch.IntTensor([len(mol)]),
            pbc=torch.Tensor(mol.pbc)
        )

        return data

    def _process_traj(self):
        ref_energy = self._read_reference_energy_traj(os.path.join(self.raw_dir, self.raw_subdir ,'system.txt'))

        m = 0
        data_list = []
        for p in tqdm(self.xyz_paths):
            system_id = p.split('/')[-1].replace('.extxyz.xz','')
            refE = ref_energy.loc[ref_energy['system_id']==system_id].iloc[0]
            
            # get properties
            for r, mol in enumerate(iread(p, index=":")):
                data_list.append(self._extract_properties(mol, m, refE))
                m+=1

        # collate and save as .pt file
        torch.save(self.collate(data_list), self.processed_paths[0])

    def _process_sets(self):
        # read in .xyz files and get properties
        m = 0
        data_list = []
        for p in tqdm(self.xyz_paths):
            ref_energy = self._read_reference_energy(p.replace('.extxyz.xz','.txt.xz'))

            # get properties
            for r, mol in enumerate(iread(p, index=":")):
                data_list.append(self._extract_properties(mol, m, ref_energy.iloc[r]))
                m+=1

        # collate and save as .pt file
        torch.save(self.collate(data_list), self.processed_paths[0])

    def process(self):
        # check if data has already been downloaded
        self._download()

        self.xyz_paths = glob.glob(os.path.join(self.raw_dir, self.raw_subdir ,'*.extxyz.xz'))

        if 'traj' in self.dataset_id:
            self._process_traj()
        else:
            self._process_sets()
