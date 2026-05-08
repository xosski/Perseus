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
import sys
import numpy as np 
import pandas as pd
import random
import torch
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
from datetime import datetime 
from wrap.train import scale, batching
from .loss import UniversalLoss

def weights_init(m):
    if hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data)
    else:
        if hasattr(m, 'parameters'):
            for p in m.parameters():
                torch.nn.init.normal_(p)

class NNP(pl.LightningModule):
    """Base NNP class

    Args:
        dataset (Dataset): Dataset from which to load the data.
        split_file (str): Path to dataset split file.
        batch_size (int, optional): How many samples per batch to load.
            (default: ``64``)
        lr (float, optional): Optimizer learning rate.
            (default: ``0.001``)
        lr_patience (int, optional): Number of epochs with no improvement after which the learning rate will be reduced.
            (default: ``10``)
        train_forces (bool, optional): Flag specifying if force error should be included in the loss function.
            (default: ``False``)
        return_forces (bool, optional): Flag specifying whether forces should be returned in the predict step.
            (default: ``False``)
        dynamic_batch (bool, optional): Flag specifying whether dynamic batching should be used. If :obj: ``True``, progress 
            bars will be infinante and cannot be used.
            (default: ``False``) 
        amp (bool, optional): Flag specifying whether AMP is used. If :obj: ``True``, forces will be scaled.
            (default: ``False``)
    """
    def __init__(
        self,
        dataset,
        split_file: str,
        batch_size: int = 64,
        lr: float = 0.001,
        lr_patience: int = 10,
        train_forces: bool = False,
        return_forces: bool = False,
        dynamic_batch: bool = False,
        amp: bool = False,
        dataset_to_calc = False,
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset[0]
        self.max_num_nodes = dataset.max_num_nodes
        self.max_num_edges = dataset.max_num_edges
        self.load_splits(split_file)

        self.lr = lr
        self.lr_patience = lr_patience
        self.batch_size = batch_size
        self.train_forces = train_forces
        self.return_forces = return_forces
        self.dynamic_batch = dynamic_batch
        self.amp = amp
        self.dataset_to_calc = dataset_to_calc

    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        self.model.reset_parameters()
        logging.info('...all model weights reset.')
            
    @property
    def loss_fn(self):
        if self.train_forces:
            return UniversalLoss(energy_weight=1.0, forces_weight=10.0, huber_delta=0.01)
        else:
            return torch.nn.HuberLoss(reduction="mean", delta=0.01)
    @property
    def force_scaler(self):
        return scale.ForceScaler(enabled=self.amp)
    
    def forward(self, batch):
        raise NotImplementedError

    def loss(self, data, preds):
        return self.loss_fn(data, preds)

    def record_log(self, log_dict, batch_size=None):
        # add all items in log_dict to log
        for k,v in log_dict.items():
            self.log(k, v, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
    
    def training_step(self, batch, batch_index):
        start_time = datetime.now()
        
        if self.train_forces:
            batch.pos.requires_grad_(True)
            
        E_out = self.forward(batch)

        if self.train_forces:
            if self.amp:
                F_out = self.force_scaler.calc_forces_and_update(E_out, batch.pos)
            else:
                F_out = -torch.autograd.grad(E_out, batch.pos, grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=True)[0]
            E_loss, F_loss = self.loss(batch, [E_out, F_out])
            loss = E_loss + F_loss
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'train_loss': torch.round(loss, decimals=6).detach(), 
                        'E_loss_contrib': torch.round(E_loss, decimals=6).detach(),
                        'F_loss_contrib': torch.round(F_loss, decimals=6).detach(),
                       }
        else:
            loss = self.loss(batch.y, E_out)
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'train_loss': torch.round(loss, decimals=6).detach()
                       }
            
        self.record_log(log_dict, batch_size=batch.y.shape[0])

        # Shuffle training set indices
        self._shuffle_train()
        
        return loss

    def validation_step(self, batch, batch_index):
        start_time = datetime.now()
        
        if self.train_forces:
            torch.set_grad_enabled(True)
            batch.pos.requires_grad_(True)
            
        E_out = self.forward(batch)

        if self.train_forces:
            if self.amp:
                F_out = self.force_scaler.calc_forces_and_update(E_out, batch.pos)
            else:
                F_out = -torch.autograd.grad(E_out, batch.pos, grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=True)[0]
            E_loss, F_loss = self.loss(batch, [E_out, F_out])
            loss = E_loss + F_loss
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'val_loss': torch.round(loss, decimals=6).detach(), 
                        'E_loss_contrib': torch.round(E_loss, decimals=6).detach(),
                        'F_loss_contrib': torch.round(F_loss, decimals=6).detach(),
                       }
        else:
            loss = self.loss(batch.y, E_out)
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'val_loss': torch.round(loss, decimals=6).detach()
                       }

        # record loss to log
        self.record_log(log_dict, batch_size=batch.y.shape[0])
        
        return loss
        
    def test_step(self, batch):
        """Energy-only prediction."""
        E_out = self.forward(batch)
        return E_out.detach()
        
    def predict_step(self, batch):
        if self.return_forces:
            torch.set_grad_enabled(True)
            batch.pos.requires_grad_(True)
            E_out = self.forward(batch)
            F_out = -torch.autograd.grad(E_out, batch.pos, grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=False)[0]
            return E_out.detach(), F_out
        else:
            E_out = self.forward(batch)
            return E_out.detach()

    def load_splits(self, split_file=None):
        # Look for default split file if no split file is passed
        split_file = os.path.join(self.dataset.processed_dir, 'split.npz') if split_file==None else split_file

        if not os.path.isfile(split_file):
            sys.exit("Split file missing.")

        split = np.load(split_file)

        self.train_idx = split['train_idx']
        self.val_idx = split['val_idx']
        self.test_idx = split['test_idx']
        
        if 'pred_idx' in split.keys():
            self.pred_idx = split['pred_idx']
        else:
            self.pred_idx = self.test_idx

    def batched_sampler(self, dataset, shuffle=False, num_workers=1):
        sampler = batching.CyclicDynamicBatchSampler(torch.hstack([d.n_atoms for d in dataset]).numpy(), shuffle=shuffle, 
                                                     max_batch_tokens=self.batch_size*self.max_num_nodes, num_replicas=1, rank=0)
        sampler.set_epoch(self.current_epoch)
        return batching.DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)

    def train_dataloader(self):
        dataset = [self.dataset[i] for i in self.train_idx]
        if self.dynamic_batch:
            return self.batched_sampler(dataset, shuffle=True, num_workers=0)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        dataset = [self.dataset[i] for i in self.val_idx]
        if self.dynamic_batch:
            return self.batched_sampler(dataset, shuffle=False, num_workers=0)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader([self.dataset[i] for i in self.test_idx], batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader([self.dataset[i] for i in self.pred_idx], batch_size=self.batch_size, shuffle=False, num_workers=4)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience),
                    "monitor": "val_loss",
                },
            }
        
    def _shuffle_train(self):
        random.shuffle(self.train_idx)
        
    def _check_batch_format(self, data):
        if hasattr(data, 'batch'):
            return torch.zeros_like(data.z, dtype=torch.long) if data.batch is None else data.batch
        else:
            return torch.zeros_like(data.z, dtype=torch.long)

