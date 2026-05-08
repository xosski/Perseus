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
import copy
import logging
import torch
from torch.nn import functional as F
from torch_scatter import scatter_add
import urllib
from datetime import datetime
from e3nn import o3
import numpy as np
from mace.tools.utils import AtomicNumberTable
from mace.modules.utils import extract_invariant
from mace.modules.blocks import AtomicEnergiesBlock
from wrap.train.args import CommonArgs
from .base import NNP, weights_init
from .loss import UniversalLoss, SmoothPinballLoss

module_dir = os.path.dirname(__file__)
local_model_path = os.path.join(
    module_dir, "foundations_models/mace-mpa-0-medium.model"
)

emb_size_map = {"large": 1024, "medium": 512, "small": 128}
emb_irrep_map = {"large": '256x0e+256x1o', "medium": '128x0e+128x1o', "small": '128x0e'}
emb_mlp_map = {"large": '256x0e', "medium": '128x0e', "small": '128x0e'}


class ModelArgs(CommonArgs):
    """MACE-MP-specific command line flags."""
    def __init__(self):
        super().__init__()

        self.add_model_args()
        
    def add_model_args(self):
        self.parser.add_argument_group("MACE-MP Arguments")
        self.parser.add_argument(
            "--model",
            default="medium",
            choices=["small", "medium", "large", 
                     "small-0b", "medium-0b", 
                     "small-0b2", "medium-0b2", "large-0b2", 
                     "medium-0b3", "medium-mpa-0"],
            help="MACE-MP model size.",
        )
        self.parser.add_argument(
            "--checkpoint",
            default=None,
            type=str,
            help="Path to checkpoint to resume training.",
        )
        self.parser.add_argument(
            "--freeze-head",
            action="store_true", 
            help="Freeze interaction head during training.",
        )
        self.parser.add_argument(
            "--fresh-start",
            action="store_true", 
            help="Re-initialize model weights before training.",
        )
        self.parser.add_argument(
            "--default-dtype",
            default="float32",
            choices=["float32", "float64"],
            help="Default dtype for model weights.",
        )

modelargs = ModelArgs()

class MACEMP(NNP):
    """MACE-MP model pretrained on the Materials Project database (89 elements).
       See https://github.com/ACEsuit/mace-mp for all models.
       
       If using this model, please cite the following:
        - MACE-MP by Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena,
            Dávid P. Kovács, Janosh Riebesell, et al., 2023, arXiv:2401.00096               
    """
    def __init__(
        self,
        dataset,
        split_file: str,
        model: str = "medium-mpa-0", 
        freeze_head: bool = False,
        fresh_start: bool = False,
        default_dtype: str = "float32",
        return_forces: bool = True,
        **kwargs
    ):
        super().__init__(dataset, split_file, **kwargs)

        self.freeze_head = freeze_head
        self.foundation = model
        self.fresh_start = fresh_start
        self.default_dtype = default_dtype
        self.torch_dtype = torch.double if self.default_dtype=="float64" else torch.float
        self.return_forces = return_forces

        # load foundation model
        self.load_foundation()

            
    @property
    def loss_fn(self):
        if self.train_forces:
            return UniversalLoss(energy_weight=1.0, forces_weight=10.0, huber_delta=0.01)
        else:
            return torch.nn.HuberLoss(reduction="mean", delta=0.01)

    def load_foundation(self):
        # load model
        self.model = torch.load(self._model_path(self.foundation))
        for param in self.model.parameters():
            param = param.type(self.torch_dtype)
            
        if self.default_dtype == "float64":
            self.model = self.model.double()
        elif self.default_dtype == "float32":
            self.model = self.model.float()
                
        self.z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.cutoff = float(self.model.r_max)

        if self.fresh_start:
            self.reset_parameters()
            
        if self.freeze_head:
            self._freeze_head()
            
    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.interactions.requires_grad_(False)
        self.model.products.requires_grad_(False)
        self.model.node_embedding.requires_grad_(False)
        self.model.radial_embedding.requires_grad_(False)
        self.model.spherical_harmonics.requires_grad_(False)
        self.model.atomic_energies_fn.requires_grad_(False)
        self.model.scale_shift.requires_grad_(False)
        logging.info('...interaction head weights frozen.')

    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        for child in self.model.children():
            if isinstance(child, torch.nn.ModuleList):
                for layer in child.children():
                    if hasattr(layer, 'children'):
                        for sublayer in layer.children():
                            weights_init(sublayer)
                    else:
                        if hasattr(layer, 'weight'):
                            weights_init(layer)
            else:
                if hasattr(child, 'children'):
                    for layer in child.children():
                        if hasattr(layer, 'weight'):
                            weights_init(layer)
                        else:
                            if hasattr(layer, 'children'):
                                for sublayer in layer.children():
                                    weights_init(sublayer)
                                    
        logging.info('...all model weights reset.')

    
    def _model_path(self, model):
        """Download pretrained MACE model."""
        if model in (None, "medium-mpa-0") and os.path.isfile(local_model_path):
            logging.info(f"Using local medium Materials Project MACE model for MACECalculator {model}")
            return local_model_path
            
        try:
            urls = {
                    "small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
                    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
                    "large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
                    "small-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
                    "medium-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model",
                    "small-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model",
                    "medium-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
                    "large-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
                    "medium-0b3": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
                    "medium-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
                    }
            
            checkpoint_url = (
                urls.get(model, urls["medium-mpa-0"])
                if model
                in (
                    None,
                    "small",
                    "medium",
                    "large",
                    "small-0b",
                    "medium-0b",
                    "small-0b2",
                    "medium-0b2",
                    "large-0b2",
                    "medium-0b3",
                    "medium-mpa-0",
                )
                else model
            )
            
            cache_dir = os.path.expanduser("~/.cache/mace")
            checkpoint_url_name = "".join(c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_")
            cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
            
            if not os.path.isfile(cached_model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                logging.info(f"Downloading MACE model from {checkpoint_url!r}")
                _, http_msg = urllib.request.urlretrieve(checkpoint_url, cached_model_path)
                if "Content-Type: text/html" in http_msg:
                    raise RuntimeError(f"Model download failed, please check the URL {checkpoint_url}")
                logging.info(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
            logging.info(f"Using Materials Project MACE for MACECalculator with {model}")
            
        except Exception as exc:
            raise RuntimeError("Model download failed and no local model found") from exc
            
        return model
    
    def reset_E0s(self, stats):
        for i in range(len(self.model.atomic_energies_fn.atomic_energies)):
            self.model.atomic_energies_fn.atomic_energies[i]=0.
        for k in stats.keys():
            self.model.atomic_energies_fn.atomic_energies[k-1]=stats[k]
    
    def forward(self, batch):
        output = self.model(batch, training=self.freeze_head, compute_force=False, compute_stress=False)
        return output['energy']

    def training_step(self, batch, batch_index):
        start_time = datetime.now()
        # returns forces and energies
        data = batch.to_dict()
        self._set_grad_and_dtype(data, grad=True)
                
        E_out = self.forward(data)

        if self.train_forces:
            F_out = -torch.autograd.grad(E_out, data["positions"], grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=True)[0]
            E_loss, F_loss = self.loss(batch, [E_out, F_out])
            loss = E_loss + F_loss
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'train_loss': torch.round(loss, decimals=6).detach(), 
                        'E_loss_contrib': torch.round(E_loss, decimals=6).detach(),
                        'F_loss_contrib': torch.round(F_loss, decimals=6).detach(),
                       }
        else:
            loss = self.loss(batch.energy, E_out)
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'train_loss': torch.round(loss, decimals=6).detach()
                       }
            
        self.record_log(log_dict, batch_size=batch.energy.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        start_time = datetime.now()
        data = batch.to_dict()
        
        if self.train_forces:
            torch.set_grad_enabled(True)
        
        self._set_grad_and_dtype(data, grad=self.train_forces)   
                
        E_out = self.forward(data)
        
        if self.train_forces:
            F_out = -torch.autograd.grad(E_out, data["positions"], grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=True)[0]
            E_loss, F_loss = self.loss(batch, [E_out, F_out])
            loss = E_loss + F_loss
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'val_loss': torch.round(loss, decimals=6).detach(), 
                        'E_loss_contrib': torch.round(E_loss, decimals=6).detach(),
                        'F_loss_contrib': torch.round(F_loss, decimals=6).detach(),
                       }
        else:
            loss = self.loss(batch.energy, E_out)
            log_dict = {'time': (datetime.now()-start_time).total_seconds(),
                        'val_loss': torch.round(loss, decimals=6).detach()
                       }
            
        self.record_log(log_dict, batch_size=batch.energy.shape[0])
        return loss
        
    def test_step(self, batch):
        """Energy-only prediction"""
        batch = batch.to_dict()
        self._set_grad_and_dtype(batch, grad=False) 
        E_out = self.forward(batch)
        return E_out.detach()
        
    def predict_step(self, batch):
        # returns forces and energies
        batch = batch.to_dict()
        self._set_grad_and_dtype(batch, grad=self.return_forces)
        
        if self.return_forces:
            torch.set_grad_enabled(True)
            E_out = self.forward(batch)
            F_out = -torch.autograd.grad(E_out, batch["positions"], grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=False)[0]
            return E_out.detach(), F_out
        else:

            E_out = self.forward(batch)
            return E_out.detach()

    def _set_grad_and_dtype(self, batch, grad=True):
        '''
        for param in batch:
            try:
                if grad:
                    param[1].requires_grad_(True)
                param[1] = param[1].type(self.torch_dtype)
            except:
                pass      
        '''
        for k,v in batch.items():
            try:
                batch[k].requires_grad_(True)
                batch[k] = v.type(self.torch_dtype)
            except:
                pass
            
    def get_descriptors(self, batch=None, invariants_only=True, num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param invariants_only: bool, if True only the invariant descriptors are returned
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if batch is None:
            raise ValueError("Batch not sent.")
    
        num_interactions = self.model.num_interactions.item()
        if num_layers == -1:
            num_layers = num_interactions
    
        output = self.model(batch, training=False, compute_force=False, compute_stress=False)
        descriptors = output['node_feats']
    
        irreps_out = o3.Irreps(str(self.model.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        
        # Equivariant features not created for the last layer
        per_layer_features[-1] = num_invariant_features
    
        if invariants_only:
            descriptors = extract_invariant(descriptors,
                                            num_layers=num_layers,
                                            num_features=num_invariant_features,
                                            l_max=l_max,
                                           )
    
        to_keep = np.sum(per_layer_features[:num_layers])
    
        return descriptors[:to_keep].detach().cpu().numpy()


class QuantileMACE(MACEMP):
    """
        Train towards energy only!
    """
    def __init__(
        self,
        dataset,
        split_file: str,
        **kwargs,
    ):
        super().__init__(dataset, split_file, **kwargs)
                
        # initialize readout layers
        self.emb_size = emb_size_map[self.foundation.split('-')[0]]
        self.readouts_lower = self.model.readouts
        self.readouts_upper = copy.deepcopy(self.model.readouts)

    @property
    def loss_fn(self):
        return SmoothPinballLoss()

    def _collect_readout(self, readouts, node_feats, node_heads):
        # readouts for each interaction block
        out_int1 = readouts[0](node_feats[...,:self.emb_size])
        out_int2 = readouts[1](node_feats[...,self.emb_size:])

        # sum readouts
        node_inter_es = torch.sum(torch.stack([out_int1, out_int2], dim=0), dim=0)

        # scale shift
        node_inter_es = self.model.scale_shift(node_inter_es, node_heads)

        return node_inter_es
    
    def forward(self, batch):
        node_heads = (batch["head"][batch["batch"]] if "head" in batch else torch.zeros_like(batch["batch"]))

        output = self.model(batch, training=self.freeze_head, compute_force=False, compute_stress=False,)
        node_feats = output['node_feats']
        
        node_inter_es_lower = self._collect_readout(self.readouts_lower, node_feats, node_heads)
        node_inter_es_lower = scatter_add(node_inter_es_lower, batch["batch"], dim=0)
        
        node_inter_es_upper = self._collect_readout(self.readouts_upper, node_feats, node_heads)
        node_inter_es_upper = scatter_add(node_inter_es_upper, batch["batch"], dim=0)

        return torch.vstack([node_inter_es_lower[...,0], node_inter_es_upper[...,0]])


    def E_loss(self, data, pred_energies):
        return self.loss_fn(pred_energies, data.energy.view(1,-1))
    
    def training_step(self, batch, batch_index):
        data = batch.to_dict()
        self._set_grad_and_dtype(data, grad=True)   
        E_out = self.forward(data)

        loss = self.loss(batch['energy'], E_out)

        self.log("train_loss", torch.round(loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.energy.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        data = batch.to_dict()
        self._set_grad_and_dtype(data, grad=False)   
        E_out = self.forward(data)
        loss = self.loss(batch['energy'], E_out)

        self.log("val_loss", torch.round(loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.energy.shape[0])
        return loss       
        
