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
import torch
import yaml
import logging 

from .base import NNP
from .loss import WeightedLoss, L2MAELoss
from .ocpmodels.models.gemnet_oc import gemnet_oc
from .ocpmodels.models.gemnet import gemnet
from .ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from .ocpmodels.models.schnet import SchNetWrap


def load_pretrained_OCP(dataset, 
                        split_file, 
                        model_params, 
                        model_weights, 
                        model_scale=None, 
                        data_scale=None, 
                        **kwargs):
    """Load pretrained OCP models.

    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        model_params: str
            Path to .yml file provided by OCP containing information on
            the pretrained model.
        model_weights: str
            Path to .pt file provided by OCP containing the model weights.
        model_scale: str
            Path to .json file provided by OCP containing model scaling factors.
        data_scale: str
            Path to .yml file provided by OCP containing data scaling factors.
        **kwargs:
            General training arguments supplied to NNP.
    """
    with open(model_params, 'r') as f:
        model_params = yaml.safe_load(f)

    model_name = model_params['model']['name']

    if model_name == 'gemnet_t':        
        model = GemNetT(dataset=dataset,
                               split_file=split_file,
                               num_spherical = model_params['model']['num_spherical'],
                               num_radial = model_params['model']['num_radial'],
                               num_blocks = model_params['model']['num_blocks'],
                               emb_size_atom = model_params['model']['emb_size_atom'],
                               emb_size_edge = model_params['model']['emb_size_edge'],
                               emb_size_trip = model_params['model']['emb_size_trip'],
                               emb_size_rbf = model_params['model']['emb_size_rbf'],
                               emb_size_cbf = model_params['model']['emb_size_cbf'],
                               emb_size_bil_trip = model_params['model']['emb_size_bil_trip'],
                               num_before_skip = model_params['model']['num_before_skip'],
                               num_after_skip = model_params['model']['num_after_skip'],
                               num_concat = model_params['model']['num_concat'],
                               num_atom = model_params['model']['num_atom'],
                               cutoff = model_params['model']['cutoff'],
                               max_neighbors = model_params['model']['max_neighbors'],
                               rbf = model_params['model']['rbf'],
                               envelope = model_params['model']['envelope'],
                               cbf = model_params['model']['cbf'],
                               extensive = model_params['model']['extensive'],
                               output_init = model_params['model']['output_init'],
                               activation = model_params['model']['activation'],
                               scale_file = model_scale,
                               **kwargs,
                              )
    elif model_name == 'gemnet_oc':
        model = GemNetOC(dataset=dataset,
                         split_file=split_file,
                         num_spherical = model_params['model']['num_spherical'],
                         num_radial = model_params['model']['num_radial'],
                         num_blocks = model_params['model']['num_blocks'],
                         emb_size_atom = model_params['model']['emb_size_atom'],
                         emb_size_edge = model_params['model']['emb_size_edge'],
                         emb_size_trip_in = model_params['model']['emb_size_trip_in'],
                         emb_size_trip_out = model_params['model']['emb_size_trip_out'],
                                emb_size_quad_in = model_params['model']['emb_size_quad_in'],
                                emb_size_quad_out = model_params['model']['emb_size_quad_out'],
                                emb_size_aint_in = model_params['model']['emb_size_aint_in'],
                                emb_size_aint_out = model_params['model']['emb_size_aint_out'],
                                emb_size_rbf = model_params['model']['emb_size_rbf'],
                                emb_size_cbf = model_params['model']['emb_size_cbf'],
                                emb_size_sbf = model_params['model']['emb_size_sbf'],
                                num_before_skip = model_params['model']['num_before_skip'],
                                num_after_skip = model_params['model']['num_after_skip'],
                                num_concat = model_params['model']['num_concat'],
                                num_atom = model_params['model']['num_atom'],
                                num_output_afteratom = model_params['model']['num_output_afteratom'],
                                cutoff = model_params['model']['cutoff'],
                                max_neighbors = model_params['model']['max_neighbors'],
                                max_neighbors_qint = model_params['model']['max_neighbors_qint'],
                                max_neighbors_aeaint = model_params['model']['max_neighbors_aeaint'],
                                max_neighbors_aint = model_params['model']['max_neighbors_aint'],
                                rbf = model_params['model']['rbf'],
                                envelope = model_params['model']['envelope'],
                                cbf = model_params['model']['cbf'],
                                sbf = model_params['model']['sbf'],
                                extensive = model_params['model']['extensive'],
                                output_init = model_params['model']['output_init'],
                                activation = model_params['model']['activation'],
                                regress_forces = model_params['model']['regress_forces'],
                                direct_forces = model_params['model']['direct_forces'],
                                forces_coupled = model_params['model']['forces_coupled'],
                                otf_graph = model_params['model']['otf_graph'],
                                quad_interaction = model_params['model']['quad_interaction'],
                                atom_edge_interaction = model_params['model']['atom_edge_interaction'],
                                edge_atom_interaction = model_params['model']['edge_atom_interaction'],
                                atom_interaction = model_params['model']['atom_interaction'],
                                num_atom_emb_layers = model_params['model']['num_atom_emb_layers'],
                                num_global_out_layers = model_params['model']['num_global_out_layers'],
                                qint_tags = model_params['model']['qint_tags'],
                                enforce_max_neighbors_strictly = model_params['model']['enforce_max_neighbors_strictly'],
                                scale_file = model_scale,
                                **kwargs,
                               )
        
    elif model_name == 'schnet':
        model = SchNet(dataset=dataset,
                       split_file=split_file,
                       hidden_channels = model_params['model']['hidden_channels'],
                       num_filters = model_params['model']['num_filters'],
                       num_interactions = model_params['model']['num_interactions'],
                       num_gaussians = model_params['model']['num_gaussians'],
                       cutoff = model_params['model']['cutoff'],
                       **kwargs,
                       )

    elif model_name == 'dimenetplusplus':
        model = DimeNetPlusPlus(dataset=dataset,
                                        split_file=split_file,
                                        hidden_channels = model_params['model']['hidden_channels'], 
                                        num_blocks = model_params['model']['num_blocks'], 
                                        out_emb_channels = model_params['model']['out_emb_channels'], 
                                        num_spherical = model_params['model']['num_spherical'],
                                        num_radial = model_params['model']['num_radial'], 
                                        cutoff = model_params['model']['cutoff'], 
                                        num_before_skip = model_params['model']['num_before_skip'], 
                                        num_after_skip = model_params['model']['num_after_skip'], 
                                        num_output_layers = model_params['model']['num_output_layers'], 
                                        **kwargs,
                                       ) 

    else:
        logging.info(f"...pretrained {model_name} not implemented.")
        sys.exit()

    # load pretrained weights
    if model_name == 'gemnet_oc':
        model.load_state_dict({k.replace('module.module','model'):v for k,v in torch.load(model_weights, map_location=torch.device(model.device))['state_dict'].items()})
    else:
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_weights, map_location=torch.device(model.device))['state_dict'].items()})

    # Load and set scaling factors for pretrained OCP models
    if data_scale != None:
        with open(data_scale, 'r') as f:
            norms = yaml.safe_load(f)

        model.set_scaling_factors(norms['dataset'][0]['target_mean'], 
                                 norms['dataset'][0]['target_std'], 
                                 norms['dataset'][0]['grad_target_mean'], 
                                 norms['dataset'][0]['grad_target_std'])

    return model



class OCP(NNP):
    """General class for OCP models: https://github.com/FAIR-Chem/fairchem/tree/main
    NB: Examples here use the Open-Catalyst-Project/ocp/ocpmodels codebase, which is
        included in this directory.

    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
            (default: :obj:`6.0`)
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
            (default: :obj:`True`)
        regress_forces: bool
            Whether to predict forces. 
            (default: :obj:`True`)
        direct_forces: bool
            If True, predict forces based on aggregation of interatomic directions.
            If False, predict forces based on negative gradient of energy potential.
            (default: :obj:`False`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`True`)
        use_pbc: bool
            Whether to use periodic boundary conditions.
            (default: :obj:`True`)
        max_neighbors: int
            Maximum number of neighbors for interatomic connections and embeddings.
        enforce_max_neighbors_strictly: bool
            When subselected edges based on max_neighbors args, arbitrarily
            select amongst degenerate edges to have exactly the correct number.
            (default: :obj:`False`)
        freeze_head: bool
            Freeze the interaction head during training and only update output block 
            weights.
            (default: :obj:`False`)
        **kwargs:
            General training arguments supplied to NNP.
    """
    def __init__(
        self,
        dataset,
        split_file,
        cutoff: float = 6.0,
        extensive: bool = True,
        regress_forces: bool = True,
        direct_forces: bool = False,
        otf_graph: bool = True,
        use_pbc: bool = True,
        max_neighbors: int = 50,
        enforce_max_neighbors_strictly: bool = False,
        freeze_head: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, split_file, **kwargs)

        self.freeze_head = freeze_head
        self.direct_forces = direct_forces
        self.regress_forces = regress_forces
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.extensive = extensive
        self.otf_graph = otf_graph

    @property
    def loss_fn(self):
        if self.train_forces:
            return WeightedLoss(reduction='mean')
        else:
            return L2MAELoss(reduction='mean')

    def set_scaling_factors(self, target_mean=0, target_std=1, grad_mean=0, grad_std=1):
        """Scaling factors for some pretrained models."""
        self.target_mean = target_mean
        self.target_std = target_std
        self.grad_mean = grad_mean
        self.grad_std = grad_std

    def _scale(self, y):
        return (y*self.target_std)+self.target_mean

    def _gradscale(self, e):
        return (e-self.grad_mean)/self.grad_std
        
    def forward(self, data):
        # Add OCP-specific data keys
        data.atomic_numbers = data.z
        data.sid = data.system_id
        data.fid = data.idx

        self.fwdout = self.model(data)
        return self.fwdout["energy"]

    def direct_forces(self):
        if hasattr(self, 'fwdout'):
            return self.fwdout["forces"]
        else:
            return torch.zeros(1)

    def training_step(self, batch, batch_index):
        if self.train_forces:
            batch.pos.requires_grad_(True)
            
        E_out = self.forward(batch)

        if self.train_forces:
            if self.direct_forces:
                F_out = self.direct_forces()
            else:
                F_out = -torch.autograd.grad(self._gradscale(E_out), batch.pos, grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=True)[0]
            E_out = self._scale(E_out)
            E_loss, F_loss = self.loss(batch, [E_out, F_out])
            loss = E_loss + F_loss
        else:
            E_out = self._scale(E_out)
            loss = self.loss(batch.y, E_out)

        # Log losses
        self.log("train_loss", torch.round(loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.y.shape[0])
        if self.train_forces:
            self.log("train_E_loss", torch.round(E_loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.y.shape[0])
            self.log("train_F_loss", torch.round(F_loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.y.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        if self.train_forces:
            torch.set_grad_enabled(True)
            batch.pos.requires_grad = True

        E_out = self.forward(batch) 

        if self.train_forces:
            if self.direct_forces:
                F_out = self.direct_forces()
            else:
                F_out = -torch.autograd.grad(self._gradscale(E_out), batch.pos, grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=True)[0]
            E_out = self._scale(E_out)
            E_loss, F_loss = self.loss(batch, [E_out, F_out])
            loss = E_loss + F_loss
        else:
            E_out = self._scale(E_out)
            loss = self.loss(batch.y, E_out)

        # Log losses
        self.log("val_loss", torch.round(loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.y.shape[0])
        if self.train_forces:
            self.log("val_E_loss", torch.round(E_loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.y.shape[0])
            self.log("val_F_loss", torch.round(F_loss, decimals=6).detach(), on_step=False, on_epoch=True, batch_size=batch.y.shape[0])
            
        return loss
        
    def test_step(self, batch):
        E_out = self.forward(batch)
        E_out = self._scale(E_out.detach())
        return E_out
    
    def predict_step(self, batch):
        if self.return_forces:
            torch.set_grad_enabled(True)
            batch.pos.requires_grad_(True)
            E_out = self.forward(batch)
            if self.direct_forces:
                F_out = self.direct_forces()
            else:
                F_out = -torch.autograd.grad(self._gradscale(E_out), batch.pos, grad_outputs=torch.ones_like(E_out), create_graph=False, retain_graph=False)[0]
            E_out = self._scale(E_out.detach())
            return E_out, F_out
        else:
            E_out = self.forward(batch)
            
            return E_out




class GemNetOC(OCP):
    """GemNet-OC, 
    
    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip_in: int
            (Down-projected) embedding size of the quadruplet edge embeddings
            before the bilinear layer.
        emb_size_trip_out: int
            (Down-projected) embedding size of the quadruplet edge embeddings
            after the bilinear layer.
        emb_size_quad_in: int
            (Down-projected) embedding size of the quadruplet edge embeddings
            before the bilinear layer.
        emb_size_quad_out: int
            (Down-projected) embedding size of the quadruplet edge embeddings
            after the bilinear layer.
        emb_size_aint_in: int
            Embedding size in the atom interaction before the bilinear layer.
        emb_size_aint_out: int
            Embedding size in the atom interaction after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        num_output_afteratom: int
            Number of residual blocks in the output blocks
            after adding the atom embedding.
        num_atom_emb_layers: int
            Number of residual blocks for transforming atom embeddings.
        num_global_out_layers: int
            Number of final residual blocks before the output.
        scale_backprop_forces: bool
            Whether to scale up the energy and then scales down the forces
            to prevent NaNs and infs in backpropagated forces.
        cutoff_qint: float
            Quadruplet interaction cutoff in Angstrom.
            Optional. Uses cutoff per default.
        cutoff_aeaint: float
            Edge-to-atom and atom-to-edge interaction cutoff in Angstrom.
            Optional. Uses cutoff per default.
        cutoff_aint: float
            Atom-to-atom interaction cutoff in Angstrom.
            Optional. Uses maximum of all other cutoffs per default.
        max_neighbors_qint: int
            Maximum number of quadruplet interactions per embedding.
            Optional. Uses max_neighbors per default.
        max_neighbors_aeaint: int
            Maximum number of edge-to-atom and atom-to-edge interactions per embedding.
            Optional. Uses max_neighbors per default.
        max_neighbors_aint: int
            Maximum number of atom-to-atom interactions per atom.
            Optional. Uses maximum of all other neighbors per default.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        rbf_spherical: dict
            Name and hyperparameters of the radial basis function used as part of the
            circular and spherical bases.
            Optional. Uses rbf per default.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the circular basis function.
        sbf: dict
            Name and hyperparameters of the spherical basis function.
        forces_coupled: bool
            If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        quad_interaction: bool
            Whether to use quadruplet interactions (with dihedral angles)
        atom_edge_interaction: bool
            Whether to use atom-to-edge interactions
        edge_atom_interaction: bool
            Whether to use edge-to-atom interactions
        atom_interaction: bool
            Whether to use atom-to-atom interactions
        scale_basis: bool
            Whether to use a scaling layer in the raw basis function for better
            numerical stability.
        qint_tags: list
            Which atom tags to use quadruplet interactions for.
            0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
        scale_file: str
            Path to the json file containing the scaling factors.
        **kwargs:
            Shared model arguments supplied to OCP and general training arguments 
            supplied to NNP.
    """
    
    def __init__(self,
        dataset,
        split_file,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 4,
        emb_size_atom: int = 256,
        emb_size_edge: int = 512,
        emb_size_trip_in: int = 64,
        emb_size_trip_out: int = 64,
        emb_size_quad_in: int = 32,
        emb_size_quad_out: int = 32,
        emb_size_aint_in: int = 64,
        emb_size_aint_out: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_sbf: int = 32,
        num_before_skip: int = 2,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        num_output_afteratom: int = 3,
        num_atom_emb_layers: int = 0,
        num_global_out_layers: int = 2,
        scale_backprop_forces: bool = False,
        cutoff_qint: float = None,
        cutoff_aeaint: float = None,
        cutoff_aint: float = None,
        max_neighbors_qint: int = None,
        max_neighbors_aeaint: int = None,
        max_neighbors_aint: int = None,
        rbf: dict = {"name": "gaussian"},
        rbf_spherical: dict = None,
        envelope: dict = {
            "name": "polynomial",
            "exponent": 5,
        },
        cbf: dict = {"name": "spherical_harmonics"},
        sbf: dict = {"name": "spherical_harmonics"},
        forces_coupled: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "silu",
        quad_interaction: bool = False,
        atom_edge_interaction: bool = False,
        edge_atom_interaction: bool = False,
        atom_interaction: bool = False,
        scale_basis: bool = False,
        qint_tags: list = [0, 1, 2],
        num_elements: int = 83,
        scale_file: str = None,
        **kwargs,
        ):
        super().__init__(dataset, split_file, **kwargs)

        self.model = gemnet_oc.GemNetOC(num_atoms=0,     # not used in model
                                        bond_feat_dim=0, # not used in model
                                        num_targets=1,
                                        num_spherical=num_spherical,
                                        num_radial=num_radial,
                                        num_blocks=num_blocks,
                                        emb_size_atom=emb_size_atom,
                                        emb_size_edge=emb_size_edge,
                                        emb_size_trip_in=emb_size_trip_in,
                                        emb_size_trip_out=emb_size_trip_out,
                                        emb_size_quad_in=emb_size_quad_in,
                                        emb_size_quad_out=emb_size_quad_out,
                                        emb_size_aint_in=emb_size_aint_in,
                                        emb_size_aint_out=emb_size_aint_out,
                                        emb_size_rbf=emb_size_rbf,
                                        emb_size_cbf=emb_size_cbf,
                                        emb_size_sbf=emb_size_sbf,
                                        num_before_skip=num_before_skip,
                                        num_after_skip=num_after_skip,
                                        num_concat=num_concat,
                                        num_atom=num_atom,
                                        num_output_afteratom=num_output_afteratom,
                                        num_atom_emb_layers=num_atom_emb_layers,
                                        num_global_out_layers=num_global_out_layers,
                                        regress_forces=self.regress_forces,
                                        direct_forces=self.direct_forces,
                                        use_pbc=self.use_pbc,
                                        scale_backprop_forces=scale_backprop_forces,
                                        cutoff=self.cutoff,
                                        cutoff_qint=cutoff_qint,
                                        cutoff_aeaint=cutoff_aeaint,
                                        cutoff_aint=cutoff_aint,
                                        max_neighbors=self.max_neighbors,
                                        max_neighbors_qint=max_neighbors_qint,
                                        max_neighbors_aeaint=max_neighbors_aeaint,
                                        max_neighbors_aint=max_neighbors_aint,
                                        enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                                        rbf=rbf,
                                        rbf_spherical=rbf_spherical,
                                        envelope=envelope,
                                        cbf=cbf,
                                        sbf=sbf,
                                        extensive=self.extensive,
                                        forces_coupled=forces_coupled,
                                        output_init=output_init,
                                        activation=activation,
                                        quad_interaction=quad_interaction,
                                        atom_edge_interaction=atom_edge_interaction,
                                        edge_atom_interaction=edge_atom_interaction,
                                        atom_interaction=atom_interaction,
                                        scale_basis=scale_basis,
                                        qint_tags=qint_tags,
                                        num_elements=num_elements,
                                        otf_graph=self.otf_graph,
                                        scale_file=scale_file)


        #self.reset_parameters()
        self.set_scaling_factors()
        
        if self.freeze_head:
            self._freeze_head()


    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.atom_emb.requires_grad_(False)
        self.model.edge_emb.requires_grad_(False)
        for interaction in self.model.int_blocks:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')



class GemNetT(OCP):
    """
    GemNet-T, triplets-only variant of GemNet

    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        num_spherical: int
            Controls maximum frequency.
            (default: :obj:`7`)
        num_radial: int
            Controls maximum frequency.
            (default: :obj:`128`)
        num_blocks: int
            Number of building blocks to be stacked.
            (default: :obj:`3`)
        emb_size_atom: int
            Embedding size of the atoms.
            (default: :obj:`512`)
        emb_size_edge: int
            Embedding size of the edges.
            (default: :obj:`512`)
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
            (default: :obj:`64`)
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
            (default: :obj:`16`)
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
            (default: :obj:`16`)
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
            (default: :obj:`64`)
        num_before_skip: int
            Number of residual blocks before the first skip connection.
            (default: :obj:`1`)
        num_after_skip: int
            Number of residual blocks after the first skip connection.
            (default: :obj:`2`)
        num_concat: int
            Number of residual blocks after the concatenation.
            (default: :obj:`1`)
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
            (default: :obj:`3`)
        rbf: dict
            Name and hyperparameters of the radial basis function.
            (default: :obj:`{"name": "gaussian"}`)
        envelope: dict
            Name and hyperparameters of the envelope function.
            (default: :obj:`{"name": "polynomial", "exponent": 5}`)
        cbf: dict
            Name and hyperparameters of the cosine basis function.
            (default: :obj:`{"name": "spherical_harmonics"}`)
        output_init: str
            Initialization method for the final dense layer.
            (default: :obj:`HeOrthogonal`)
        activation: str
            Name of the activation function.
            (default: :obj:`swish`)
        num_elements: int
            Number of elements comprising the atom embedding layer.
            (default: :obj:`83`)
        scale_file: str
            Path to the json file containing the scaling factors.
            (default: :obj:`None`)
        **kwargs:
            Shared model arguments supplied to OCP and general training arguments 
            supplied to NNP.
    """

    def __init__(
        self,
        dataset,
        split_file,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        num_elements: int = 83,
        scale_file: str = None,
        **kwargs,
    ):
        super().__init__(dataset, split_file, **kwargs)

        self.model = gemnet.GemNetT(num_atoms=0,     # not used in model
                                    bond_feat_dim=0, # not used in model
                                    num_targets=1,
                                    num_spherical=num_spherical,
                                    num_radial = num_radial,
                                    num_blocks = num_blocks,
                                    emb_size_atom = emb_size_atom,
                                    emb_size_edge = emb_size_edge,
                                    emb_size_trip = emb_size_trip,
                                    emb_size_rbf = emb_size_rbf,
                                    emb_size_cbf = emb_size_cbf,
                                    emb_size_bil_trip = emb_size_bil_trip,
                                    num_before_skip = num_before_skip,
                                    num_after_skip = num_after_skip,
                                    num_concat = num_concat,
                                    num_atom = num_atom,
                                    regress_forces = self.regress_forces,
                                    direct_forces = self.direct_forces,
                                    cutoff = self.cutoff,
                                    max_neighbors = self.max_neighbors,
                                    rbf = rbf,
                                    envelope = envelope,
                                    cbf = cbf,
                                    extensive = self.extensive,
                                    otf_graph = self.otf_graph,
                                    use_pbc = self.use_pbc,
                                    output_init = output_init,
                                    activation = activation,
                                    num_elements = num_elements,
                                    scale_file = scale_file)

        self.reset_parameters()
        self.set_scaling_factors()
        
        if self.freeze_head:
            self._freeze_head()


    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.radial_basis.requires_grad_(False)
        self.model.cbf_basis3.requires_grad_(False)
        self.model.mlp_rbf3.requires_grad_(False)
        self.model.mlp_cbf3.requires_grad_(False)
        self.model.mlp_rbf_h.requires_grad_(False)
        self.model.mlp_rbf_out.requires_grad_(False)
        self.model.atom_emb.requires_grad_(False)
        self.model.edge_emb.requires_grad_(False)
        for interaction in self.model.int_blocks:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')


class DimeNetPlusPlus(OCP):
    """DimeNet++ from the `"Fast and Uncertainty-Aware Directional Message 
    Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>.

    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        hidden_channels: int 
            Hidden embedding size.
            (default: :obj:`128`)
        out_channels: int 
            Size of each output sample.
            (default: :obj:`1`)
        num_blocks: int 
            Number of building blocks.
            (default: :obj:`4`)
        int_emb_size: int
            Size of embedding in the interaction block.
            (default: :obj:`64`)
        basis_emb_size: int
            Size of basis embedding in the interaction block.
            (default: :obj:`8`)
        out_emb_channels: int
            Size of embedding in the output block.
            (default: :obj:`256`)
        num_spherical: int
            Number of spherical harmonics.
            (default: :obj:`7`)
        num_radial: int
            Number of radial basis functions.
            (default: :obj:`6`)
        envelope_exponent: int
            Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: int
            Number of residual layers in the interaction blocks 
            before the skip connection. 
            (default: :obj:`1`)
        num_after_skip: int
            Number of residual layers in the interaction blocks 
            after the skip connection. 
            (default: :obj:`2`)
        num_output_layers: int
            Number of linear layers for the output blocks. 
            (default: :obj:`3`)
        act: str
            The activation funtion.
            (default: :obj:`"swish"`)
        **kwargs:
            Shared model arguments supplied to OCP and general training arguments 
            supplied to NNP.
        """

    def __init__(self,
                 dataset,
                 split_file,
                 hidden_channels: int = 128, 
                 out_channels: int = 1, 
                 num_blocks: int = 4, 
                 int_emb_size: int = 64, 
                 basis_emb_size: int = 8, 
                 out_emb_channels: int = 256, 
                 num_spherical: int = 7,
                 num_radial: int = 6, 
                 envelope_exponent: int = 5, 
                 num_before_skip: int = 1, 
                 num_after_skip: int = 2, 
                 num_output_layers: int = 3, 
                 act: str = "silu",
                 **kwargs
                ):

        super().__init__(dataset, split_file, **kwargs)

        self.model = DimeNetPlusPlusWrap(num_atoms=0,     # not used in model
                                         bond_feat_dim=0, # not used in model
                                         num_targets=1,
                                         regress_forces = self.regress_forces,
                                         hidden_channels = hidden_channels,
                                         num_blocks = num_blocks,
                                         int_emb_size = int_emb_size,
                                         basis_emb_size = basis_emb_size,
                                         out_emb_channels = out_emb_channels,
                                         num_spherical = num_spherical,
                                         num_radial = num_radial,
                                         otf_graph = self.otf_graph,
                                         cutoff = self.cutoff,
                                         envelope_exponent = envelope_exponent,
                                         num_before_skip = num_before_skip,
                                         num_after_skip = num_after_skip,
                                         num_output_layers = num_output_layers)

        self.reset_parameters()
        self.set_scaling_factors()
        
        if self.freeze_head:
            self._freeze_head()

    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.rbf.requires_grad_(False)
        self.model.sbf.requires_grad_(False)
        self.model.emb.requires_grad_(False)
        for interaction in self.model.interaction_blocks:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')


class SchNet(OCP):
    """Continuous-filter convolutional neural network from "SchNet: A
    Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566> 

    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        hidden_channels: int
            Number of hidden channels.
            (default: :obj:`128`)
        num_filters: int
            Number of filters to use.
            (default: :obj:`128`)
        num_interactions: int
            Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians: int
            The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        readout: str
            Whether to apply :obj:`"add"` or :obj:`"mean"` global aggregation. 
            (default: :obj:`"add"`)
        **kwargs:
            Shared model arguments supplied to OCP and general training arguments 
            supplied to NNP.
    """

    def __init__(self,
                 dataset,
                 split_file,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 readout: str = "add",
                 **kwargs,
                ):
        super().__init__(dataset, split_file, **kwargs)

        self.model = SchNetWrap(num_atoms=0,      # not used in model
                                bond_feat_dim=0,  # not used in model
                                num_targets=1,
                                use_pbc = self.use_pbc,
                                regress_forces = self.regress_forces,
                                otf_graph = self.otf_graph,
                                hidden_channels = hidden_channels,
                                num_filters = num_filters,
                                num_interactions = num_interactions,
                                num_gaussians = num_gaussians,
                                cutoff = self.cutoff,
                                readout = readout)

        self.reset_parameters()
        self.set_scaling_factors()
        
        if self.freeze_head:
            self._freeze_head()

    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.embedding.requires_grad_(False)
        self.model.distance_expansion.requires_grad_(False)
        for interaction in self.model.interactions:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')
