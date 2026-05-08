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

import sys
import logging
import torch
from torch_geometric.nn.models import dimenet

from .base import NNP

class MolecularDimeNet(NNP):
    """DimeNet and DimeNet++ implementations from torch_geometric.
    NB: This model class does NOT account for periodic boundary conditions.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to use for training.
        split_file (str): Path to split file (.npy) defining training and
            validation sets.
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
    """

    def __init__(self,
                 dataset,
                 split_file,
                 model: str = "dimenetpp",
                 hidden_channels: int = 128, 
                 out_channels: int = 1, 
                 num_blocks: int = 4, 
                 int_emb_size: int = 64, #pp only
                 basis_emb_size: int = 8, #pp only
                 out_emb_channels: int = 256,  #pp only
                 num_bilinear: int = 8, #dimenet only
                 num_spherical: int = 7,
                 num_radial: int = 6, 
                 cutoff: float = 5.0, 
                 envelope_exponent: int = 5, 
                 num_before_skip: int = 1, 
                 num_after_skip: int = 2, 
                 num_output_layers: int = 3, 
                 max_neighbors: int = 50,
                 act: str = "swish",
                 ckpt_path: str = None,
                 freeze_head: bool = False,
                 **kwargs
                ):

        super().__init__(dataset, split_file, **kwargs)


        self.variation = model.lower()
        self.ckpt_path = ckpt_path
        self.freeze_head = freeze_head

        if self.variation == 'dimenet':
            self.model = dimenet.DimeNet(hidden_channels=hidden_channels,
                                         out_channels=out_channels,
                                         num_blocks=num_blocks,
                                         num_bilinear = num_bilinear,
                                         num_spherical = num_spherical,
                                         num_radial = num_radial,
                                         cutoff = cutoff,
                                         max_num_neighbors = max_neighbors,
                                         envelope_exponent = envelope_exponent,
                                         num_before_skip = num_before_skip,
                                         num_after_skip = num_after_skip,
                                         num_output_layers = num_output_layers,
                                         act = act,
                                        )
        elif self.variation == 'dimenetpp':
            self.model = dimenet.DimeNetPlusPlus(hidden_channels=hidden_channels,
                                                 out_channels=out_channels,
                                                 num_blocks=num_blocks,
                                                 int_emb_size=int_emb_size,
                                                 basis_emb_size=basis_emb_size,
                                                 num_spherical = num_spherical,
                                                 out_emb_channels = out_emb_channels,
                                                 num_radial = num_radial,
                                                 cutoff = cutoff,
                                                 max_num_neighbors = max_neighbors,
                                                 envelope_exponent = envelope_exponent,
                                                 num_before_skip = num_before_skip,
                                                 num_after_skip = num_after_skip,
                                                 num_output_layers = num_output_layers,
                                                 act = act,
                                                )
        else:
            logging.info(f"...{self.variation} not a valid implemented variation of DimeNet")
            sys.exit()
            
        if self.ckpt_path == None:
            self.reset_parameters()
        else:
            self.load_checkpoint()
            if self.freeze_head:
                self._freeze_head()
                
    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.rbf.requires_grad_(False)
        self.model.emb.requires_grad_(False)
        for interaction in self.model.interaction_blocks:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')
        
    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        self.model.reset_parameters()
        logging.info('...all model weights reset.')

    def forward(self, batch):
        return self.model(batch.z, batch.pos, batch.batch)

    def load_checkpoint(self):
        """Load your own checkpoints after training."""
        self.model.load_state_dict(
            {k.replace('module.',''):v for k,v in torch.load(self.ckpt_path, map_location=torch.device(self.model.device))['state_dict'].items()}
        )
        logging.info(f'...model weights loaded from {self.ckpt_path}')
