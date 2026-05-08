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
from typing import Callable
import torch
from torch_geometric.nn.models import dimenet, schnet

from .base import NNP



class PyG(NNP):
    """General class for PyTorch Geometric models: 
    NB: PyG models typically do not consider periodic boundary conditions.

    Args:
        dataset: Dataset object
            Loaded Dataset object for training and/or inference.
        split_file: str
            Path to .npz file containing the train-val-test split.
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
            (default: :obj:`6.0`)
        max_neighbors: int
            Maximum number of neighbors for interatomic connections and embeddings.
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
        max_neighbors: int = 50,
        freeze_head: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, split_file, **kwargs)

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.freeze_head = freeze_head

    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        self.model.reset_parameters()
        logging.info('...all model weights reset.')

    def forward(self, batch):
        return self.model(batch.z, batch.pos, batch.batch)


class DimeNet(PyG):
    """DimeNet implementation from torch_geometric.
    NB: This model does NOT account for periodic boundary conditions.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to use for training.
        split_file (str): Path to split file (.npy) defining training and
            validation sets.
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
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
        **kwargs:
            Shared model arguments supplied to PyG and general training arguments 
            supplied to NNP.
    """

    def __init__(self,
                 dataset,
                 split_file,
                 hidden_channels: int = 128, 
                 out_channels: int = 1, 
                 num_blocks: int = 4, 
                 num_bilinear: int = 8, 
                 num_spherical: int = 7,
                 num_radial: int = 6, 
                 envelope_exponent: int = 5, 
                 num_before_skip: int = 1, 
                 num_after_skip: int = 2, 
                 num_output_layers: int = 3, 
                 act: str = "swish",
                 **kwargs
                ):

        super().__init__(dataset, split_file, **kwargs)


        self.model = dimenet.DimeNet(hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     num_blocks=num_blocks,
                                     num_bilinear = num_bilinear,
                                     num_spherical = num_spherical,
                                     num_radial = num_radial,
                                     cutoff = self.cutoff,
                                     max_num_neighbors = self.max_neighbors,
                                     envelope_exponent = envelope_exponent,
                                     num_before_skip = num_before_skip,
                                     num_after_skip = num_after_skip,
                                     num_output_layers = num_output_layers,
                                     act = act,
                                    )


            
        self.reset_parameters()

        if self.freeze_head:
            self._freeze_head()
                
    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.rbf.requires_grad_(False)
        self.model.emb.requires_grad_(False)
        for interaction in self.model.interaction_blocks:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')
        




class DimeNetPlusPlus(PyG):
    """DimeNet++ implementation from torch_geometric.
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
        **kwargs:
            Shared model arguments supplied to PyG and general training arguments 
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
                 act: str = "swish",
                 **kwargs
                ):

        super().__init__(dataset, split_file, **kwargs)
        
        self.model = dimenet.DimeNetPlusPlus(hidden_channels=hidden_channels,
                                             out_channels=out_channels,
                                             num_blocks=num_blocks,
                                             int_emb_size=int_emb_size,
                                             basis_emb_size=basis_emb_size,
                                             num_spherical = num_spherical,
                                             out_emb_channels = out_emb_channels,
                                             num_radial = num_radial,
                                             cutoff = self.cutoff,
                                             max_num_neighbors = self.max_neighbors,
                                             envelope_exponent = envelope_exponent,
                                             num_before_skip = num_before_skip,
                                             num_after_skip = num_after_skip,
                                             num_output_layers = num_output_layers,
                                             act = act,
                                            )
        
        self.reset_parameters()
        
        if self.freeze_head:
            self._freeze_head()


    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.rbf.requires_grad_(False)
        self.model.emb.requires_grad_(False)
        for interaction in self.model.interaction_blocks:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')


class SchNet(PyG):
    """SchNet implementation from torch_geometric.
    NB: This model class does NOT account for periodic boundary conditions.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. 
            (default: :obj:`None`)
        atomref (torch.Tensor, optional): 
            The reference of single-atom properties. Expects a vector of 
            shape :obj:`(max_atomic_number, )`.
            (default: :obj:`None`)
        **kwargs:
            Shared model arguments supplied to PyG and general training arguments 
            supplied to NNP.
    """
    def __init__(self,
                 dataset,
                 split_file,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 interaction_graph: Callable = None,
                 readout: str = 'add',
                 dipole: bool = False,
                 mean: float = None,
                 std: float = None,
                 atomref: torch.Tensor = None,
                 freeze_head: bool = False,
                 **kwargs
                ):

        super().__init__(dataset, split_file, **kwargs)

        model = schnet.SchNet(hidden_channels = hidden_channels,
                              num_filters = num_filters,
                              num_interactions = num_interactions,
                              num_gaussians = num_gaussians,
                              interaction_graph = interaction_graph,
                              readout = readout,
                              dipole = dipole,
                              mean = mean,
                              std = std,
                              atomref = atomref,
                             )

        self.reset_parameters()
        
        if self.freeze_head:
            self._freeze_head()

    def _freeze_head(self):
        """Freeze all weights except readout."""
        self.model.embedding.requires_grad_(False)
        self.model.distance_expansion.requires_grad_(False)
        for interaction in self.model.interactions:
            interaction.requires_grad_(False)
        logging.info('...interaction head weights frozen.')


