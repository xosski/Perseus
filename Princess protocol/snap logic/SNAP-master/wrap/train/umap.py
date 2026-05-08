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
import numpy as np
import umap


def collect_embeddings(model, dataloader, set_tag, savedir, num_layers=-1, invariants_only=True):
    """Save per-atom and per-sample embeddings from trained model.

    Args:
        model (nn model): 
        dataloader (torch dataloader): 
        set_tag (str): 
        savedir (str):
        num_layers (Optional, int): Model layer from which to extract embeddings. [Default = -1]
        invariants_only (Optional, bool): Only relevant to MACE models. [Default = True]
    """
    descriptors=[]
    d_dict={}
    dl=iter(dataloader)

    for i in range(len(dl)):
        batch=next(dl)
        batch.cuda()
        ds = model.get_descriptors(batch, invariants_only=invariants_only, num_layers=num_layers)
        d_dict[str(i)] = ds
        descriptors.append(ds.T.mean(axis=-1))

    descriptors=np.vstack(descriptors)

    np.savez_compressed(os.path.join(savedir, f'{set_tag}_set_per_atom_embeddings.npz'), **d_dict)
    np.savez_compressed(os.path.join(savedir, f'{set_tag}_set_per_sample_embeddings.npz'), descriptors=descriptors)

    return descriptors


def run_umap(model, savedir, 
             num_layers=-1, invariants_only=True, 
             n_neighbors=100, min_dist=0.1, n_components=2, metric='minkowski'):
    """Train UMAP on model training set; collect umap embeddings for train, val, and test sets.

    Args:
        model (nn model): Trained model.
        savedir (str): Path to save output.
        num_layers (Optional, int): Model layer from which to extract embeddings. [Default = -1]
        invariants_only (Optional, bool): Extract invariants only. Only relevant to MACE models. [Default = True]
        n_neighbors (Optional, int): Size of local neighborhood (in terms of number of neighboring sample points) 
                                     used for manifold approximation. 
        min_dist (Optional, float): The effective minimum distance between embedded points. [Default = 0.1]
        n_components (Optional, int): The dimension of the space to embed into. [Default = 2]
        metric (Optional, str): The metric to use to compute distances in high dimensional space. [Default = 'minkowski']
    """
    
    train_descriptors = collect_embeddings(model, model.train_dataloader(), 'train', savedir, num_layers, invariants_only)
    val_descriptors   = collect_embeddings(model, model.val_dataloader(),   'val',   savedir, num_layers, invariants_only)
    test_descriptors  = collect_embeddings(model, model.test_dataloader(),  'test',  savedir, num_layers, invariants_only)

    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        metric=metric
                       )
    
    train_embeddings = reducer.fit_transform(train_descriptors)
    val_embeddings   = reducer.transform(val_descriptors)
    test_embeddings  = reducer.transform(test_descriptors)

    np.savez_compressed(os.path.join(savedir, 'umap_embeddings.npz'), 
                        train_embeddings=train_embeddings, 
                        val_embeddings=val_embeddings,
                        test_embeddings=test_embeddings
                       )
