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

import torch
import numpy as np
import pandas as pd
import logging
from lightning.pytorch import Trainer

torch.pi = torch.acos(torch.zeros(1)).item() * 2 

def force_magnitude_error(actual, pred):
    # per-atom force magnitude error
    # ||f_hat|| - ||f||
    return torch.sub(torch.norm(pred, dim=1), torch.norm(actual, dim=1))

def force_angular_error(actual, pred):
    # per-atom force angular error
    # cos^-1( f_hat/||f_hat|| • f/||f|| ) / pi
    # batched dot product obtained with torch.bmm(A.view(-1, 1, 3), B.view(-1, 3, 1))
    
    a = torch.norm(actual, dim=1)
    p = torch.norm(pred, dim=1)
    
    return torch.div(torch.acos(torch.bmm(torch.div(actual.T, a).T.view(-1, 1, 3), torch.div(pred.T, p).T.view(-1, 3,1 )).view(-1)), torch.pi)

def evaluate_test_set(model):
    if model.return_forces:
        model.batch_size = 1
    
    trainer = Trainer(inference_mode=not model.return_forces, devices="auto", accelerator='auto', barebones=True)
    output = trainer.predict(model, model.test_dataloader())

    if model.return_forces:
        E_preds = np.concatenate([o[0].cpu() for o in output])   
        F_preds = [o[1].cpu() for o in output]
    else:
        E_preds = np.concatenate([o.cpu().numpy().flatten() for o in output])   
        F_preds = []

    y_idx = [model.dataset[s].idx.item() for s in model.test_idx]
    y_e = [model.dataset[s].energy.item() for s in model.test_idx]
    y_ref = [model.dataset[s].reference_energy.item() for s in model.test_idx]
    n_atoms = [int(model.dataset[s].n_atoms.item()) for s in model.test_idx]
    n_electrons = [sum(model.dataset[s].z).item() for s in model.test_idx]
    true_forces = [model.dataset[s].forces for s in model.test_idx]

    if len(true_forces) == 0 or len(F_preds) == 0:
        logging.info("Force errors not calculated.")
        force_errors = np.array([np.NaN]*len(E_preds))
        force_angular_errors = np.array([np.NaN]*len(E_preds))
    else:
        force_errors = [force_magnitude_error(true_forces[i], F_preds[i]).numpy().mean() for i in range(len(F_preds))]
        force_angular_errors = [force_angular_error(true_forces[i], F_preds[i]).numpy().mean() for i in range(len(F_preds))]
        
    df = pd.DataFrame({'idx': y_idx, 'n_atoms': n_atoms, 'n_electrons': n_electrons,
                         'E_true': y_e, 'E_predicted': E_preds, 'E_reference': y_ref, 
                         'F_magnitude_error': force_errors, 'F_angular_error': force_angular_errors,
                        })

    df['E_MAE_per_atom']=np.abs(df['E_predicted']-df['E_true'])/df['n_atoms']
    df['E_MAE_per_electron']=np.abs(df['E_predicted']-df['E_true'])/df['n_electrons']
    
    return df

def evaluate_quantile_test_set(model):
    model.batch_size = 1
    
    trainer = Trainer(inference_mode=False, devices="auto", accelerator='auto', barebones=True)
    output = trainer.predict(model, model.test_dataloader())

    E_preds = torch.hstack([o[0] for o in output]).squeeze().cpu().numpy()

    y_idx = [model.dataset[s].idx.item() for s in model.test_idx]
    y_e = [model.dataset[s].energy.item() for s in model.test_idx]
    y_ref = [model.dataset[s].reference_energy.item() for s in model.test_idx]
    n_atoms = [int(model.dataset[s].n_atoms.item()) for s in model.test_idx]
    n_electrons = [sum(model.dataset[s].z).item() for s in model.test_idx]

    df = pd.DataFrame({'idx': y_idx, 'n_atoms': n_atoms, 'n_electrons': n_electrons,
                        'E_true': y_e, 'E_predicted_lower': E_preds[0],
                        'E_predicted_upper': E_preds[1], 'E_predicted_mean': E_preds.mean(axis=0),
                        'E_reference': y_ref,
                        })
    
    df['U_predicted']=np.abs(df['E_predicted_upper']-df['E_predicted_lower'])/2
    df['U_per_atom']=df['U_predicted']/df['n_atoms']
    df['U_per_electron']=df['U_predicted']/df['n_electrons']
    df['E_MAE_per_atom']=np.abs(df['E_predicted_mean']-df['E_true'])/df['n_atoms']
    df['E_MAE_per_electron']=np.abs(df['E_predicted_mean']-df['E_true'])/df['n_electrons']
    
    return df