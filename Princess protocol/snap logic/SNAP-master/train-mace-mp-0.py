import os
import numpy as np
import pandas as pd
from lightning.pytorch import Trainer, callbacks 
import logging 
import torch

from wrap import nn, train
from wrap.nn import macemp
from wrap.data import read_stats
from wrap.data.mace import XYZ4MACEMP
from wrap.calc import evaluate_test_set, evaluate_quantile_test_set


if __name__ == "__main__":
    # Set backend for distributed processes

    # Gather arguments
    parser = macemp.modelargs.get_parser()
    args, override_args = parser.parse_known_args()

    # Set up directory structure and save args
    setup = train.StartUp(args)
    logging.info(f'...model saving to {setup.savedir}')

    # Load data
    dataset = XYZ4MACEMP(root=args.datadir, model=args.model, total_energy=args.total_energy)
    if not os.path.isfile(args.split_file):
        logging.info('...generating split file')
        dataset.generate_split(args.split_file)
    
    # Load model
    if args.quantile:
        model = macemp.QuantileMACE(dataset=dataset, **vars(args))
    else:
        model = macemp.MACEMP(dataset=dataset, **vars(args))
    
    # Set training callbacks
    callbacklist = [callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience, min_delta=0.0001, 
                                            check_finite=True, check_on_train_epoch_end=False),
                    callbacks.ModelCheckpoint(dirpath=setup.savedir, filename='epoch{epoch}', 
                                              auto_insert_metric_name=False, save_last=True),
                   ]

    if args.swa:
        callbacklist.append(callbacks.StochasticWeightAveraging(swa_lrs=args.lr, annealing_epochs=args.lr_patience))

    if args.default_dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    
    torch.set_float32_matmul_precision('high')
    
    # Set up trainer
    trainer = Trainer(max_epochs=args.max_epochs, 
                      min_epochs=args.min_epochs,
                      max_time=args.max_time,
                      logger=setup.logger,
                      inference_mode=False,
                      deterministic=args.full_reproducibility,
                      accelerator="gpu" if torch.cuda.is_available() else "auto",
                      devices=-1,
                      strategy="ddp" if torch.cuda.device_count()>1 else "auto",
                      enable_progress_bar=args.progress_bar, 
                      precision="16-mixed" if args.amp else "32-true",
                      gradient_clip_val=args.clip,
                      reload_dataloaders_every_n_epochs=1 if args.dynamic_batch else 0,
                      limit_train_batches=args.train_fraction,
                      callbacks=callbacklist,
                     )

    # Train model
    trainer.fit(model)

    if args.quantile:
        df = evaluate_quantile_test_set(model)
    else:
        df = evaluate_test_set(model)
    df.to_csv(os.path.join(setup.savedir, 'predictions_on_test_set.csv'))

    # Write LAMMPS model
    if args.for_lammps:
        from wrap.calc import lammps
        lammps.compile_torchscript(os.path.join(setup.savedir,'last.ckpt'), model_size=args.model, default_dtype='float32') 
