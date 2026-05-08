# It's A SNAP!
## PyTorch Lightning-based NNP Training Wrapper

### Environment
Create a new conda environment as follows:

```
conda create --name wrap python=3.11
conda activate wrap 
export PYTHONUSERBASE=$CONDA_PREFIX
python -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install --user torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install --user lightning
python -m pip install --user torch_geometric
python -m pip install --user torch_ema
python -m pip install --user e3nn 
python -m pip install --user ase pandas h5py prettytable
python -m pip install --user matscipy 
```

Load environment with `conda activate wrap`

Install this repo as follows:

```
python -m pip install git+https://github.com/pnnl/SNAP.git
```

#### Tested Package Versions
* python 3.11
* pytorch 2.5  (cu12-12.4.127)
* torch_scatter 2.1.2
* lightning 2.4.0
* torch_gemetric 2.6.1
* e3nn 0.5.1
* torch-ema 0.3
* numpy 1.25.2

### Data Preprocessing
Structures should be saved in .extxyz format, *including atomic forces*, and placed in the following file structure where `DATADIR` is the top-level directory.

```
$DATADIR 
       |_raw
          |_$SAMPLE
                   |_files.extxyz (or files.xyz)
                   |_statistics.json
```

It is recommended that per-atom E0 values computed at the same level of theory as your data are used to normalize the total energy. These values for all atoms should be saved in the statistics.json file in dictionary format as follows: `{'atomic_energies': {Z_i: E0_i, ...}, 'atomic_numbers': [Z_i, ...]}`. If statistics.json is not present during the preprocessing step, one will be computed for each $SAMPLE folder using the fitting algorithm used in MACE.

See [ASE io](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) for converting simulation output files to .extxyz. Note that MACE-MP-0 expects energies to be in eV and forces to be in eV/Å.

### Model Training
See train-mace-mp-0.py for example training script to finetune MACE-MP-0.

The below example shows how to finetune the 'small' MACE-MP-0 model.
```
srun python train-mace-mp-0.py --savedir {SAVEDIR} --model 'small' \
    --datadir ${DATADIR} --split-file ${DATADIR}/processed/split.npz \
    --batch-size 16 --max-epochs 500 --min-epochs 25 \
    --train-forces
```

#### Training Flags
 | 	Flag	 | 	Description	 | 	Default	 | 	NB	 | 
 | 	:--------	 | 	:--------	 | 	:--------:	 | 	:--------	 | 
 | 	--datadir	 | 	Top-level directory to containing the training set.	 | 		 | 	The format of the data directory must be as follows datadir/raw/$SAMPLE. Training will run over all $SAMPLE directories in raw. Files in $SAMPLE should be in .xyz or .extxyz format. A new directory called datadir/processed will be created to store the processed data in .pt format.	 | 
 | 	--savedir	 | 	Top-level directory to save training results.	 | 		 | 	The directory can but does not have to exist. A subdirectory will be created with the date and time to distinguish training runs with same savedir.	 | 
 | 	--split-file	 | 	Path to file containing train-val-test split in .npz format.	 | 	None	 | 	If no split-file is provided a randomized 80-10-10 split will be used, and the resulting split will be saved in savedir/processed.	 | 
 |  --quantile  |  Train a quantile model. | False | Currently only implemented for optimization of energy quantiles. |
 | 	--total-energy	 | 	Train on total energy instead of normalized energy.	 | 	False	 | 	Normally energy is normalized by subtracting single atom values during preprocessing. This flag will skip that step and train on total energies instead.	 | 
 | 	--train-fraction	 | 	Fraction of the training set to use each epoch.	 | 	1.0	 | 	Each training batch will be a randomized subset of the full training data. A new randomized subset will be used each epoch.	 | 
 | 	--train-forces	 | 	Include forces in the loss functions	 | 	False	 | 		 | 
 | 	--batch-size	 | 	Number of samples per training batch.	 | 	32	 | 		 | 
 | 	--dynamic-batch	 | 	Use dynamic batching based on the number of nodes per sample.	 | 	False	 | 	If applied, progress bar will be disabled. Currently only available for single GPU training.	 | 
 | 	--max-epochs	 | 	Maximum number of training epochs.	 | 	500	 | 		 | 
 | 	--min-epochs	 | 	Minimum number of training epochs.	 | 	1	 | 		 | 
 | 	--max-time	 | 	Maximum amount of time for training.	 | 	None	 | 	Formatted as a string, for example, 00:12:00:00.	 | 
 | 	--lr	 | 	Initial learning rate.	 | 	0.001	 | 		 | 
 | 	--lr-patience	 | 	Number of epochs before decreasing learning rate.	 | 	10	 | 		 | 
 | 	--es-patience	 | 	Number of epochs without improvement in validation set loss before stopping.	 | 	25	 | 		 | 
 | 	--swa	 | 	Apply Stochastic Weight Averaging.	 | 	False	 | 		 | 
 | 	--clip	 | 	Gradient clipping value.	 | 	200	 | 		 | 
 | 	--progress-bar	 | 	Display progress bar during training.	 | 	False	 | 	Not applied if dynamic-batch is called.	 | 
 | 	--seed	 | 	Seed for torch, cuda, numpy.	 | 	42	 | 		 | 
 | 	--full-reproducibility	 | 	Use all deterministic algorithms.	 | 	False	 | 	Will make training on GPUs slightly slower.	 | 
 | 	--amp	 | 	Use Automatic Mixed Precision.	 | 	False	 | 	If train-forces is called, forces will be scaled in the loss.	 | 

#### Additional MACE-MP-0 Flags
 | 	Flag	 | 	Description	 | 	Default	 | 	NB	 | 
 | 	:--------	 | 	:--------	 | 	:--------:	 | 	:--------	 | 
 | 	--model	 | 	Size of the MACE-MP-0 foundation model to use.	 | 	medium	 | 	Choices: small, medium, large. Pretrained model weights will be loaded.	 | 
 | 	--checkpoint	 | 	Path to checkpoint to resume training.	 | 	None	 | 		 | 
 | 	--freeze-head	 | 	Freeze interaction head during training.	 | 	False	 | 	If applied, only weights from the readout layer will be updated during training.	 | 
 | 	--fresh-start	 | 	Re-initialize model weights before training.	 | 	False	 | 	Pretrained weights will be removed and training will begin from scratch.	 | 
 | 	--default-dtype	 | 	Default dtype for model weights.	 | 	float32	 | 		 | 


#### Multi-GPU
There are two parameters in the SLURM submission script that determine how many processes will run your training, the `#SBATCH --nodes=X` setting and `#SBATCH --ntasks-per-node=Y` settings. The numbers there need to match what is configured in your Trainer in the code: `Trainer(num_nodes=X, devices=Y)`. If you change the numbers, update them in BOTH places. 

The example script sets both `num_nodes` and `devices` to be automatically be detected by the Trainer. If using the example script, training over 2 gpus (nproc_per_node) on 1 node (nnodes) can be performed as follows:
```
srun python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train-mace-mp-0.py \
    --datadir ${DATADIR} --split-file ${DATADIR}/processed/split.npz \
    --batch-size 16 --max-epochs 500 --min-epochs 25 \
    --train-forces
```



### References
If you use this code, please cite our associated publication:

```
@article{bilbrey2025uncertainty,
  title={Uncertainty Quantification for Neural Network Potential Foundation Models},
  author={Bilbrey, Jenna A and Firoz, Jesun S and Lee, Mal-Soon and Choudhury, Sutanay},
  journal={npj Computational Materials},
  volume={11},
  number={109},
  year={2025},
  doi={10.1038/s41524-025-01572-y},
  url={https://www.nature.com/articles/s41524-025-01572-y},
}
```

### Acknowledgements
Initial development of this codebase was supported by the "Transferring exascale computational chemistry to cloud computing environment and emerging hardware technologies (TEC4)"  project, which is funded by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, the Division of Chemical Sciences, Geosciences, and Biosciences (under FWP 82037).
