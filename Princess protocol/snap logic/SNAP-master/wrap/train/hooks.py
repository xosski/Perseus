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
import os
from csv import writer
import json
import time
from datetime import datetime
from lightning.pytorch import seed_everything, callbacks, loggers 

class StartUp():
    """Various startup functions for training.
    """
    def __init__(self, args):
        # create unique identifier
        self.identifier = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")

        # set seeds for numpy, torch and python.random
        seed_everything(args.seed, workers=True)

        # setup functions
        self.create_file_structure(args.savedir)
        self.genereate_logger()
        self.save_args(args)

    def create_file_structure(self, basesavedir):
        # create file structure      
        self.savedir = ''
        for d in [basesavedir, self.identifier]:
            self.savedir = os.path.join(self.savedir, d)
            if not os.path.isdir(self.savedir):
                os.makedirs(self.savedir, exist_ok=True) 

    def genereate_logger(self):
        self.logger = loggers.CSVLogger(self.savedir)

    def save_args(self, args):
        # save arguments to file
        with open(os.path.join(self.savedir, 'args.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)

        

class Logger:
    """Training logger.

    Attributes:
        filename: Full path to log file.
    """
    def __init__(self, filename: str = 'log.csv'):
        self.filename = filename
        self.step = 0

        if os.path.isfile(self.filename):
            # if log already exists, advance step count
            with open(self.filename, "r") as f:
                last_line = f.readlines()[-1]
                if last_line[0]!='e':
                    self.step = int(last_line.split(',')[0])
        else:
            # else, write new log file with header
            with open(self.filename,'w') as f:
                writer_object = writer(f)
                writer_object.writerow(['epoch','time','train loss','val loss'])

    def write_log(self, start, train_loss, val_loss):
        with open(self.filename,'a') as f:
            writer_object = writer(f)
            writer_object.writerow([self.step, round(time.time()-start), train_loss, val_loss])
        
        self.step+=1
            
