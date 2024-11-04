import pickle
import os
import argparse
from numpy.core.fromnumeric import size
from numpy.core.numeric import indices
import torch
import numpy as np
import time
import random

from model_mae import no_diffusion_model_cross_attention_parallel as no_diffusion_model
from data_loader import  traj_dataloader

data_loader = traj_dataloader(city='jinan',data_dir='../data/jinan',batch_size=1)
data = data_loader.generate_batch()
for x,y,z,w in data:
    print(x.shape,y.shape,z.shape,w.shape)
    break