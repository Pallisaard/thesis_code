import torch
import argparse
from os import path
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from models import ConvNext, SWIN
from data import cifar10, isic_2019

print(f"pytorch version: {torch.__version__}")
