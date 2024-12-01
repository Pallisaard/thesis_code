# Thesis Code

## Introduction

This repository contains the code for my thesis project, in which I seek to apply differential privacy to the methods of training medical data generation models.

## Building the environment

The project can be build using `uv` by running the following command in the root directory:

```bash
uv sync
```

It will install PyTorch with the correct device. Due to wheels of PyTorch only being provided up to `torch>=2.2.0`. the project is fixed to that version. The project is build with python 3.12.

## Pre-training models

To run pre-training, run either of the slurm scripts with

```bash
sbatch thesis_code/training/pre_training/slurm_pretrain_hagan_lambda_{1,5}.sh
```

This assumes that at least 1 A100 is available on the slurm cluster. If this is not the case or you want to run the training on a custom setup, check out the script directly.

## Folder Structure

inside the `thesis_code` folder, the following subfolders are present:

### `thesis_code/dataloading`

Contains scripts and modules for loading and preprocessing MRI datasets.

### `thesis_code/fastsurfer`

Includes scripts for running FastSurfer, a tool for processing MRI scans.

### `thesis_code/metrics`

Contains modules for calculating various metrics used in model evaluation.

### `thesis_code/models`

Includes the implementation of different deep learning models used in the project.

### `thesis_code/scripts`

Contains various utility scripts for data processing and other tasks.

### `thesis_code/training`

Includes scripts and modules for training and fine-tuning the models

## `thesis_code/evaluation`

Includes scripts and notebooks for evaluating the models.

## Getting Started

To get started with this project, clone the repository and follow the instructions in the respective folders for setting up the environment and running the scripts.
