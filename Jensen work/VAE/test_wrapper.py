from models.vanilla_vae import VAE
from wrapper import DPModelWrapper
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from lightning import LightningDataModule
from models.VQVAE.VQVAE import VQVAE
from argparse import ArgumentParser, Namespace
from pathlib import Path

INPUT_SIZE_H_W = 16
INPUT_SIZE_D   = 16
architecture = VAE
num_samples = 10
epochs = 5

def parse_arguments():
    parser = ArgumentParser()

    #arser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    parser.add_argument('--rescale-input', type=int, nargs='+')
    parser.add_argument("--batch-size", type=int)
    #parser.add_argument("dataset_path", type=Path)

    parser.set_defaults(
        gpus="-1",
        accelerator='ddp',
        benchmark=True,
        num_sanity_val_steps=0,
        precision=16,
        log_every_n_steps=50,
        val_check_interval=0.5,
        flush_logs_every_n_steps=100,
        weights_summary='full',
        max_epochs=int(1e5),
    )
    args = parser.parse_args()
    return args

# Function to load and resize the image
def load_and_resize_image(image_path, size=(INPUT_SIZE_H_W, INPUT_SIZE_H_W)):
    image = Image.open(image_path).convert("L")  # Load image and convert to grayscale (1 channel)
    transform = transforms.Compose([
        transforms.Resize(size),   # Resize to HxW
        transforms.ToTensor()      # Convert image to tensor
    ])
    image_tensor = transform(image)  # Shape: [1, H, W]
    return image_tensor

# Function to duplicate the image tensor along the depth dimension
def duplicate_image_tensor(image_tensor):
    return image_tensor.repeat(INPUT_SIZE_D, 1, 1).unsqueeze(0)  # Shape: [1, D, H, W]

# Create a dataset with 1024 examples of the same tensor
def create_dataset(image_tensor, num_samples=4):
    duplicated_image = duplicate_image_tensor(image_tensor)
    dataset = (torch.stack([duplicated_image for _ in range(num_samples)]))  # Stack num_samples copies
    return dataset


class DataModule(LightningDataModule):
        def __init__(self,data):
            super().__init__()
            self.train_data = data[4:]
            self.val_data = data[2:4]
            self.test_data = data[:2]
            
        def train_dataloader(self):
            return DataLoader(self.train_data,batch_size = 1,num_workers=8)

        def val_dataloader(self):
            
            return DataLoader(self.val_data,num_workers=8)

        def test_dataloader(self):
            return DataLoader(self.test_data,num_workers=8 )
# Create a random dataset

def show(model,input):

    image_slice = input[0,0,0,:,:]
    recon_image = model(input)[0]
    recon_image_slice = recon_image[0,0,0,:,:].detach()
    plt.imshow(image_slice)
    plt.imshow(recon_image_slice)
    plt.show()

if __name__ == '__main__':
    args = parse_arguments() #VQVAE

    image_path = 'data/cat.png'  # Path to your PNG image
    image_tensor = load_and_resize_image(image_path)  # Resize image
    dataset = create_dataset(image_tensor,num_samples=num_samples)            # Create dataset
    dataloader = DataLoader(dataset)           # Create DataLoader
    dm = DataModule(dataset)

    
    
    # Wrap the model
    model = architecture(args)
    wrapped_model = DPModelWrapper(model, enable_dp=True, 
                                   max_grad_norm=5.0,
                                   target_epsilon=100,
                                   target_delta=1e-1,
                                   learning_rate=1e-3)
    i = next(iter(dm.test_dataloader()))
    #show(wrapped_model,i)

    # Use PyTorch Lightning Trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=epochs,log_every_n_steps=5)
    trainer.fit(wrapped_model, dm )
    i = next(iter(dm.test_dataloader()))
    show(wrapped_model,i)
    

