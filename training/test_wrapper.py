from models.vanilla_vae import VAE
from wrapper import DPModelWrapper
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from lightning import LightningDataModule
# Function to load and resize the image
def load_and_resize_image(image_path, size=(32, 32)):
    image = Image.open(image_path).convert("L")  # Load image and convert to grayscale (1 channel)
    transform = transforms.Compose([
        transforms.Resize(size),   # Resize to 32x32
        transforms.ToTensor()      # Convert image to tensor
    ])
    image_tensor = transform(image)  # Shape: [1, 32, 32]
    return image_tensor

# Function to duplicate the image tensor along the depth dimension
def duplicate_image_tensor(image_tensor):
    return image_tensor.repeat(32, 1, 1).unsqueeze(0)  # Shape: [1, 32, 32, 32]

# Create a dataset with 1024 examples of the same tensor
def create_dataset(image_tensor, num_samples=124):
    duplicated_image = duplicate_image_tensor(image_tensor)
    dataset = (torch.stack([duplicated_image for _ in range(num_samples)]))  # Stack 1024 copies
    return dataset



class CatDataModule(LightningDataModule):
        def __init__(self,data):
            super().__init__()
            self.train_data = data
            self.val_data = data[:2]
            self.test_data = data[:2]
           

        def train_dataloader(self):
            return DataLoader(self.train_data,batch_size = 16,num_workers=8)

        def val_dataloader(self):
            
            return DataLoader(self.val_data,num_workers=8)

        def test_dataloader(self):
            return DataLoader(self.test_data,num_workers=8 )
# Create a random dataset

def show(model,input):
    print(input.shape)
    image = input[0,0,0,:,:]
    print(image.shape)
    #plt.imshow(image)
    #plt.show()
    recon_image,_,_ = model(input)
    recon_image = recon_image[0,0,0,:,:].detach()
    plt.imshow(recon_image)
    plt.show()

if __name__ == '__main__':
    # Example usage
    image_path = 'data/cat.png'  # Path to your PNG image
    image_tensor = load_and_resize_image(image_path)  # Resize image
    dataset = create_dataset(image_tensor)            # Create dataset
    dataloader = DataLoader(dataset)           # Create DataLoader
    dm = CatDataModule(dataset)

    

    # Wrap the model
    model = VAE()
    wrapped_model = DPModelWrapper(model, enable_dp=True, max_grad_norm=1.0)
    i = next(iter(dm.test_dataloader()))
    #show(wrapped_model,i)

    # Use PyTorch Lightning Trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=1,log_every_n_steps=1)
    trainer.fit(wrapped_model, dm )
    show(wrapped_model,i)
    

