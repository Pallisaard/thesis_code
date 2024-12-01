import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lightning import LightningModule
from lightning import LightningDataModule
from opacus import PrivacyEngine
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class DPModelWrapper(LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-2, enable_dp: bool = False, 
                 max_grad_norm: float = 1.0,target_epsilon = 10, 
                 target_delta: float = 1e-5,epochs = 50):
        """
        Args:
            model (nn.Module): The model to be wrapped.
            learning_rate (float): Learning rate for optimizer.
            enable_dp (bool): If True, enables differential privacy training using Opacus.
            max_grad_norm (float): Maximum norm for gradient clipping (used in DP training).
            noise_multiplier (float): The noise multiplier for DP-SGD (used in DP training).
            target_delta (float): Target delta for privacy (used in DP training).
        """
        super(DPModelWrapper, self).__init__()
        self.model = model 
        self.learning_rate = learning_rate
        self.enable_dp = enable_dp
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.epochs = epochs
        self.loss  = model.loss
        
        self.save_hyperparameters(ignore=['model'])
        self.privacy_engine = None


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        try:
            loss = self.loss(x_hat, x)
            self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except:
            raise Exception('Model\'s loss function is invalid')
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        try:
            loss = self.loss(x_hat, x)
            self.log('validation_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except:
            raise Exception('Model\'s loss function is invalid')
        
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        try:
            loss = self.loss(x_hat, x)
            self.log('test_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except:
            raise Exception('Model\'s loss function is invalid')
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
       
        if self.enable_dp:
            self.trainer.fit_loop.setup_data()
            self.privacy_engine = PrivacyEngine()
            self.model, optimizer,dataloader = self.privacy_engine.make_private_with_epsilon(
                    module= self.model,
                    optimizer=optimizer,
                    data_loader=self.trainer.train_dataloader,
                    target_epsilon= self.target_epsilon,
                    target_delta= self.target_delta,
                    epochs=self.epochs,
                    max_grad_norm=self.max_grad_norm
                )
            #self.trainer.train_dataloader = dataloader
        return optimizer

    def on_train_epoch_end(self):
        if self.enable_dp and self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(self.target_delta)
            self.log('epsilon', epsilon,on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log the privacy budget spent (epsilon)

    def on_train_end(self):
        pass

# Example usage
if __name__ == "__main__":
    from models.vanilla_vae import VAE
    from torch.utils.data import DataLoader
    
    class CustomDataModule(LightningDataModule):
        def __init__(self,data):
            super().__init__()
            self.train_data = data
            self.val_data = data
            self.test_data = data
           

        def train_dataloader(self):
            return DataLoader(self.train_data,batch_size = 1, num_workers = 15,persistent_workers=True)

        def val_dataloader(self):
            return DataLoader(self.val_data, num_workers = 15,persistent_workers=True)

        def test_data(self):
            return DataLoader(self.test_data, num_workers = 15,persistent_workers=True)
    # Create a random dataset
    
    if __name__ == '__main__':
        train_dataset = torch.rand([128,1,32,32,32])
        dm = CustomDataModule(train_dataset)
        
        # Wrap the model
        model = VAE()
        wrapped_model = DPModelWrapper(model, enable_dp=True, max_grad_norm=1.0)

        # Use PyTorch Lightning Trainer
        from lightning import Trainer
        trainer = Trainer(max_epochs=5,callbacks=[EarlyStopping(monitor="epsilon", stopping_threshold=)])
        trainer.fit(wrapped_model, dm )
