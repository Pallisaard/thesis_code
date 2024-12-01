
from wrapper import DPModelWrapper
from models.vanilla_vae import VAE

path = "lightning_logs/version_67/checkpoints/epoch=1-step=16.ckpt"
#model = VAE()
wrapped_model = DPModelWrapper.load_from_checkpoint(path)

#wrapped_model.load_from_checkpoint(path)
print(wrapped_model)