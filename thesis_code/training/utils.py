from typing import Literal
import lightning as L
from thesis_code.models.vaes import LitVAE3D
from thesis_code.models.gans.kwon_gan import LitKwonGan


MODEL_NAME = Literal["cicek_3d_vae_64", "cicek_3d_vae_256", "kwon_gan"]


def get_specific_model(
    model_class, checkpoint_path: str | None = None, model_kwargs={}
) -> L.LightningModule:
    if checkpoint_path is not None:
        return model_class.load_from_checkpoint(checkpoint_path)
    return model_class(**model_kwargs)


def get_model(
    model_name: MODEL_NAME, latent_dim: int | None, checkpoint_path: str | None
) -> L.LightningModule:
    if model_name == "cicek_3d_vae_256":
        return get_specific_model(
            LitVAE3D,
            checkpoint_path=checkpoint_path,
            model_kwargs=dict(
                in_shape=(1, 256, 256, 256),
                encoder_out_channels_per_block=[8, 16, 32, 64],
                decoder_out_channels_per_block=[64, 64, 16, 8, 1],
                latent_dim=latent_dim,
                beta_annealing=["monotonic"],
                max_beta=4.0,
                warmup_epochs=25,
            ),
        )
    elif model_name == "cicek_3d_vae_64":
        return get_specific_model(
            LitVAE3D,
            checkpoint_path=checkpoint_path,
            model_kwargs=dict(
                in_shape=(1, 64, 64, 64),
                encoder_out_channels_per_block=[16, 32, 64],
                decoder_out_channels_per_block=[64, 32, 16, 1],
                latent_dim=latent_dim,
            ),
        )
    elif model_name == "kwon_gan":
        return get_specific_model(
            LitKwonGan,
            checkpoint_path=checkpoint_path,
            model_kwargs=dict(
                generator=None,
                critic=None,
                code_critic=None,
                encoder=None,
                lambda_grad_policy=10.0,
                n_critic_steps=5,
                lambda_recon=1.0,
            ),
        )
    else:
        raise ValueError(f"Model name {model_name} not recognized")