sbatch thesis_code/training/pre_training/slurm_pretrain_hagan_lambda_5.sh
sbatch thesis_code/training/pre_training/slurm_pretrain_alpha_gan.sh
sbatch thesis_code/training/pre_training/slurm_pretrain_kwon_gan.sh
sbatch thesis_code/training/pre_training/slurm_pretrain_vae_64.sh
sbatch thesis_code/training/pre_training/slurm_pretrain_wgan_gp.sh

mv ../data/pre-training/brain-masked-zerosliced ../data/pre-training/brain-masked-zerosliced-old
mv ../data/pre-training/brain-masked-zerosliced-64 ../data/pre-training/brain-masked-zerosliced-64-old
mv ../data/pre-training/brain-masked-zerosliced-new ../data/pre-training/brain-masked-zerosliced
mv ../data/pre-training/brain-masked-zerosliced-64-new ../data/pre-training/brain-masked-zerosliced-64

scancel 3498
scancel 2496
scancel 3492

rm -r lightning* slurm_pretrain*
