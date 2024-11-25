python -m thesis_code.training.evaluation.generate_n_sampled_mris --output-dir ../torch-output/generated-examples \
                --n-samples 200 \
                --checkpoint-path ../torch-output/pretraining/lightning/checkpoints/last.ckpt \
                --device 'cuda' \
                --lambdas 5 \
                --batch-size 4 \