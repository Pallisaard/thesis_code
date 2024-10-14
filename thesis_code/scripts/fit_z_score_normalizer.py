import argparse

from dataloading.mri_dataset import MRIDataset
from dataloading.transforms import ZScoreNormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Path to directory containing the data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = MRIDataset(data_path=args.data_dir)

    print(f"Dataset: {dataset}")

    print(f"Number of samples: {len(dataset)}")

    zscore_normalizer = ZScoreNormalize().fit(dataset, batch_size=args.batch_size)
    print(f"Mean: {zscore_normalizer.mean}")
    print(f"Standard deviation: {zscore_normalizer.std}")

    zscore_normalizer.save("zscore_normalizer.txt")


if __name__ == "__main__":
    main()
