"""Convert MATLAB simulation files to a GRU-ready NumPy dataset."""

from __future__ import annotations

import argparse

from qubit_surrogate.data import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-folder", required=True)
    parser.add_argument("--out-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, Y = build_dataset(args.mat_folder, args.out_file)

    print(f"Saved {args.out_file}")
    print(f"data_input shape : {X.shape}")
    print(f"data_target shape: {Y.shape}")


if __name__ == "__main__":
    main()
