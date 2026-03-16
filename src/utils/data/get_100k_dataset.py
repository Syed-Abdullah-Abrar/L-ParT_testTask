import os
import tarfile
import requests
import numpy as np
from src.utils.data.dataloader import read_file


def prepare_100k_split():

    url = "https://zenodo.org/record/6619768/files/JetClass_Pythia_val_5M.tar"
    os.makedirs("data/raw", exist_ok=True)

    print("Connecting to Zenodo to stream the tarball...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    extracted_file_path = None

    with tarfile.open(fileobj=response.raw, mode="r|") as tar:
        for member in tar:

            if member.name.endswith(".root"):
                print(f"Found {member.name}! Extracting exactly 100,000 events...")# noqa
                tar.extract(member, path="data/raw")
                extracted_file_path = os.path.join("data/raw", member.name)
                break

    print(f"Successfully extracted: {extracted_file_path}")

    print("Converting ROOT data to Numpy arrays...")
    X_particles, X_jets, y = read_file(extracted_file_path)

    # Slice into the required 80-10-10 splits
    splits = {
        "train_80k": (0, 80000),
        "val_10k": (80000, 90000),
        "test_10k": (90000, 100000)
    }

    for split_name, (start, end) in splits.items():
        out_dir = f"data/{split_name}"
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X_particles.npy"), X_particles[start:end])# noqa
        np.save(os.path.join(out_dir, "X_jets.npy"), X_jets[start:end])
        np.save(os.path.join(out_dir, "y.npy"), y[start:end])
        print(f"Saved {split_name} ({end-start} events) to {out_dir}/")


if __name__ == "__main__":
    prepare_100k_split()
