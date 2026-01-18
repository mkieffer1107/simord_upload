import json
from datasets import load_dataset


if __name__ == "__main__":
    # load all data
    dataset = load_dataset("mkieffer/SIMORD")

    # load only train split
    dataset_train = load_dataset("mkieffer/SIMORD", split="train")

    # load only test1 split
    dataset_test1 = load_dataset("mkieffer/SIMORD", split="test1")

    print("\nfull dataset:\n", dataset)
    print("\ntrain split:\n", dataset_train)
    print("\ntest1 split:\n", dataset_test1)

    print("\ntrain sample:\n", json.dumps(dataset_train[0], indent=2))
    print("\ntest1 sample:\n", json.dumps(dataset_test1[0], indent=2))