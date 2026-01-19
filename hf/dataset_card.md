# SIMORD

HuggingFace re-upload of the [SIMORD dataset](https://huggingface.co/datasets/microsoft/SIMORD), **a medical order extraction benchmark based on doctor-patient conversations**, with corrections to data splits and all text transcripts now included by default. If used, please cite the original authors using the citation below.


## Dataset Details

### Dataset Sources

- **HuggingFace:** https://huggingface.co/datasets/microsoft/SIMORD
- **Paper:** https://arxiv.org/pdf/2507.05517


### Dataset Description

The dataset contains three splits (with their corresponding original SIMORD files):

1) `train` (from `train.json`): examples for in-context learning or fine-tuning.
2) `test1` (from `dev.json`): test set used for the EMNLP 2025 industry track paper.
3) `test2` (from `test.json`): test set for MEDIQA-OE shared task of ClinicalNLP 2025.

With the following distribution

| Split | Original | New | Change |
| :--- | :---: | :---: | :---: |
| `train` | 63 | 81 | +18 |
| `test1` | 100 | 90 | -10 |
| `test2` | 100 | 92 | -8 |
| **TOTAL** | **263** | **263** | **-** |

Note: Both the original SIMORD dataset and this upload use the split name `test1` instead of dev/validation (even though the file is `dev.json`) and `test2` instead of test (even though the file is `test.json`), since both were used as test sets.


### Dataset Changes

The SIMORD dataset is derived from both [ACI-Bench](https://github.com/wyim/aci-bench) and [PriMock57](https://github.com/babylonhealth/primock57).

While PriMock57 doesn't contain any explicit data splits, ACI-Bench contains five splits: `train`, `valid`, `test1`, `test2`, and `test3`. As discussed in an [open HF issue](https://huggingface.co/datasets/microsoft/SIMORD/discussions/2), these splits were not respected when being merged into SIMORD.

For example, SIMORD's `test.json` contains an ACI-Bench train sample:

`
"id": "acibench_D2N036_aci_train"
`

The official SIMORD HF upload contains three data files that are mapped to the following splits

| SIMORD File | Mapped Split | Total | Train | Valid/Dev | Test1 | Test2 | Test3 | PriMock57 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [train.json](https://huggingface.co/datasets/microsoft/SIMORD/blob/main/data/train.json) | `train` | 63 | 15 | 8 | 8 | 10 | 8 | 14 |
| [dev.json](https://huggingface.co/datasets/microsoft/SIMORD/blob/main/data/dev.json) | `test1` | 100 | 27 | 3 | 20 | 14 | 13 | 23 |
| [test.json](https://huggingface.co/datasets/microsoft/SIMORD/blob/main/data/test.json) | `test2` | 100 | 25 | 9 | 11 | 16 | 19 | 20 |



This updated version of SIMORD reallocates samples using the following logic:

- **New `train`** = old train (train+PriMock57 samples) + old test1 (train samples) + old test2 (train samples)
- **New `test1`** = old test1 (non-train samples) + half of old train (non-train, non-PriMock57 samples)
- **New `test2`** = old test2 (non-train samples) + half of old train (non-train, non-PriMock57 samples)

In other words:
- Samples with `_train` suffix are moved to `train`, regardless of which original file they came from
- PriMock57 samples stay in their original splits, since PriMock57 has no explicit data splits
- Non-train samples in the original `test1` and `test2` splits stay where they are
- Non-train, non-PriMock57 samples that were misplaced in the original `train` split are evenly distributed between `test1` and `test2`

After reallocation, the new splits contain the following counts:

| New Split | Total | Train | Valid/Dev | Test1 | Test2 | Test3 | PriMock57 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train` | 81 | 67 | 0 | 0 | 0 | 0 | 14 |
| `test1` | 90 | 0 | 7 | 24 | 19 | 17 | 23 |
| `test2` | 92 | 0 | 13 | 15 | 21 | 23 | 20 |


### Direct Use

```python
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
```


## Citation 

```bibtex
@inproceedings{corbeil-etal-2025-empowering,
    title = "Empowering Healthcare Practitioners with Language Models: Structuring Speech Transcripts in Two Real-World Clinical Applications",
    author = "Corbeil, Jean-Philippe  and
      Ben Abacha, Asma  and
      Michalopoulos, George  and
      Swazinna, Phillip  and
      Del-Agua, Miguel  and
      Tremblay, Jerome  and
      Daniel, Akila Jeeson  and
      Bader, Cari  and
      Cho, Kevin  and
      Krishnan, Pooja  and
      Bodenstab, Nathan  and
      Lin, Thomas  and
      Teng, Wenxuan  and
      Beaulieu, Francois  and
      Vozila, Paul",
    editor = "Potdar, Saloni  and
      Rojas-Barahona, Lina  and
      Montella, Sebastien",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2025",
    address = "Suzhou (China)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-industry.58/",
    doi = "10.18653/v1/2025.emnlp-industry.58",
    pages = "859--870",
    ISBN = "979-8-89176-333-3"
}
```