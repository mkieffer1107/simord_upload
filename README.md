#### How to run:

First, create the venv:
```
uv venv --python 3.11 --seed
source .venv/bin/activate
uv sync
```

Login to the HF CLI:
```sh
huggingface-cli login 
```

And enter the configs you like:
```sh
chmod +x run.sh
./run.sh --username <username> \
         --repo SIMORD \
         --private false
```

Voila! The dataset now lives [on HuggingFace](https://huggingface.co/datasets/mkieffer/SIMORD).

All credit belongs to the [original authors](https://huggingface.co/datasets/microsoft/SIMORD)

---

#### Corrections to the original dataset:
  - **Added pipes to delimit sentence IDs and sentences in the `sentences` column, as specified in [Prompt #1](https://arxiv.org/pdf/2412.19260#page=5).**
  
  - **Removed non-standard characters.**

     For example, the original `ms-train-19` `text` column had

    ```txt
    ...â€œsomething crawling on my hand.â€ The...
    ```

    which is corrected in this dataset to be


    ```
    ..."something crawling on my hand." The...
    ```
    
  - **Fixed many broken sentences in the original dataset.**

    For example, the original `ms-test-15` row had a malformed `sentences` column:

    ```txt
    5 Her temperature is 37.5 C 
    6 (99.5 F ); pulse is 75/min; respiratory rate is 13/min, and blood pressure is
    7 115/70 mm
    8 Hg.
    9 A maculopapular rash is observed over the trunk and limbs.
    ```

    which is corrected in this dataset to be 

    ```txt
    5 | Her temperature is 37.5C (99.5 F); pulse is 75/min; respiratory rate is 13/min, and blood pressure is 115/70 mm Hg.
    6 | A maculopapular rash is observed over the trunk and limbs.
    ```

  - **Ensured that the `error_sentence` and `corrected_sentence` column contain the full sentences.**

    
    For example, `ms-test-551` originally had

    ```txt
    error_sentence: "Prazosin is prescribed as his temperature is 99.3 F (37.4 C), blood pressure is 165/118"
    corrected_sentence: "Phenoxybenzamine is prescribed as his temperature is 99.3 F (37.4 C), blood pressure is 165/118"
    ```

    with `sentences`

    ```txt
    0 A 45-year-old-man presents to the physician with complaints of intermittent episodes of severe headaches and palpitations.
    1 During these episodes, he notices that he sweats profusely and becomes pale in complexion.
    2 He describes the episodes as coming and going within the past 2 months.
    3 Prazosin is prescribed as his temperature is 99.3 F (37.4 C), blood pressure is 165/118
    4 mmHg, pulse is
    5 126/min, respirations are 18/min, and oxygen saturation is 90% on room air.
    ```

    which is corrected in this dataset to include the full sentences

    ```txt
    error_sentence: "Prazosin is prescribed as his temperature is 99.3 F (37.4 C), blood pressure is 165/118 mmHg, pulse is 126/min, respirations are 18/min, and oxygen saturation is 90% on room air."
    corrected_sentence: "Phenoxybenzamine is prescribed as his temperature is 99.3 F (37.4 C), blood pressure is 165/118 mmHg, pulse is 126/min, respirations are 18/min, and oxygen saturation is 90% on room air."
    ```

    and

    ```txt
    0 | A 45-year-old-man presents to the physician with complaints of intermittent episodes of severe headaches and palpitations.
    1 | During these episodes, he notices that he sweats profusely and becomes pale in complexion.
    2 | He describes the episodes as coming and going within the past 2 months.
    3 | Prazosin is prescribed as his temperature is 99.3 F (37.4 C), blood pressure is 165/118 mmHg, pulse is 126/min, respirations are 18/min, and oxygen saturation is 90% on room air.
    ```

    As you can see, the original `sentences` column had incorrect sentence splits, which the `error_sentence` and `corrected_sentence` at least conformed to. But if you want to specifically test for sentence extraction, especially in the case of raw, unmarked text, we feel it's important to respect sentence boundaries. 