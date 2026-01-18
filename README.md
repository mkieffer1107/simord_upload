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