# MOLE-CIE
Mixture of LoRA Experts for Continual Information Extraction with LLMs, EMNLP Findings 2025

## Requirements

```
conda env create -f environment.yaml
```

## Data

You can download the ED, RE, NER and replay data from [here](https://anonymous.4open.science/r/MoLE-CIE-data-E1FD/).

Not that we use the ACE and MAVEN datasets for ED evaluation. Due to that the ACE dataset are not released publicly, we can't provide the dataset after processing.

## Preparation

Download the LLaMA3.1-8b model and put it in ./pretrain_model

## Training and Evaluating

First obtain the training script.

```sh
python script/gen_script.py
```

Then run the resulting script to start the training process.

```
sh MoLE-CIE.sh
```

