# Package `hlm12erc`

Hi, my name is Hudson Leonardo MENDES!

This python package contains all the software produced for my<br />
CM3070 Final Computer Science Project by the Unviersity of London<br />
for my BSc Computer Science specialised in Machine Learning & Data Science.

This package has a command line interface (or "CLI") able to run<br />
most if not all functionality required to run emotion recognition in<br />
conversations, including (a) **Model Training**, (b) **ERC Inference**<br />
and (c) **Model Evaluation**.

## ETL

```bash
python -m hlm12erc etl kaggle \
    --owner "zaber666" \
    --dataset "meld-dataset" \
    --subdir "MELD-RAW/MELD.Raw" \
    --dest "./data" \
    --force "False"
```

## Training a new ERC Model

```bash
python -m hlm12erc erc train \
    --train_dataset "./data/sample.csv" \
    --valid_dataset "./data/sample.csv" \
    --n_epochs 3 \
    --batch_size 4 \
    --config "./dev/configs/baseline.yml" \
    --out ./target
```

## ERC Model Evaluation

```bash
python -m hlm12erc erc evaluate \
    --test_dataset "./data/test.csv" \
    --out ./target
```

## Running ERC Inference

```bash
python -m hlm12erc erc classify \
    --audio "./utterance.wav" \
    --visual "./scene.wav" \
    --previous_dialog "./dialog_so_far.csv" \
    --utterance "what up?" \
    --out "./target/output/classifications.csv"
```
