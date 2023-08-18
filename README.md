# Package `hlm12erc`

Hi, my name is Hudson Leonardo MENDES!

This python package contains all the software produced for my<br />
CM3070 Final Computer Science Project by the Unviersity of London<br />
for my BSc Computer Science specialised in Machine Learning & Data Science.

This package has a command line interface (or "CLI") able to run<br />
most if not all functionality required to run emotion recognition in<br />
conversations, including (a) **Model Training**, (b) **ERC Inference**<br />
and (c) **Model Evaluation**.

## ML Workflow Recipe

This is a personal workflow that I follow to simplify ML development,
which can often be extremely complex, specially depending on which
ML infrastructure is being used.

1. Setup your package to have the following subpackages: `modelling`, `training` and `serving`;

2. Create your first version of `yourpackage.training.your_dataset_record` trying to make your data as flat as possible and having either (a) `strings` or `tensors` as its attributes;

3. Create your first version of `yourpackage.training.your_dataset_record_reader` to transform and flatten out your data. Very important to unit test your transformations here and ensure data is precisely what you need it to be, including their ranges and scale;

4. Create your first version of `yourpackage.training.your_dataset` to hold the dataframe that will have pointers to your feature sources, but won't load everything in memory;

5. Create your model building blocks, their simples possible implementation, and unit test them. Assert that all transformations and output shapes are exactly what they need to be;

6. Wrap your building blocks int o `your_model` and unit-test it. Ensure that your model is able to choose between your building block implementation using a simple configuration dictionary, call it `my_config`;

7. Write your metric calculation classes and unit test them. Make sure it works with the exact output of your model and the result of your label_encoder; **Important** use scikit-learn here, don't reinvent the wheel;

8. Get your `mlflow`, `aim`, `wandb` setup and build your first `dev/local.ipynb`. Build your mlops pipeline there, as simple as you can, to ensure you _can_ train your model using Jupyter Notebook; **Important**: only train on a `sample.csv` of your data, not your entire training dataset, and ensure your metrics are showing in your monitoring tool correctly;

9. If you are serving your model as an api, time to wrap it up in your REST stack and start trying to serve it. **Important**: test it _as an API_, including load tests. This will give you an initial idea of trade-offs you will need to do in your model to make it work in production;

10. Ship your code to run in your GPU/TPU-accelerated infrastructure and run a full training on your dataset. Your monitoring should work perfectly, and you should have your baseline metrics to start getting your modelling work done.

From this point ownwards, the amount of choices you're gonna have to make is far too large for a recipe, but the above should get you started.

## Offline Tests

```bash
export WANDB_MODE=offline

python -m hlm12erc erc  train --config ./configs/baseline.yml             --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/baseline-a.yml           --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/baseline-t.yml           --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/baseline-v.yml           --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target

python -m hlm12erc erc  train --config ./configs/losses-dice.yml          --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/losses-dice-lr-5e-3.yml  --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/losses-dice-lr-5e-4.yml  --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target

python -m hlm12erc erc  train --config ./configs/losses-focal.yml         --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/losses-focal-lr-5e-3.yml --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
python -m hlm12erc erc  train --config ./configs/losses-focal-lr-5e-4.yml --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target

python -m hlm12erc erc  train --config ./configs/adv-text-gpt2.yml        --train_dataset ./data/sample.csv --valid_dataset ./data/sample.csv --n_epochs 1 --batch_size 4 --out ./target
```

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
