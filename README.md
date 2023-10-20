# Commonalizer

## Installation

We use Poetry to manage the project dependencies, they are specified in the `pyproject.toml` file. To install poetry run :

```bash
pip install poetry
```

To install the environment, run : `poetry install`. Finally, to activate the virtual environment, run the following command: `poetry shell`.

### Download datasets and pretrained checkpoints

- Download CelebA

    ```bash
    python scripts/download.py --root="./data/" --type="images/celeba"
    ```

- Download Waterbirds

    ```bash
    python scripts/download.py --root="./data/" --type="images/waterbirds"
    ```

- Download pretrained checkpoints of CelebA

    ```bash
    python scripts/download.py --root="./data/" --type="pretrained_models"
    ```

## Evaluation

An example notebook demonstrating how B2T works on the CelebA and Waterbirds datasets is provided in [b2t_waterbirds.ipynb](https://colab.research.google.com/gist/badrmarani/cf49ac83016da8bf4f1256dc8ddb6591/b2t_waterbirds.ipynb). Or, after downloading the dataset and pretrained models, you can run the following command

```bash
python ./experiments/b2t.py\
    --classification_model_path="./data/pretrained_models/clf_resnet_dro_waterbirds.pth"\
    --dataset_name="waterbirds"\
    --captioning_model="clipcap"
```
