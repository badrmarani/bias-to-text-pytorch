# Commonalizer

## Installation

We use Poetry to manage the project dependencies, they are specified in the `pyproject.toml` file. To install poetry run :

```bash
pip install poetry
```

To install the environment, run : `poetry install`. Finally, to activate the virtual environment, run the following command : `poetry shell`.

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
