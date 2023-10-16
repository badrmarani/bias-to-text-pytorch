# Commonalizer

## Installation

We use Poetry to manage the project dependencies, they are specified in the `pyproject.toml` file. To install poetry run :

```bash
pip install poetry
```

To install the environment run : `poetry install`.

### Download datasets and pretrained checkpoints

- Download CelebA

    ```bash
    python scripts/download.py --root="./data/" --type="images/celeba"
    ```

- Download pretrained checkpoints of CelebA

    ```bash
    python scripts/download.py --root="./data/" --type="pretrained_models"
    ```
