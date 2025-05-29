# Bias-to-Text

A PyTorch implementation of Bias-to-Text for discovering and mitigating visual biases in classification models through keyword-based explanations.

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

## References

This is a list of models, metrics, and datasets used in this project:

### Models

- Kim, Y., Mo, S., Kim, M., Lee, K., Lee, J., & Shin, J. (2023). Bias-to-text : Debiasing unknown visual biases through language interpretation. arXiv. <https://doi.org/10.48550/arXiv.2301.11104>.
- Yenamandra, S., Ramesh, P., Prabhu, V., & Hoffman, J. (2023). Facts : First amplify correlations and then slice to discover bias. arXiv. <https://doi.org/10.48550/arXiv.2309.17430>

### Datasets

- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. arXiv. <https://doi.org/10.48550/arXiv.1411.7766>
- Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts : On the importance of regularization for worst-case generalization. arXiv. <https://doi.org/10.48550/arXiv.1911.08731>
