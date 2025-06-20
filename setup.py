# flake8: noqa
from setuptools import find_packages, setup

setup(
    name="b2t",
    version="0.1.0",
    author="Badr-Eddine Marani",
    author_email="badr-eddine.marani@outlook.com",
    install_requires=[
        "torch=^2.1.0",
        "torchvision=^0.16.0",
        "transformers=^4.34.0",
        "ipykernel=^6.25.2",
        "ipywidgets=^8.1.1",
        "pandas=^2.1.1",
        "ftfy=^6.1.1",
        "clip=git+https://github.com/openai/CLIP.git",
        "scikit-image=^0.22.0",
        "numpy=^1.26.1",
        "yake=^0.4.8",
    ],
    packages=find_packages(),
)
