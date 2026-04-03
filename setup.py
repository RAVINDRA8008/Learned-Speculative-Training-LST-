from setuptools import setup, find_packages

setup(
    name="lst-train",
    version="0.1.0",
    description="Learned Speculative Training: Extending Weight Prediction into Chaotic Training Regimes",
    author="Ravindra",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "datasets",
        "wandb",
        "tqdm",
    ],
)
