from setuptools import setup, find_packages

setup(
    name="neuraflux-slm",
    version="0.1.0",
    description="A small language model implementation from scratch",
    author="Saptarshi Halder",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.15.0",
        "datasets>=2.15.0",
        "sentencepiece>=0.1.99",
        "scikit-learn>=1.0.0",
        "tqdm>=4.66.0",
        "evaluate>=0.4.0",
        "accelerate>=0.25.0",
    ],
    python_requires=">=3.8",
) 