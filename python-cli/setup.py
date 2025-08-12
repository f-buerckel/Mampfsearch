from setuptools import setup, find_packages

setup(
    name="mampfsearch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "srt",
        "qdrant-client",
        "pysrt",
        "FlagEmbedding",
        "rerankers",
    ],
    entry_points={
        "console_scripts": [
            "mampfsearch=mampfsearch.cli:main",
        ],
    },
    python_requires=">=3.8",
)