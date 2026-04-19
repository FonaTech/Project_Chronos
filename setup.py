from setuptools import setup, find_packages

setup(
    name="project-chronos",
    version="0.1.0",
    description="On-device low-latency lookahead dual-layer MoE inference architecture",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "safetensors>=0.4.0",
        "optuna>=3.6.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "chronos-train=train_chronos:main",
            "chronos-eval=chronos.eval.io_profiler:main",
        ],
    },
)
