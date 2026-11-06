"""Chronos corpus pipeline: download, clean, deduplicate, mix and shard.

The package is task-agnostic.  Translation is one configuration of it; the
same stages handle monolingual pretraining text and chat/SFT records.
Modules import only the standard library, numpy and pyarrow at top level so
that data preparation never pulls in Torch or MLX.
"""
