"""English↔Chinese translation demo built on the Chronos kernel.

Everything translation-specific lives here, outside the ``chronos`` kernel:

* ``corpus``: download, clean, dedup and shard en–zh parallel data
  (WMT19, OPUS-100, FLORES+, …) into the kernel's shard layout;
  ``corpus.docmix`` builds merged multi-sentence ("n-in-1") sets;
* ``run``: ``TranslationLongRun``, the kernel ``LongRun`` with BLEU/chrF
  validation (``chronos-translate run --spec …``);
* ``doc_run``: the doc-mix variant with document-level evaluation;
* ``probe``: offline FLORES+ probe for a run checkpoint;
* ``bundle``: self-contained model bundles (build, check, translate,
  prepare-data, fine-tune, serve);
* ``model_presets``: the 50M en→zh model topology and tokenizer tokens.

The kernel needs none of this to train a general model.
"""

__version__ = "0.3.1a2"
