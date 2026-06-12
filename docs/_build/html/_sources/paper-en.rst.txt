Technical Report (English)
===========================

The full technical report is available as a PDF:

:download:`Download PDF <../paper.pdf>`

.. raw:: html

   <embed src="../../paper.pdf" width="100%" height="800px" type="application/pdf" />

Abstract
--------

Frankestein Transformer presents a unified configuration-driven toolkit for systematic experimentation with modern transformer architectures, spanning seventeen sequence mixer variants and twenty-three optimizer families. The system supports both encoder-style masked language modeling (MLM) and decoder-style autoregressive (AR) next-token prediction through flexible model class and mode configuration, with specialized fine-tuning workflows for both architectures.

The research contributions are threefold:

1. A strict schema-based configuration contract that enables reproducible experimentation across diverse attention mechanisms
2. A comprehensive optimizer routing framework supporting variance-reduction methods, memory-efficient variants, schedule-free approaches, second-order preconditioners, and low-rank APOLLO-family optimizers
3. End-to-end workflows spanning quantized deployment via ternary weight packing and sentence-embedding training inspired by SBERT

The toolkit implements a web-based configuration interface that provides schema-driven form rendering with inline documentation and real-time validation.
