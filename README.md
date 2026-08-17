# Gradient-based explanations for t-SNE

This Python implementation explains t-SNE embeddings through local gradients,
showing how each input feature influences the position of a sample in the
two-dimensional embedding.

The method is described in:

S. Corbugy, R. Marion, and B. Frénay, “Gradient-based explanation for
non-linear non-parametric dimensionality reduction,” *Data Mining and Knowledge
Discovery*, 38, 3690–3718, 2024.
[https://doi.org/10.1007/s10618-024-01055-6](https://doi.org/10.1007/s10618-024-01055-6)

A C++ implementation is available at
[github.com/sady410/tsne-explanations-cpp](https://github.com/sady410/tsne-explanations-cpp).
