# Gradient-based explanations for t-SNE

This repository contains the original Python research implementation of gradient-based explanations for t-SNE. It explores how each input feature locally influences the position of a sample in a two-dimensional embedding, using gradients in a way similar to saliency maps.

The code is an early, basic implementation developed to explore the method and its examples. For the current interactive application, visit **[Insight-SNE](https://insight-sne.netlify.app/)**. For a standalone numerical implementation in C++, see **[tsne-explanations-cpp](https://github.com/sady410/tsne-explanations-cpp)**.

## Related projects

| Project | Purpose |
| --- | --- |
| **This repository** | Original Python research prototype, with scripts and notebooks for exploring gradient-based explanations. |
| **[C++ implementation](https://github.com/sady410/tsne-explanations-cpp)** | Standalone C++17 implementation of exact t-SNE and the explanation method, with command-line numerical workflows. |
| **[Insight-SNE website](https://insight-sne.netlify.app/)** | Current desktop product, screenshots, downloads, and getting-started instructions. |
| **[Insight-SNE desktop repository](https://github.com/sady410/Insight-SNE-desktop)** | Source repository for the current C++/Qt desktop application. |
| **[Original Insight-SNE repository](https://github.com/sady410/Insight-SNE)** | Earlier Qt application using the Python explanation implementation. |

## Explore the method interactively

The current version of Insight-SNE integrates the C++ explanation implementation into a desktop workspace. It lets you import CSV data, compute an embedding, inspect feature importance and directions, compare points and selections, and explore gradient vectors and point scope. You can also save sessions and export data and figures.

**[Visit the website](https://insight-sne.netlify.app/)** · **[Download Insight-SNE](https://insight-sne.netlify.app/#download)** · **[First-session guide](https://insight-sne.netlify.app/#first-session)**

## Research implementation

The Python code and notebooks here preserve the original experimental workflow:

- `custom_tsne.py`: t-SNE implementation used by the research examples.
- `explainer.py`: gradient-based explanation code.
- `countries.ipynb` and `vertebral.ipynb`: example analyses.
- `compute_countries_embedding.py` and `compute_vertebral_embedding.py`: scripts for computing example embeddings.
- `datasets/` and `embeddings/`: data and embedding files used by the examples.
- `requirements.txt`: Python dependencies.

Start with the example notebooks to follow the original analyses. This is experimental research code; check the implementation, assumptions, and outputs for your intended use. Local feature influence should be interpreted alongside the original data and embedding settings.

## Paper and citation

The method is described in:

S. Corbugy, R. Marion, and B. Frénay, “Gradient-based explanation for non-linear non-parametric dimensionality reduction,” *Data Mining and Knowledge Discovery*, 38, 3690–3718, 2024.

**[Read the paper](https://doi.org/10.1007/s10618-024-01055-6)**

If you use the method in research, please cite the paper and identify the implementation you used.

## License

This repository is distributed under the [MIT License](LICENSE). Related repositories and application packages provide their own licenses and third-party notices.
