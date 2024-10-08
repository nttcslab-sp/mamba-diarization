## Contents

- The source code relevant for running and evaluating Mamba-based models in `src/`
- The subset splits used for each dataset in `databases/` (in pyannote.audio format)
- The predictions outputted by the model (in `eval_rttms.zip` files) and the detail of all computed metrics in `.csv` files, all contained in `results/`
- Tutorial notebooks in [`tutorials/`](tutorials/):
  1. [Setting up the environment](tutorials/0_setup_environment.ipynb)
  2. [Training a Mamba-based segmentation model from scratch](tutorials/1_training_from_scratch.ipynb)
  3. Evaluating the full pipeline (TODO)

## Installation

You can install the `plaqntt` package using pip.

1. Clone this repository and open a terminal in the same folder as this file.
2. Run `pip install -e .`

## License

Please refer to the [LICENSE](LICENSE) file for details.
