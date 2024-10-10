# Mamba-based Segmentation Model for Speaker Diarization

Alexis Plaquet, Naohiro Tawara, Marc Delcroix, Shota Horiguchi, Atsushi Ando, and Shoko Araki



> Mamba is a newly proposed architecture which behaves like a recurrent neural network (RNN) with attention-like capabilities. These properties are promising for speaker diarization, as attention-based models have unsuitable memory requirements for long-form audio, and traditional RNN capabilities are too limited.
In this paper, we propose to assess the potential of Mamba for diarization by comparing the state-of-the-art neural segmentation of the pyannote.audio pipeline with our proposed Mamba-based variant. Mamba's stronger processing capabilities allow usage of longer local windows, which significantly improve diarization quality by making the speaker embedding extraction more reliable. We find Mamba to be a superior alternative to both traditional RNN and the tested attention-based model. Our proposed Mamba-based system achieves state-of-the-art performance on three widely used diarization datasets.

[📄 Read the paper on arXiv](https://arxiv.org/abs/2410.06459)

## Citations

```bibtex
@misc{plaquet2024mambabasedsegmentationmodelspeaker,
      title={Mamba-based Segmentation Model for Speaker Diarization}, 
      author={Alexis Plaquet and Naohiro Tawara and Marc Delcroix and Shota Horiguchi and Atsushi Ando and Shoko Araki},
      year={2024},
      eprint={2410.06459},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.06459}, 
}
```

## Repository contents

- The source code relevant for running and evaluating Mamba-based models in `src/`
- Tutorial notebooks in [`tutorials/`](tutorials/):
  1. [Setting up the environment](tutorials/0_setup_environment.ipynb)
  2. [Training a Mamba-based segmentation model from scratch](tutorials/1_training_from_scratch.ipynb)
  3. Evaluating the full pipeline (TODO)
- The subset splits used for each dataset in `databases/` (in pyannote.audio format)
- The predictions outputted by the model (in `eval_rttms.zip` files) and the detail of all computed metrics in `.csv` files, all contained in `results/`


## Installation

You can install the `plaqntt` package using pip.

1. Clone this repository and open a terminal in the same folder as this file.
2. Run `pip install -e .`

## License

Please refer to the [LICENSE](LICENCE) file for details.
