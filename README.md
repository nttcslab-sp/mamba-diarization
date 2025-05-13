# Mamba-based Segmentation Model for Speaker Diarization

Alexis Plaquet, Naohiro Tawara, Marc Delcroix, Shota Horiguchi, Atsushi Ando, and Shoko Araki



> Mamba is a newly proposed architecture which behaves like a recurrent neural network (RNN) with attention-like capabilities. These properties are promising for speaker diarization, as attention-based models have unsuitable memory requirements for long-form audio, and traditional RNN capabilities are too limited.
In this paper, we propose to assess the potential of Mamba for diarization by comparing the state-of-the-art neural segmentation of the pyannote.audio pipeline with our proposed Mamba-based variant. Mamba's stronger processing capabilities allow usage of longer local windows, which significantly improve diarization quality by making the speaker embedding extraction more reliable. We find Mamba to be a superior alternative to both traditional RNN and the tested attention-based model. Our proposed Mamba-based system achieves state-of-the-art performance on three widely used diarization datasets.

- [🌐 Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/10889446)
- [📄 Read the preprint on arXiv](https://arxiv.org/abs/2410.06459)

## Citations

```bibtex
@INPROCEEDINGS{plaquet2025mambabasedsegmentationmodel,
  author={Plaquet, Alexis and Tawara, Naohiro and Delcroix, Marc and Horiguchi, Shota and Ando, Atsushi and Araki, Shoko},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Mamba-based Segmentation Model for Speaker Diarization}, 
  year={2025},
  pages={1-5},
  keywords={Pipelines;Memory management;Bidirectional long short term memory;Signal processing;Acoustics;Reliability;Speech processing;Speaker diarization;end-to-end neural diarization;Mamba;state-space model},
  doi={10.1109/ICASSP49660.2025.10889446}}
```

## Repository contents

- The source code relevant for running and evaluating Mamba-based models in `src/`
- Tutorial notebooks in [`tutorials/`](tutorials/):
  1. [Setting up the environment](tutorials/0_setup_environment.ipynb)
  2. [Training a Mamba-based segmentation model from scratch](tutorials/1_training_from_scratch.ipynb)
- The subset splits used for each dataset in `databases/` (in pyannote.audio format)
- The predictions outputted by the model (in `eval_rttms.zip` files) and the detail of all computed metrics in `.csv` files, all contained in `results/`


## Installation

You can install the `plaqntt` package using pip.

1. Clone this repository and open a terminal in the same folder as this file.
2. Run `pip install -e .`

## License

Please refer to the [LICENSE](LICENCE) file for details.
