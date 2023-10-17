# CAVSR: Compression-Aware Video Super-Resolution [CVPR 2023]

[[Paper](http://arxiv.org/abs/2205.09753)] 
[[Poster](.doc/poster.pdf)] 
[[Video](https://youtu.be/5XXwcUcGAqY)]
[[PPT](.doc/PPT.pdf)]

## Introduction

In this paper, we propose a novel and practical compression-aware video super-resolution model, which could adapt its video enhancement process to the estimated compression level.

- A compression encoder is designed to model compression levels of input frames, and a base VSR model is then conditioned on the implicitly computed representation by inserting compression-aware modules. 
- In addition, we propose to further strengthen the VSR model by taking full advantage of meta data that is embedded naturally in compressed video streams in the procedure of information fusion.

## Getting Started

### Installation

  ```bash
  pip install -r requirements.txt
  python setup.py develop
  ```

### Evaluation
1. Copy the dataset and checkpoints to the workplace. 
2. Run scripts:
    ```bash
    python basicsr/test.py   -opt script/test_sota.yml
    ```


## License

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@InProceedings{Wang_2023_CVPR,
    title     = {Compression-Aware Video Super-Resolution},
    author    = {Wang, Yingwei and Isobe, Takashi and Jia, Xu and Tao, Xin and Lu, Huchuan and Tai, Yu-Wing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}
```