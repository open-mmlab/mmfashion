# MMFashion

<p align="center">
    <img src='./misc/logo_mmfashion.png' width=320>
</p>


## Introduction

`MMFashion` is an open source visual fashion analysis toolbox based on [PyTorch](https://pytorch.org/). It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Lab, CUHK](http://mmlab.ie.cuhk.edu.hk/).

<p align="left">
    <img src='./misc/demo_attribute.gif' height=220>
    <img src='./misc/demo_retrieval.gif' height=220>
</p>


## Updates
[2019-11-01] `MMFashion` v0.1 is released.


## Features
- **Flexible:** modular design and easy to extend
- **Friendly:** off-the-shelf models for layman users
- **Comprehensive:** support a wide spectrum of fashion analysis tasks

    - [x] Fashion Attribute Prediction
    - [x] Fashion Recognition and Retrieval
    - [x] Fashion Landmark Detection
    - [ ] Fashion Parsing and Segmentation
    - [ ] Fashion Compatibility and Recommendation


## Requirements

- [Python 3.5+](https://www.python.org/)
- [PyTorch 1.0.0+](https://pytorch.org/)
- [mmcv](https://github.com/open-mmlab/mmcv)


## Installation

1. Install from pypi.

    ```bash
    pip install mmfashion
    ```

2. Install from source.

    ```bash
    git clone https://github.com/open-mmlab/mmfashion.git
    python setup.py install
    ```


## Get Started
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of `MMFashion`.


## Data Preparation
Please refer to [DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for the dataset specifics of `MMFashion`.


## Model Zoo
Please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for a comprehensive set of pre-trained models in `MMFashion`.


## Contributing

We appreciate all contributions to improve `MMFashion`. Please refer to [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the contributing guideline.


## License
This project is released under the [Apache 2.0 license](LICENSE).


## Team

* Xin Liu ([veralauee](https://github.com/veralauee))
* Jiancheng Li ([lijiancheng0614](https://github.com/lijiancheng0614))
* Jiaqi Wang ([myownskyW7](https://github.com/myownskyW7))
* Ziwei Liu ([liuziwei7](https://github.com/liuziwei7))
