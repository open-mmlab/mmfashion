# mmfashion

## Introduction

mmfashion is an open source visual fashion analysis toolbox based on PyTorch **(ZW: PyTorch1.1 or higher?)**.


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


## Development

1. [Design documentation](https://github.com/open-mmlab/mmfashion/blob/master/doc/design.md)

2. [Issues](https://github.com/open-mmlab/mmfashion/issues)


## Dataset

1. [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

2. **(ZW: Please elaborate how to download and pre-process the dataset.)**


## Model Zoo

### Attribute Prediction

|   Backbone  |      Pooling     |      Loss     |  Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :---------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |             |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |             |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |             |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |             |     [model]()      |

### In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     |  Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :---------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |             |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |             |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |             |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |             |     [model]()      |


## Contributors

* Xin Liu ([veralauee](https://github.com/veralauee))
* Jiancheng Li ([lijiancheng0614](https://github.com/lijiancheng0614))
* Jiaqi Wang ([myownskyW7](https://github.com/myownskyW7))
* Ziwei Liu ([liuziwei7](https://github.com/liuziwei7))
