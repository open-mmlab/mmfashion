# mmfashion

## Introduction

`mmfashion` is an open source visual fashion analysis toolbox based on PyTorch.


## Requirements

- Python 3.5+
- [PyTorch 0.4.1+](https://pytorch.org/)
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


## Dataset

1. [DeepFashion (Category and Attribute Prediction Benchmark)](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)

2. [DeepFashion (In-Shop Clothes Retrieval Benchmark)](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

**(ZW: Please elaborate how to download and pre-process the dataset.)**

**(Vera: dataset download instruction.)**

To use the DeepFashion dataset you need to first download it to 'data/' , then follow these steps to re-organize the dataset.

```
cd data/
mv Category\ and\ Attribute\ Prediction\ Benchmark Attr_Predict
mv In-shop\ Clothes\ Retrieval\ Benchmark In-shop
python prepare_attr_pred.py
python prepare_in_shop.py

```

The directory should be like this:


```
mmfashion
├── mmfashion
├── tools
├── configs
├── data
│   ├── Attr_Predict
│   │   ├── Anno
│   |   │   ├──train.txt
|   |   |   ├──test.txt
│   |   │   ├──val.txt
│   |   │   ├──train_attr.txt
│   |   │   ├── ...
│   │   ├── Img
│   |   │   ├──img
│   │   ├── Eval
│   |   │   ├── ...
│   ├── In-shop
│   │   ├── Anno
│   |   │   ├──train.txt
|   |   |   ├──query.txt
│   |   │   ├──gallery.txt
│   |   │   ├──train_labels.txt
│   |   │   ├── ...


```


## Model Zoo

### Attribute Prediction

|   Backbone  |      Pooling     |      Loss     | Top-5 Recall. | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :-----------: |:---------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |               |            |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |     22.3      |   99.25    |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |               |            |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |               |            |     [model]()      |

### In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     |  Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :---------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |             |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |             |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |             |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |             |     [model]()      |


## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.


## Contributors

* Xin Liu ([veralauee](https://github.com/veralauee))
* Jiancheng Li ([lijiancheng0614](https://github.com/lijiancheng0614))
* Jiaqi Wang ([myownskyW7](https://github.com/myownskyW7))
* Ziwei Liu ([liuziwei7](https://github.com/liuziwei7))
