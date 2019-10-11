# MMFashion

<p align="center">
<img src='./misc/logo_mmfashion.png' width=320>
</p>


## Introduction

`MMFashion` is an open source visual fashion analysis toolbox based on [PyTorch](https://pytorch.org/). It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Lab, CUHK](http://mmlab.ie.cuhk.edu.hk/).


## Updates
[2019-10-21] `MMFashion` v0.1 is released.


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


## Dataset

1. [DeepFashion - Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)

2. [DeepFashion - In-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

2. [DeepFashion - Fashion Landmark Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)

To use the DeepFashion dataset you need to first download it to 'data/' , then follow these steps to re-organize the dataset.

```
cd data/
mv Category\ and\ Attribute\ Prediction\ Benchmark Attr_Predict
mv In-shop\ Clothes\ Retrieval\ Benchmark In-shop
 mv Fashion\ Landmark\ Detection\ Benchmark/ Landmark_Detect
python prepare_attr_pred.py
python prepare_in_shop.py
python prepare_landmark_detect.py

```

The directory should be like this:


```
mmfashion
├── mmfashion
├── tools
├── configs
├── data
│   ├── Attr_Predict
│   |   ├──train.txt
|   |   ├──test.txt
│   |   |──val.txt
│   |   |──train_attr.txt
│   |   ├── ...
│   │   ├── Img
│   |   │   ├──img
│   │   ├── Eval
│   |   │   ├── ...
│   ├── In-shop
│   |   ├──train.txt
|   |   ├──query.txt
│   |   ├──gallery.txt
│   |   ├──train_labels.txt
│   |   │   ├── ...


```


## Model Zoo

### Attribute Prediction

|   Backbone  |      Pooling     |      Loss     | Top-5 Recall | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |              |            |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |     22.3     |   99.25    |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |              |            |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |              |            |     [model]()      |

### In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |            |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |            |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |            |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |            |     [model]()      |

### Fashion Landmark Detection

|   Backbone  |   Loss  | Normalized Error | % of Det. Landmarks |      Download      |
| :---------: | :-----: | :--------------: | :-----------------: | :----------------: |
|    VGG-16   | L2 Loss |       0.0845     |                     |     [model]()      |
|  ResNet-50  | L2 Loss |       0.0078     |                     |     [model]()      |


## Contributing

We appreciate all contributions to improve `MMFashion`. Please refer to [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the contributing guideline.


## License
This project is released under the [Apache 2.0 license](LICENSE).


## Team

* Xin Liu ([veralauee](https://github.com/veralauee))
* Jiancheng Li ([lijiancheng0614](https://github.com/lijiancheng0614))
* Jiaqi Wang ([myownskyW7](https://github.com/myownskyW7))
* Ziwei Liu ([liuziwei7](https://github.com/liuziwei7))
