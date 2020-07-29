# Model Zoo

More models with different backbones will be added to the model zoo.

## Init models

|   Backbone  |      Download      |
| :---------: | :----------------: |
|    VGG-16   |     [vgg16.pth](https://download.pytorch.org/models/vgg16-397923af.pth)      |


## Attribute Prediction(Coarse)

|   Backbone  |      Pooling     |      Loss     | Top-5 Recall | Top-5 Acc. |      Download (Google)      |      Download (Baidu)      |
| :---------: | :--------------: | :-----------: | :----------: | :--------: | :----------------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |     13.70    |   99.81    |     [model](https://drive.google.com/open?id=1lJlUtEQUxeWCDLj1nIhUBw8QtgaYtuqe)      |     [model](https://pan.baidu.com/s/1QB_10Yx9hU1xqjoZSuZsaQ), passwd: j9qd      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |     14.79    |   99.27    |     [model](https://drive.google.com/open?id=18ZWz9Tr6vsAW5Lxq81ps6GddCJPJ0gMx)      |     -     |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |     23.52    |   99.29    |     [model](https://drive.google.com/open?id=1LmC4aKiOY3qmm9qo6RNDU5v_o-xDCAdT)      |     -     |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |     30.84    |   99.30    |     [model](https://drive.google.com/open?id=1bOL4GhLyBEcXgATiVcZ-g3RD8xhKsj5f)      |     -     |


## Category and Attribute Prediction(Fine)
|   Backbone  |      Pooling     |      Loss     | Top-5 Cate. Recall | Top-5 Attr. Recall. |      Download (Google)      |      Download (Baidu)      |
| :---------: | :--------------: | :-----------: | :----------------: | :-----------------: | :----------------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |        35.91       |        25.44        |     [model](https://drive.google.com/file/d/10SZ3Lw4U0F9OKAuHWc-tBbvLS6yfE_x8/view?usp=sharing)      |     -     |
|    VGG-16   | Landmark Pooling | Cross-Entropy |        37.71       |        26.69        |     [model](https://drive.google.com/file/d/17XlihpZS9iY__i7rPxqlzpenHHRSbLGa/view?usp=sharing)      |     -     |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |        42.87       |        29.37        |     [model](https://drive.google.com/file/d/1zsgxJAkdumpw4uDkapb1Ulq-aG1Hwz45/view?usp=sharing)      |     -     |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |        48.25       |        32.57        |     [model](https://drive.google.com/file/d/1zsgxJAkdumpw4uDkapb1Ulq-aG1Hwz45/view?usp=sharing)      |     -     |


## In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download (Google)      |      Download (Baidu)      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |   38.76    |     [model](https://drive.google.com/open?id=1J3FmP5iVE-arwQZKP2QVrOwDTtTvlzJZ)      |     [model](https://pan.baidu.com/s/1n8BnYBUm4Dug4aREFfYfpQ), passwd: wz69      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |   46.29    |     [model](https://drive.google.com/open?id=1BQxjEqDF4ZQV4X57SiT28qCzIttUAZc-)      |     -     |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |   41.81    |     [model](https://drive.google.com/open?id=1UYaIaDhuCwMQiQIcOEzYlPh0M1RFfdw-)      |     -     |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |   48.82    |     [model](https://drive.google.com/open?id=1HZ13jijnjXxQ4nnsiss-UZ7bxHLN0kjw)      |     -     |


## Consumer-to-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download (Google)      |      Download (Baidu)      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: | :----------------: |
|    VGG-16   | Landmark Pooling | Cross-Entropy |   7.18     |     [model](https://drive.google.com/open?id=1I5_VBDKmjqNtG0-H0e9rvGXhhrz_-lDy)      |     [model](https://pan.baidu.com/s/1WUOihnZzav_8vl6pfHSy1A), passwd: grfx      |


## Fashion Landmark Detection

|   Backbone  |   Loss  | Normalized Error | % of Det. Landmarks |      Download (Google)      |      Download (Baidu)      |
| :---------: | :-----: | :--------------: | :-----------------: | :----------------: | :----------------: |
|    VGG-16   | L2 Loss |       0.0813     |        55.35        |     [model](https://drive.google.com/open?id=1LWhPnkT9AbbldvteFn8u_s21PCQ-h00h)      |     [model](https://pan.baidu.com/s/1tzVGeV5P5Sed3diXEr1UPg), passwd: 4ebx      |
|  ResNet-50  | L2 Loss |       0.0758     |        56.32        |     [model](https://drive.google.com/open?id=1VGbOgkqBOgs2MaZ6qvLplopqqt7vKAM1)      |


## Fashion Compatibility Predictor
|   Backbone  |   Dataset   |  Embedding Projection |                             Loss                            | Fill-in-blank Acc | Compatibility AUC |      Download (Google)      |
| :---------: | :---------: | :-------------------: | :---------------------------------------------------------: | :----------------:| :----------------:| :-------------------------: |
|  ResNet-18  |   Disjoint  | fully-connected layer | Triplet loss, Type-specific loss, Similarity loss, VSE loss |       50.4        |        0.80       | [model](https://drive.google.com/open?id=1T-9BLWbuZhEHpX8xV6f1ZoGUpcjV3efc)   |
|  ResNet-18  |   Disjoint  |     learned metric    | Triplet loss, Type-specific loss, Similarity loss, VSE loss |       55.6        |        0.84       | [model](https://drive.google.com/open?id=1sYF0vqfI2z1riBvF9O643IH6mDikER7n)   |
|  ResNet-18  | Nondisjoint | fully-connected layer | Triplet loss, Type-specific loss, Similarity loss, VSE loss |       53.5        |        0.85       | [model](https://drive.google.com/open?id=177W9T7Szl7Z--mF2Zjv89q6u7nUQG2Lq)   |


## Fashion Segmentation
|   Backbone  |  Model type  |       Dataset       |  bbox detection Average Precision  | segmentation Average Precision |      Download (Google)      |
| :---------: | :----------: | :-----------------: | :--------------------------------: | :----------------------------: | :-------------------------: |
|   Resnet50  |   Mask RCNN  | DeepFashion-In-shop |                0.599               |              0.584             |  [model](https://drive.google.com/open?id=1q6zF7J6Gb-FFgM87oIORIt6uBozaXp5r)   |
