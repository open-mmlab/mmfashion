# Model Zoo

More models with different backbones will be added to the model zoo.

## Attribute Prediction

|   Backbone  |      Pooling     |      Loss     | Top-5 Recall | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |     13.70    |   99.81    |     [model](https://drive.google.com/open?id=1lJlUtEQUxeWCDLj1nIhUBw8QtgaYtuqe)      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |     14.79    |   99.27    |     [model](https://drive.google.com/open?id=18ZWz9Tr6vsAW5Lxq81ps6GddCJPJ0gMx)      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |     23.52    |   99.29    |     [model](https://drive.google.com/open?id=1LmC4aKiOY3qmm9qo6RNDU5v_o-xDCAdT)      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |     30.84    |   99.30    |     [model](https://drive.google.com/open?id=1bOL4GhLyBEcXgATiVcZ-g3RD8xhKsj5f)      |

## In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |   38.76    |     [model](https://drive.google.com/open?id=1J3FmP5iVE-arwQZKP2QVrOwDTtTvlzJZ)      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |   46.29    |     [model](https://drive.google.com/open?id=1BQxjEqDF4ZQV4X57SiT28qCzIttUAZc-)      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |   37.61    |     [model](https://drive.google.com/open?id=1UYaIaDhuCwMQiQIcOEzYlPh0M1RFfdw-)      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |   48.82    |     [model](https://drive.google.com/open?id=1HZ13jijnjXxQ4nnsiss-UZ7bxHLN0kjw)      |

## Consumer-to-Shop Clothes Retrieval
|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: |
|    VGG-16   | Landmark Pooling | Cross-Entropy |   7.18     |     [model](https://drive.google.com/open?id=1I5_VBDKmjqNtG0-H0e9rvGXhhrz_-lDy)      |

## Fashion Landmark Detection

|   Backbone  |   Loss  | Normalized Error | % of Det. Landmarks |      Download      |
| :---------: | :-----: | :--------------: | :-----------------: | :----------------: |
|    VGG-16   | L2 Loss |       0.0813     |        55.35        |     [model](https://drive.google.com/open?id=1LWhPnkT9AbbldvteFn8u_s21PCQ-h00h)      |
|  ResNet-50  | L2 Loss |       0.0758     |        56.32        |     [model](https://drive.google.com/open?id=1VGbOgkqBOgs2MaZ6qvLplopqqt7vKAM1)      |
