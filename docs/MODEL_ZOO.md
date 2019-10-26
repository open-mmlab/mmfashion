# Model Zoo

More models with different backbones will be added to the model zoo.

## Attribute Prediction

|   Backbone  |      Pooling     |      Loss     | Top-5 Recall | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |     13.70    |   99.81    |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |     14.79    |   99.27    |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |     23.52    |   99.29    |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |     30.84    |   99.30    |     [model]()      |

## In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |   38.76    |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |   46.29    |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |   37.61    |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |   48.82    |     [model]()      |

## Fashion Landmark Detection

|   Backbone  |   Loss  | Normalized Error | % of Det. Landmarks |      Download      |
| :---------: | :-----: | :--------------: | :-----------------: | :----------------: |
|    VGG-16   | L2 Loss |       0.0813     |        55.38        |     [model]()      |
|  ResNet-50  | L2 Loss |       0.0758     |        56.32        |     [model]()      |
