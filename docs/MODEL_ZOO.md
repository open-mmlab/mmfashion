# Model Zoo

More models with different backbones will be added to the model zoo.

## Attribute Prediction

|   Backbone  |      Pooling     |      Loss     | Top-5 Recall | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |              |            |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |     22.3     |   99.25    |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |              |            |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |              |            |     [model]()      |

## In-Shop Clothes Retrieval

|   Backbone  |      Pooling     |      Loss     | Top-5 Acc. |      Download      |
| :---------: | :--------------: | :-----------: | :--------: | :----------------: |
|    VGG-16   |  Global Pooling  | Cross-Entropy |   38.76    |     [model]()      |
|    VGG-16   | Landmark Pooling | Cross-Entropy |            |     [model]()      |
|  ResNet-50  |  Global Pooling  | Cross-Entropy |            |     [model]()      |
|  ResNet-50  | Landmark Pooling | Cross-Entropy |            |     [model]()      |

## Fashion Landmark Detection

|   Backbone  |   Loss  | Normalized Error | % of Det. Landmarks |      Download      |
| :---------: | :-----: | :--------------: | :-----------------: | :----------------: |
|    VGG-16   | L2 Loss |       0.0813     |        55.38        |     [model]()      |
|  ResNet-50  | L2 Loss |       0.0758     |        56.32        |     [model]()      |
