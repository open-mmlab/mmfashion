# Data Preparation

1. [DeepFashion - Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)

2. [DeepFashion - In-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

4. [DeepFashion - Consumer-to-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/Consumer2ShopRetrieval.html)

4. [DeepFashion - Fashion Landmark Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)

To use the DeepFashion dataset you need to first download it to 'data/' , then follow these steps to re-organize the dataset.

```
cd data/
mv Category\ and\ Attribute\ Prediction\ Benchmark Attr_Predict
mv In-shop\ Clothes\ Retrieval\ Benchmark In-shop
mv Fashion\ Landmark\ Detection\ Benchmark/ Landmark_Detect
mv  Consumer-to-shop\ Clothes\ Retrieval\ Benchmark Consumer_to_shop
python prepare_attr_pred.py
python prepare_in_shop.py
python prepare_consumer_to_shop.py
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
