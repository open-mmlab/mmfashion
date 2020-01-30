# Data Preparation

1. [DeepFashion - Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)

2. [DeepFashion - In-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

4. [DeepFashion - Consumer-to-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/Consumer2ShopRetrieval.html)

4. [DeepFashion - Fashion Landmark Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)

To use the DeepFashion dataset you need to first download it to 'data/' , then follow these steps to re-organize the dataset.

```sh
cd data/
mv Category\ and\ Attribute\ Prediction\ Benchmark Attr_Predict
mv In-shop\ Clothes\ Retrieval\ Benchmark In-shop
mv Consumer-to-shop\ Clothes\ Retrieval\ Benchmark Consumer_to_shop
mv Fashion\ Landmark\ Detection\ Benchmark/ Landmark_Detect
python prepare_attr_pred.py
python prepare_in_shop.py
python prepare_consumer_to_shop.py
python prepare_landmark_detect.py
```

The directory should be like this:

```sh
mmfashion
├── mmfashion
├── tools
├── configs
├── data
│   ├── Attr_Predict
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── val.txt
│   │   ├── train_attr.txt
│   │   ├── ...
│   │   ├── Img
│   │   │   ├──img
│   │   ├── Eval
│   │   │   ├── ...
│   ├── In-shop
│   │   ├── train.txt
│   │   ├── query.txt
│   │   ├── gallery.txt
│   │   ├── train_labels.txt
│   │   │   ├── ...
```

## Prepare an AttrDataset

The file tree should be like this:

```
Attr_Predict
├── Anno
│   ├── list_attr_cloth.txt
│   ├── list_attr_img.txt
│   ├── list_bbox.txt
│   ├── list_category_cloth.txt
│   ├── list_category_img.txt
│   └── list_landmarks.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    ├── XXX.jpg
    └── ...
```

Then run `python prepare_attr_pred.py` to re-organize the dataset.

Please refer to [dataset/ATTR_DATASET.md](dataset/ATTR_DATASET.md) for more info.


## Prepare an InShopDataset

The file tree should be like this:

```
In-shop
├── Anno
│   ├── list_attr_cloth.txt
│   ├── list_attr_items.txt
│   ├── list_bbox_inshop.txt
│   ├── list_description_inshop.json
│   ├── list_item_inshop.txt
│   └── list_landmarks_inshop.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    ├── XXX.jpg
    └── ...
```

Then run `python prepare_in_shop.py` to re-organize the dataset.

Please refer to [dataset/IN_SHOP_DATASET.md](dataset/IN_SHOP_DATASET.md) for more info.


## Prepare a ConsumerToShopDataset

The file tree should be like this:

```
Consumer_to_shop
├── Anno
│   ├── list_attr_cloth.txt
│   ├── list_attr_items.txt
│   ├── list_attr_type.txt
│   ├── list_bbox_consumer2shop.txt
│   ├── list_item_consumer2shop.txt
│   └── list_landmarks_consumer2shop.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    ├── XXX.jpg
    └── ...
```

Then run `python prepare_consumer_to_shop.py` to re-organize the dataset.

Please refer to [dataset/CONSUMER_TO_SHOP_DATASET.md](dataset/CONSUMER_TO_SHOP_DATASET.md) for more info.


## Prepare a LandmarkDetectDataset

The file tree should be like this:

```
Landmark_Detect
├── Anno
│   ├── list_bbox.txt
│   ├── list_joints.txt
│   └── list_landmarks.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    ├── XXX.jpg
    └── ...
```

Then run `python prepare_landmark_detect.py` to re-organize the dataset.

Please refer to [dataset/LANDMARK_DETECT_DATASET.md](dataset/LANDMARK_DETECT_DATASET.md) for more info.
