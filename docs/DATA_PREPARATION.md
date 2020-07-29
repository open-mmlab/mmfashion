# Data Preparation

1. [DeepFashion - Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)

2. [DeepFashion - In-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

4. [DeepFashion - Consumer-to-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/Consumer2ShopRetrieval.html)

4. [DeepFashion - Fashion Landmark Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)

5. [Polyvore Outfits](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing)

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
│   │   ├── Anno_coarse
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── val.txt
│   │   │   ├── train_attr.txt
│   │   │   ├── ...
│   │   ├── Anno_fine
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── val.txt
│   │   │   ├── ...
│   │   ├── Img
│   │   │   ├── img
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

**Updated**: We re-labeled a more accurate attribute prediction dataset, please [download](https://drive.google.com/drive/folders/19J-FY5NY7s91SiHpQQBo2ad3xjIB42iN?usp=sharing) them.

The file tree should be like this:

```sh
Attr_Predict
├── Anno_fine
│   ├── list_attr_cloth.txt
│   ├── list_attr_img.txt
│   ├── list_bbox.txt
│   ├── list_category_cloth.txt
│   ├── list_category_img.txt
│   ├── list_landmarks.txt
│   ├── ...
├── Anno_coarse
├── Eval
│   └── list_eval_partition.txt
└── Img
    ├── XXX.jpg
    └── ...
```

Note that if you use "Anno_fine" that contains 26 better-labeled attributes, nothing else need to be done.

If you use run "Anno_coarse" that contains 1000 roughly labeled attributes,
 first run the following script to re-organize them.
 `python prepare_attr_pred.py` 

Please refer to [dataset/ATTR_DATASET.md](dataset/ATTR_DATASET.md) for more info.


## Prepare an InShopDataset
We add segmentation annotations for "Fashion Parsing and Segmentation" task. Please download the updated data.
The file tree should be like this:

```sh
In-shop
├── Anno
│   ├── segmentation
│   |   ├── DeepFashion_segmentation_train.json
│   |   ├── DeepFashion_segmentation_query.json
│   |   ├── DeepFashion_segmentation_gallery.json
│   ├── list_bbox_inshop.txt
│   ├── list_description_inshop.json
│   ├── list_item_inshop.txt
│   └── list_landmarks_inshop.txt
├── Eval
│   └── list_eval_partition.txt
└── Img
    ├── img
    |   ├──XXX.jpg
    ├── img_highres
    └── ├──XXX.jpg

```

Then run `python prepare_in_shop.py` to re-organize the dataset.

Please refer to [dataset/IN_SHOP_DATASET.md](dataset/IN_SHOP_DATASET.md) for more info.


## Prepare a ConsumerToShopDataset

The file tree should be like this:

```sh
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

```sh
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


## Prepare Polyvore-Outfit dataset
Polyvore dataset is widely used for learning fashion compatibility, containing rich multimodel information like
images and descriptions of fashion items, number of likes of the outfit, etc.
It is firstly collected by [Maryland](https://arxiv.org/pdf/1707.05691.pdf).
Here we use a better sorted and grouped version from [UIUC](https://arxiv.org/pdf/1803.09196.pdf).

Download [Polyvore](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing)
and put it in the `data/`

The file tree should be like this:

```sh
Polyvore
├── disjoint
│   ├── compatibility_test.txt
│   ├── compatibility_train.txt
│   ├── fill_in_blank_test.json
|   └── ...
├── nondisjoint
│   └── ...
└── images
│   ├── XXX.jpg
├── categories.csv
├── polyvore_item_metadata.json
├── polyvore_outfit_titles.json
```
