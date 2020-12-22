# Getting Started

This page provides basic tutorials about the usage of `MMFashion`.


## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (Category and Attribute Prediction Benchmark, In-Shop Clothes Retrieval Benchmark, Fashion Landmark Detection Benchmark etc.),
and also some high-level apis for easier integration to other projects.

### Test an image

You can use the following commands to test an image.

```sh
python demo/test_*.py --input ${INPUT_IMAGE_FILE}
```

Examples:

Assume that you have already downloaded the checkpoints to `checkpoints/`.

1. Test an attribute predictor(coarse prediction).

    ```sh
    # Prepare `Anno/list_attr_cloth.txt` which is specified in `configs/attribute_predict/global_predictor_vgg_attr.py`
    python demo/test_attr_predictor.py \
        --input demo/imgs/attr_pred_demo1.jpg
    ```

   Test a category and attribute predictor(**more accurate** prediction).

   ```sh
    # Prepare `Anno/list_attr_cloth.txt` which is specified in `configs/category_attribute_predict/global_predictor_vgg_attr.py`
    python demo/test_cate_attr_predictor.py \
        --input demo/imgs/attr_pred_demo1.jpg
   ```

2. Test an in-shop / Consumer-to_shop clothes retriever.

    ```sh
    # Prepare the gallery data which is specified in `configs/retriever_in_shop/global_retriever_vgg_loss_id.py`
    python demo/test_retriever.py \
        --input demo/imgs/retrieve_demo1.jpg
    ```

3. Test a landmark detector.

    ```sh
    python demo/test_landmark_detector.py \
        --input demo/imgs/04_1_front.jpg
    ```

4. Test a fashion-compatibility predictor.

    ```sh
    python demo/test_fashion_recommender.py \
        --input_dir demo/imgs/fashion_compatibility/set2
    ```


### Test a dataset

You can use the following commands to test a dataset.

```sh
python tools/test_*.py --config ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE}
```

Examples:

Assume that you have already downloaded the checkpoints to `checkpoints/` and prepared the dataset in `data/`.

1. Test an attribute predictor.

    ```sh
    python tools/test_predictor.py \
        --config configs/attribute_predict/roi_predictor_vgg_attr.py \
        --checkpoint checkpoint/Predict/vgg/roi/latest.pth
    ```

   Test a category and attribute predictor.

   ```
   python test tools/test_cate_attr_predictor.py \
        --config configs/category_attribute_predict/roi_predictor_vgg.py \
        --checkpoint checkpoint/CateAttrPredict/vgg/roi/latest.pth
   ```

2. Test an in-shop / Consumer-to_shop clothes retriever.

    ```sh
    python tools/test_retriever.py \
        --config configs/retriever_in_shop/roi_retriever_vgg.py \
        --checkpoint checkpoint/Retrieve_in_shop/vgg/latest.pth
    ```

    ```sh
    python tools/test_retriever.py \
        --config configs/retriever_consumer_to_shop/roi_retriever_vgg.py \
        --checkpoint checkpoint/Retrieve_consumer_to_shop/vgg/latest.pth
    ```

3. Test a landmark detector.

    ```sh
    python tools/test_landmark_detector.py \
        --config configs/landmark_detect/landmark_detect_vgg.py
        --checkpoint checkpoint/LandmarkDetect/vgg/latest.pth
    ```

4. Test a fashion-compatibility predictor.

    ```sh
    python tools/test_fashion_recommender.py \
        --config configs/fashion_recommendation/type_aware_recommendation_polyvore_disjoint.py
        --checkpoint checkpoint/FashionRecommend/TypeAware/latest.pth
    ```


6. Test a virtual try-on module.

   Step 1, use the geometric matching module(GMM) to generate warp-cloth and warp-mask,
   ```sh
    python tools/test_virtual_tryon.py \
        --config configs/virtual_tryon/cp_vton.py \
        --stage GMM
    ```

    Step 2, use the tryon module(TOM) to generate the results.
    The default result directory is `data/VTON/result`, you can modify the path in config file(line 103).
   ```sh
    python tools/test_virtual_tryon.py \
        --config configs/virtual_tryon/cp_vton.py \
        --stage TOM
    ```



## Train a model

You can use the following commands to train a model.

```sh
python tools/train_*.py --config ${CONFIG_FILE}
```

Examples:

1. Train an attribute predictor.

    ```sh
    python tools/train_predictor.py \
        --config configs/attribute_predict/roi_predictor_vgg_attr.py
    ```

2. Train an in-shop clothes / Consumer-to-shop retriever.

    ```sh
    python tools/train_retriever.py \
        --config configs/retriever_in_shop/roi_retriever_vgg.py
    ```

    ```sh
    python tools/train_retriever.py \
        --config configs/retriever_consumer_to_shop/roi_retriever_vgg.py
    ```

3. Train a landmark detector.

    ```sh
    python tools/train_landmark_detector.py \
        --config configs/landmark_detect/landmark_detect_vgg.py
    ```

4. Train a fashion-compatibility predictor.

    ```sh
    python tools/train_fashion_recommender.py \
        --config configs/fashion_recommendation/type_aware_recommendation_polyvore_disjoint.py
    ```

5. Train a fashion detector.

    ```sh
    python mmdetection/tools/train.py \
        configs/fashion_parsing_segmentation/mask_rcnn_r50_fpn_1x.py
    ```

6. Train a virtual try-on module.

   Step 1, train a geometric matching module(GMM)
   ```sh
    python tools/train_virtual_tryon.py \
        --config configs/virtual_tryon/cp_vton.py \
        --stage GMM
    ```

    After training GMM, you need to generate the training images which are required by Step 2.

    **Note** that you need to modify the config file `configs/virtual_tryon/cp_vton.py` as follows,

    Line 87 should be `datamode='train'`,

    Line 89 should be `data_list='test_pairs.txt'`,

    Line 93 should be `save_dir=os.path.join(data_root, 'vton_resize', 'train')`

    Then,
   ```sh
    python tools/test_virtual_tryon.py \
        --config configs/virtual_tryon/cp_vton.py \
        --stage GMM
    ```

   Step 2, train a tryon module(TOM),
   ```sh
    python tools/train_virtual_tryon.py \
        --config configs/virtual_tryon/cp_vton.py \
        --stage TOM
    ```



## Use custom datasets

The simplest way is to prepare your dataset to existing dataset formats (AttrDataset, InShopDataset, ConsumerToShopDataset or LandmarkDetectDataset).

Please refer to [DATA_PREPARATION.md](DATA_PREPARATION.md) for the dataset specifics.
