# Getting Started

This page provides basic tutorials about the usage of `MMFashion`.


## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (Category and Attribute Prediction Benchmark, In-Shop Clothes Retrieval Benchmark, Fashion Landmark Detection Benchmark etc.),
and also some high-level apis for easier integration to other projects.

### Test a dataset

You can use the following commands to test a dataset.

```shell
python tools/test_*.py --config ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE}
```

Examples:

Assume that you have already downloaded the checkpoints to `checkpoints/`.

1. Test a attribute predictor.

```shell
python tools/test_predictor.py \
    --config configs/attribute_predict/roi_predictor_vgg_attr.py \
    --checkpoint checkpoint/Predict/vgg/roi/latest.pth
```

2. Test a in-shop clothes retriever.

```shell
python tools/test_retriever.py \
    --config configs/retriever/roi_retriever_vgg.py \
    --checkpoint checkpoint/Retrieve/vgg/latest.pth
```

3. Test a landmark detector.

```shell
python tools/test_landmark_detector.py \
    --config configs/landmark_detect/landmark_detect_vgg.py
    --checkpoint checkpoint/LandmarkDetect/vgg/latest.pth
```


## Train a model

You can use the following commands to train a model.

```shell
python tools/train_*.py --config ${CONFIG_FILE}
```

Examples:
1. Train a attribute predictor.

```shell
python tools/train_predictor.py \
    --config configs/attribute_predict/roi_predictor_vgg_attr.py
```

2. Train a in-shop clothes retriever.

```shell
python tools/train_retriever.py \
    --config configs/retriever/roi_retriever_vgg.py
```

3. Train a landmark detector.

```shell
python tools/train_landmark_detector.py \
    --config configs/landmark_detect/landmark_detect_vgg.py
```
