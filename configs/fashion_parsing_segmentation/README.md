We develop "fashion parsing and segmentation" module based on "mmdetection"(https://github.com/open-mmlab/mmdetection).

We provide a configuration file that follows Mask-RCNN to build this clothes detection and segmentation tool.
Users can also refer to other model configurations in mmdetection or extend theirs based on the design principle of mmdetection.


## File description
Please create a folder named `mmfashion` and put it into `mmdetection/configs/`, place the config file under this folder.
Also, replace `mmdetection/mmdet/datasets/__init__.py` with our provided `__init__.py`.
Replace `inference.py` in `mmdetection/mmdet/apis/`.
Add the data config `mmfashion.py` into `mmdetection/mmdet/datasets/`.
Add the demo script `demo.py` into `mmdetection/tools/`.

File configuration shown as follows,
```sh
mmfashion
├── mmdetection
│   ├── configs
│   |    ├── mmfashion
│   |    |    └── mask_rcnn_r50_fpn_1x.py
│   ├── mmdet
│   |    ├── datasets
│   |    |    ├── __init__.py
│   |    |    └── mmfashion.py
│   |    ├── apis
│   |    |    ├── inference.py
│   ├── tools
│   |    ├── demo.py
├── data
└── ...
```
