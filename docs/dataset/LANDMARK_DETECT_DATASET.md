# LandmarkDetectDataset

- Annotations (Anno/)

    - Bounding Box Annotations (list_bbox.txt)

        bounding box labels. See BBOX LABELS section below for more info.

    - Human Joint Annotations (list_joints.txt)

        human joint labels. See JOINT LABELS section below for more info.

    - Fashion Landmark Annotations (list_landmarks.txt)

        fashion landmark labels. See LANDMARK LABELS section below for more info.

- Images (Img/)

    clothes images. See IMAGE section below for more info.

- Evaluation Partitions (Eval/list_eval_partition.txt)

    image names for training, validation and testing set respectively. See EVALUATION PARTITIONS section below for more info.


## IMAGE

*.jpg

format: JPG

Notes:
1. The long side of images are resized to 512;
2. The aspect ratios of original images are kept unchanged.


## BBOX LABELS

list_bbox.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image name> <bbox location>
```

Notes:
1. The order of bbox labels accords with the order of entry names;
2. In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "x_2" and "y_2" represent the lower right point coordinate of bounding box. Bounding box locations are listed in the order of [x_1, y_1, x_2, y_2].


## LANDMARK LABELS

list_landmarks.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image name> <clothes type> <variation type> [<landmark visibility 1> <landmark location x_1> <landmark location y_1>, ... <landmark visibility 8> <landmark location x_8> <landmark location y_8>]
```

Notes:
1. The order of landmark labels accords with the order of entry names;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes. Upper-body clothes possess six fashion landmarks, lower-body clothes possess four fashion landmarks, full-body clothes possess eight fashion landmarks;
3. In variation type, "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in;
4. In landmark visibility state, "0" represents visible, "1" represents invisible/occluded, "2" represents truncated/cut-off;
5. For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]; For lower-body clothes, landmark annotations are listed in the order of ["left waistline", "right waistline", "left hem", "right hem"]; For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].


## JOINT LABELS

list_joints.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image name> <clothes type> <variation type> [<joint visibility 1> <joint location x_1> <joint location y_1>, ... <joint visibility 14> <joint location x_14> <joint location y_14>]
```

Notes:
1. The order of joint labels accords with the order of entry names. Overall there are fourteen human joints;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes;
3. In variation type, "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in;
4. In landmark visibility state, "0" represents visible, "1" represents invisible.


## EVALUATION PARTITIONS

list_eval_partition.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image name> <evaluation status>
```

Notes:
1. In evaluation status, "train" represents training image, "val" represents validation image, "test" represents testing image;
2. Please refer to the paper "Fashion Landmark Detection in the Wild" for more details.
