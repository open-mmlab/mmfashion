# ConsumerToShopDataset

- Annotations (Anno/)

    - Attribute Annotations (list_attr_cloth.txt & list_attr_type.txt & list_attr_items.txt)

        clothing attribute labels. See ATTRIBUTE LABELS section below for more info.

    - Bounding Box Annotations (list_bbox_consumer2shop.txt)

        bounding box labels. See BBOX LABELS section below for more info.

    - Item Annotations (list_item_consumer2shop.txt)

        item labels. See ITEM LABELS section below for more info.

    - Fashion Landmark Annotations (list_landmarks_consumer2shop.txt)

        fashion landmark labels. See LANDMARK LABELS section below for more info.

- Images (Img/)

    consumer-to-shop clothes images. See IMAGE section below for more info.

- Evaluation Partitions (Eval/list_eval_partition.txt)

    image pair names for training, validation and testing set respectively. See EVALUATION PARTITIONS section below for more info.


## IMAGE

*.jpg

format: JPG

Notes:
1. The long side of images are resized to 300;
2. The aspect ratios of original images are kept unchanged.


## BBOX LABELS

list_bbox_consumer2shop.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image name> <clothes type> <source type> <bbox location>
```

Notes:
1. The order of bbox labels accords with the order of entry names;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes;
3. In source type, "1" represents shop image, "2" represents consumer image;
4. In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "x_2" and "y_2" represent the lower right point coordinate of bounding box. Bounding box locations are listed in the order of [x_1, y_1, x_2, y_2].


## LANDMARK LABELS

list_landmarks_consumer2shop.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image name> <clothes type> <variation type> [<landmark visibility 1> <landmark location x_1> <landmark location y_1>, ... <landmark visibility 8> <landmark location x_8> <landmark location y_8>]
```

Notes:
1. The order of landmark labels accords with the order of entry names;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes. Upper-body clothes possess six fahsion landmarks, lower-body clothes possess four fashion landmarks, full-body clothes possess eight fashion landmarks;
3. In variation type, "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in;
4. In landmark visibility state, "0" represents visible, "1" represents invisible/occluded, "2" represents truncated/cut-off;
5. For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]; For lower-body clothes, landmark annotations are listed in the order of ["left waistline", "right waistline", "left hem", "right hem"]; For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].


## ITEM LABELS

list_items_consumer2shop.txt

First Row: number of items

Rest of the Rows: <item id>

Notes:
1. Please refer to the paper "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations" for more details.


## ATTRIBUTE LABELS

list_attr_cloth.txt

```sh
First Row: number of attributes
Second Row: entry names
Rest of the Rows: <attribute name (Chinese)> <attribute name (English)> <attribute type>
```

list_attr_type.txt

```sh
First Row: number of attribute types
Second Row: entry names
Rest of the Rows: <attribute type (Chinese)> <attribute type (English)>
```

list_attr_items.txt

```sh
First Row: number of items
Second Row: entry names
Rest of the Rows: <item id> <attribute labels>
```

Notes:
1. The order of attribute labels accords with the order of attribute names;
2. In attribute labels, "1" represents positive while "-1" represents negative, '0' represents unknown;
3. Attribute prediction is treated as a multi-label tagging problem.


## EVALUATION PARTITIONS

list_eval_partition.txt

```sh
First Row: number of images
Second Row: entry names
Rest of the Rows: <image pair name 1> <image pair name 2> <item id> <evaluation status>
```

Notes:
1. In evaluation status, "train" represents training image, "val" represents validation image, "test" represents testing image;
2. The gallery set here are all the shop images in "val + test" set;
3. Items of clothes images are NOT overlapped within this dataset partition;
4. Please refer to the paper "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations" for more details.
