<<<<<<< HEAD
# fashion-attribute
deep fashion attribute prediction system
=======
# fashion-detection
predict attributes of fashion items

1. create a model to save pretrained ImageNet weigths
mkdir saved_models && cd saved_models
download weights from 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
name it as 'vgg16_bn.pth'

cd ..

2. download dataset
cd dataset

first setup images
download DeepFashion dataset from https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk?usp=sharing

then setup labels
download labels from https://drive.google.com/open?id=1EfLX2mGg7cFSqon7gxpgCaofnorZ5ZG2
unzip labels.zip

In such case, we have directory structures like dataset/Img and dataset/labels

3. train
python train_AttrNet.py

4. test
python test_AttrNet.py

5. demo
python demo.py --line_num [line_num]

line_num represents a line in dataset/labels/test.txt, which is the path of an testing image.

>>>>>>> 25f378c83817b3c105cb0fc0bde758b84169eeeb
