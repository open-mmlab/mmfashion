|             Config File               |   Structure  |      Pooling     |            Loss function            |
| :-----------------------------------: | :----------: | :--------------: | :---------------------------------: |
|    global_retriever_vgg.py            |     VGG16    |  Global Pooling  | loss_id+triplet_loss+loss_attribute |
|    global_retriever_vgg_loss_id.py    |     VGG16    |  Global Pooling  |               loss_id               |
|global_retriever_vgg_loss_id_triplet.py|     VGG16    |  Global Pooling  |         loss_id+triplet_loss        |
|       roi_retriever_vgg.py            |     VGG16    | Landmark Pooling | loss_id+triplet_loss+loss_attribute |
|    roi_retriever_vgg_loss_id.py       |     VGG16    | Landmark Pooling |               loss_id               |
| roi_retriever_vgg_loss_id_triplet.py  |     VGG16    | Landmark Pooling |         loss_id+triplet_loss        |
|      global_retriever_resnet.py       |    Resnet50  |  Global Pooling  | loss_id+triplet_loss+loss_attribute |
|       roi_retriever_resnet.py         |    Resnet50  | Landmark Pooling | loss_id+triplet_loss+loss_attribute |
|   roi_retriever_resnet_loss_id.py     |    Resnet50  | Landmark Pooling |               loss_id               |
|roi_retriever_resnet_loss_id_triplet.py|    Resnet50  | Landmark Pooling |         loss_id+triplet_loss        |
