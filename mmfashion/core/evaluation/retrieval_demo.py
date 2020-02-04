import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.distance import cosine as cosine


class ClothesRetriever(object):

    def __init__(self,
                 gallery_im_fn,
                 data_dir,
                 img_path,
                 topks=[5],
                 extract_feature=False):
        self.topks = topks
        self.data_dir = data_dir
        self.img_path = img_path
        self.gallery_idx2im = {}
        gallery_imgs = open(gallery_im_fn).readlines()
        for i, img in enumerate(gallery_imgs):
            self.gallery_idx2im[i] = img.strip('\n')

    def show_topk_retrieved_images(self, retrieved_idx):
        for idx in retrieved_idx:
            retrieved_img = self.gallery_idx2im[idx]
            plt.figure()
            img = cv2.imread(os.path.join(self.img_path, retrieved_img))
            plt.imshow(img)
            plt.show()
    def show_retrieved_images(self, query_embed, gallery_embeds):
        query_dist = []
        for i, feat in enumerate(gallery_embeds):
            cosine_dist = cosine(feat, query_embed.reshape(1, -1))
            query_dist.append(cosine_dist)

        order = np.argsort(query_dist)
        for topk in self.topks:
            print('Retrived Top%d Results' % topk)
            self.show_topk_retrieved_images(order[:topk])
