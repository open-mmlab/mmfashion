import os

import numpy as np
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

    def show_topk_retrieved_images(self, retrieved_idxes):
        for idx in retrieved_idxes:
            retrieved_img = self.gallery_idx2im[idx]
            print(os.path.join(self.data_dir, self.img_path, retrieved_img))

    def show_retrieved_images(self, query_feat, gallery_embeds):
        query_dist = []
        for i, feat in enumerate(gallery_embeds):
            cosine_dist = cosine(
                feat.reshape(1, -1), query_feat.reshape(1, -1))
            query_dist.append(cosine_dist)

        query_dist = np.array(query_dist)
        order = np.argsort(query_dist)

        for topk in self.topks:
            print('Retrieved Top%d Results' % topk)
            self.show_topk_retrieved_images(order[:topk])
