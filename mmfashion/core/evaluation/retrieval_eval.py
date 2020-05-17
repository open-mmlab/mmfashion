import numpy as np
from scipy.spatial.distance import cosine as cosine


class Evaluator(object):

    def __init__(self,
                 query_dict_fn,
                 gallery_dict_fn,
                 topks=[3, 5, 10, 20, 30, 50],
                 extract_feature=False):
        """Create the empty array to count

        Args:
            query_dict_fn(dict): the mapping of the index to the id of each
                query_embed.
            gallery_dict_fn(dict): the mapping of the index to the id of each
                gallery_embed.
            tops_type(int): default retrieve top3, top5.
            extract_feature(bool): whether to save extracted garment feature
                or not.
        """

        self.topks = topks
        """ recall@k = ture_positive/k"""
        self.recall = dict()

        for k in topks:
            self.recall[k] = []

        self.query_dict, self.query_id2idx = self.get_id_dict(query_dict_fn)
        self.gallery_dict, self.gallery_id2idx = self.get_id_dict(
            gallery_dict_fn)

        self.extract_feature = extract_feature

    def load_dict(self, fn):
        dic = dict()
        rf = open(fn).readlines()
        for i, line in enumerate(rf):
            dic[i] = int(line.strip('\n'))
        return dic

    def inverse_dict(self, idx2id):
        """ invert "idx2id" dict to "id2idx" dict """
        id2idx = dict()
        for k, v in idx2id.items():  # k:idx v:id
            if v not in id2idx:
                id2idx[v] = [k]
            else:
                id2idx[v].append(k)
        return id2idx

    def single_query(self, query_id, query_feat, gallery_embeds, query_idx):
        query_dist = []
        for j, feat in enumerate(gallery_embeds):
            cosine_dist = cosine(
                feat.reshape(1, -1), query_feat.reshape(1, -1))
            query_dist.append(cosine_dist)
        query_dist = np.array(query_dist)

        order = np.argsort(query_dist)
        single_recall = dict()

        print(self.query_id2idx[query_id])
        for k in self.topks:
            retrieved_idxes = order[:k]
            tp = 0
            relevant_num = len(self.gallery_id2idx[query_id])
            for idx in retrieved_idxes:
                retrieved_id = self.gallery_dict[idx]
                if query_id == retrieved_id:
                    tp += 1

            single_recall[k] = float(tp) / relevant_num
        return single_recall

    def show_results(self):
        print('--------------- Retrieval Evaluation ------------')
        for k in self.topks:
            recall = 100 * float(sum(self.recall[k])) / len(self.recall[k])
            print('Recall@%d = %.2f' % (k, recall))

    def evaluate(self, query_embeds, gallery_embeds):
        for i, query_feat in enumerate(query_embeds):
            query_id = self.query_dict[i]
            single_recall = self.single_query(query_id, query_feat,
                                              gallery_embeds, i)

            for k in self.topks:
                self.recall[k].append(single_recall[k])
            self.show_results()

        self.show_results()

    def show_retrieved_images(self, query_feat, gallery_embeds):
        query_dist = []

        for i, feat in enumerate(gallery_embeds):
            cosine_dist = cosine(
                feat.reshape(1, -1), query_feat.reshape(1, -1))
            query_dist.append(cosine_dist)

        query_dist = np.array(query_dist)
        order = np.argsort(query_dist)

        for k in self.topks:
            retrieved_idxes = order[:k]
            for idx in retrieved_idxes:
                retrieved_id = self.gallery_dict[idx]
                print('retrieved id', retrieved_id)

    def get_id_dict(self, id_file):
        ids = []
        id_fn = open(id_file).readlines()
        id2idx, idx2id = {}, {}
        for idx, line in enumerate(id_fn):
            img_id = int(line.strip('\n'))
            ids.append(img_id)
            idx2id[idx] = img_id

            if img_id not in id2idx:
                id2idx[img_id] = [idx]
            else:
                id2idx[img_id].append(idx)
        return idx2id, id2idx
