import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist as cdist


class Evaluator(object):

    def __init__(self, query_dict_fn, gallery_dict_fn, topks=[3, 5]):
        """ create the empty array to count
        Args:
        query_dict_fn : the index of query_embeds to id of query_embeds
        tops_type : default retrieve top3, top5, top10, top20
        """

        self.topks = topks
        """ recall@k = ture_positive/k"""
        self.recall = dict()

        for k in topks:
            self.recall[k] = []

        self.query_dict = self.load_dict(query_dict_fn)
        self.gallery_dict = self.load_dict(gallery_dict_fn)

        self.query_id2idx = self.inverse_dict(self.query_dict)
        self.gallery_id2idx = self.inverse_dict(self.gallery_dict)


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


    def single_query(self, query_id, query_feat, gallery_embeds):
        query_dist = []
        for j, feat in enumerate(gallery_embeds):
            adist = cdist(
                    feat.reshape(1,-1), 
                    query_feat.reshape(1,-1), 
                    'euclidean')
            query_dist.append(adist[0][0])
        query_dist = np.array(query_dist)

        order = np.argsort(query_dist)

        single_recall = dict()
        for k in self.topks:
            retrieved_idxes = order[:k]
            tp = 0
            relevant_num = len(self.gallery_id2idx[query_id])
            for idx in retrieved_idxes:
                retrieved_id = self.gallery_dict[idx]
                if query_id == retrieved_id:
                    tp += 1
            single_recall[k] = float(tp) / min(relevant_num,k)

        return single_recall


    def show_results(self):
        print('--------------- Retrieval Evaluation ------------')
        for k in self.topks:
            recall = 100 * float(sum(self.recall[k])) / len(self.recall[k])
            print('Recall@%d = %.2f' % (k, recall))


    def evaluate(self, query_embeds, gallery_embeds):
        for i, query_feat in enumerate(query_embeds):
            query_id = self.query_dict[i]
            single_recall = self.single_query(query_id, 
                                              query_feat,
                                              gallery_embeds)
            
            for k in self.topks:
                self.recall[k].append(single_recall[k])
            self.show_results()
            
        self.show_results()


if __name__ == "__main__":
   query_embeds = sio.loadmat('query_embeds.mat')['embeds']
   print('query_embeds', query_embeds.shape)

   gallery_embeds = sio.loadmat('query_embeds.mat')['embeds']
   print('gallery_embeds', gallery_embeds.shape)

   query_dict = 'datasets/In-shop/Anno/query_idx2id.txt'
   gallery_dict = 'datasets/In-shop/Anno/query_idx2id.txt'
   e = Evaluator(query_dict, gallery_dict)
   e.evaluate(query_embeds, gallery_embeds)
