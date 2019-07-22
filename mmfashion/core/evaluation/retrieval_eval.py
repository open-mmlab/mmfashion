import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist as cdist


class Evaluator(object):
    def __init__(self, query_dict_fn, gallery_dict_fn, topns=[3,5,10,20]):
        """ create the empty array to count 
        Args:
        query_dict_fn : the index of query_embeds to id of query_embeds
        tops_type : default retrieve top3, top5, top10, top20
        """
 
        self.topns = topns

        """ recall@k = ture_positive/relevant items(tp+fp)"""
        self.recall = dict()

        for n in topns:
            self.recall[n] = []
           
        self.query_dict = self.load_dict(query_dict_fn)
        self.gallery_dict = self.load_dict(gallery_dict_fn)

        self.query_id2idx = self.inverse_dict(self.query_dict)
        self.gallery_id2idx = self.inverse_dict(self.gallery_dict)


    def load_dict(self, fn):
        dic = dict()
        rf = open(fn).readlines()
        for i,line in enumerate(rf):
            dic[i] = int(line.strip('\n'))
        return dic

    def inverse_dict(self, idx2id):
        """ invert "idx2id" dict to "id2idx" dict """
        id2idx = dict()
        for k,v in idx2id.items(): # k:idx v:id
            if v not in id2idx:
               id2idx[v] = [k]
            else:
               id2idx[v].append(k)
        return id2idx
     
 
    def single_query(self, query_id, query_feat, gallery_embeds):
        dist = []
        for i, embed in enumerate(gallery_embeds):
            query_dist = []
            for j, feat in enumerate(query_feat):
                adist = cdist(feat.reshape(1,-1), embed.reshape(1,-1),  'euclidean')
                query_dist.append(adist[0][0])
            query_dist = np.array(query_dist)
            dist.append(np.average(query_dist))
        
        order = np.argsort(dist)
        relevant_items = len(self.gallery_id2idx[query_id])
        
        single_recall = dict()
        for n in self.topns:
            retrieved_idxes = order[:n]
            tp=0
            for idx in retrieved_idxes:
                retrieved_id = self.gallery_dict[idx]
                if query_id == retrieved_id:
                   tp += 1
            single_recall[n] = float(tp)/relevant_items
            
        return single_recall


    def show_results(self):
        print('--------------- Retrieval Evaluation ------------')
        for n in self.topns:
            recall = 100*float(sum(self.recall[n]))/ len(self.recall[n])
            print('Recall@%d = %.2f' % (n, recall))
    

    def evaluate(self, query_embeds, gallery_embeds):
        for i, embed in enumerate(query_embeds):
            query_feat = []
            query_id = self.query_dict[i]
            query_idxes = self.query_id2idx[query_id]
            for idx in query_idxes:
                query_feat.append(query_embeds[idx])
            query_feat = np.array(query_feat)

            single_recall = self.single_query(query_id, query_feat, gallery_embeds)

            for n in self.topns:
                self.recall[n].append(single_recall[n])

            if i%10==0:
               self.show_results()            

        self.show_results()

