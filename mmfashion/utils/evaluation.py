import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist as cdist


class Evaluator(object):
    def __init__(self, topns=[3,5,10,20]):
        """ create the empty array to count 
        true positive(tp), true negative(tn), false positive(fp) and false negative(fn);
        Args:
        class_num : number of classes in the dataset
        tops_type : default calculate top3, top5 and top10
        """
 
        self.topns = topns

        """ prec@k = true_positive/retrieved items(k)"""
        self.prec = dict()
        
        """ recall@k = ture_positive/relevant items(tp+fp)"""
        self.recall = dict()

        """ acc@k = (true_pos+true_neg)/total"""
        self.acc = dict()
         
        for n in topns:
            self.prec[n] = []
            self.recall[n] = []
            self.acc[n] = []
           
        self.query_dict = self.load_dict('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Anno/query_idx2id.txt')
        self.gallery_dict = self.load_dict('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Anno/gallery_idx2id.txt')

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
        
        single_prec = dict()
        single_recall = dict()
        single_acc = dict()
        for n in self.topns:
            retrieved_idxes = order[:n]
            tp,tn,fp,fn=0,0,0,0
            for idx in retrieved_idxes:
                retrieved_id = self.gallery_dict[idx]
                if query_id == retrieved_id:
                   tp += 1
                else:
                   fp += 1
                
            unretrieved_idxes = order[n:]
            for idx in unretrieved_idxes:
                unretrieved_id = self.gallery_dict[idx]
                if query_id != unretrieved_id:
                   tn += 1
                else:
                   fn += 1
                   
            single_prec[n] = float(tp)/n
            single_recall[n] = float(tp)/relevant_items
            single_acc[n] = float(tp+tn)/(tp+tn+fp+fn)
            
        return single_prec, single_recall, single_acc


    def show_results(self):
        print('--------------- Retrieval Evaluation ------------')
        for n in self.topns:
            assert len(self.prec[n])==len(self.recall[n])
            prec = 100*float(sum(self.prec[n]))/ len(self.prec[n])
            recall = 100*float(sum(self.recall[n]))/ len(self.recall[n])
            print('Prec@%d = %.2f' % (n, prec))
            print('Recall@%d = %.2f' % (n, recall))
    
            acc = 100*float(sum(self.acc[n]))/len(self.acc[n])
            print('Acc@%d = %.2f' % (n, acc))

    def evaluate(self, query_embeds, gallery_embeds):
        for i, embed in enumerate(query_embeds):
            query_feat = []
            query_id = self.query_dict[i]
            query_idxes = self.query_id2idx[query_id]
            for idx in query_idxes:
                query_feat.append(query_embeds[idx])
            query_feat = np.array(query_feat)

            single_prec, single_recall, single_acc = self.single_query(query_id, query_feat, gallery_embeds)

            for n in self.topns:
                self.prec[n].append(single_prec[n])
                self.recall[n].append(single_recall[n])
                self.acc[n].append(single_acc[n])

            if i%10==0:
               self.show_results()            

        self.show_results()

e = Evaluator()
query_embeds = sio.loadmat('query_embeds.mat')['embeds']
gallery_embeds = sio.loadmat('gallery_embeds.mat')['embeds']

e.evaluate(query_embeds, gallery_embeds)
