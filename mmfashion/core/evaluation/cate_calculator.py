import numpy as np
import operator
from scipy.spatial.distance import cdist

import torch

class CateCalculator(object):
    def __init__(self, cfg, topn=10):
        self.collector = dict()
        self.num_cate = cfg.category_num
        # true positive
        self.tp = np.zeros(self.num_cate)
        
        # collect target per category
        self.target_per_cate = np.zeros(self.num_cate)

        # num of total predictions
        self.total = 0


    def collect_result(self, pred, target):
        if isinstance(pred, torch.Tensor):
           data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
           data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))
        
        for i in range(pred.size(0)): # batch size
            self.total += 1
            pred_idx = np.argsort(data[i])[-1]
            self.target_per_cate[target[i]] += 1

            if pred_idx == target[i]:
               self.tp[int(pred_idx)] += 1


    def show_result(self):
        print('----------- Category Prediction -----------')
        recall_rate = np.zeros(self.num_cate)
        empty = 0 
        for i, tp in enumerate(self.tp):
            recall_rate[i] = float(tp) / self.total
            if self.target_per_cate[i] == 0:
               empty += 1

        # average recall rate for all categories
        avg_recall_rate = 100 * sum(recall_rate) / (self.num_cate-empty)
        print('[Recall Rate] = %.2f' % avg_recall_rate)
