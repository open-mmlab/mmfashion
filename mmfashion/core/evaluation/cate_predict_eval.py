import numpy as np
import torch


class CateCalculator(object):
    """Calculate Category prediction top-k recall rate
    """
    def __init__(self, cfg, topns=[1, 3]):
        self.collector = dict()
        self.num_cate = cfg.category_num
        self.topns = topns
        # true positive
        for topi in topns:
            self.collector['top%s' % str(topi)] = dict()
            tp = np.zeros(self.num_cate)
            fn = np.zeros(self.num_cate)
            self.collector['top%s' % str(topi)]['tp'] = tp
            self.collector['top%s' % str(topi)]['fn'] = fn

        """ topn recall rate """
        self.recall = dict()
        
        # collect target per category
        self.target_per_cate = np.zeros(self.num_cate)

        # num of total predictions
        self.total = 0
        # topn category prediction
        self.topns = topns

    def collect(self, indexes, target, topk):
        """calculate and collect recall rate
        Args:
            indexes(list): predicted top-k indexes  
            target(list): ground-truth
            topk(str): top-k, e.g., top1, top3, top5
        """
        for i, cnt in enumerate(self.collector[topk]['tp']): # true-positive
            if i in indexes and i in target:
                self.collector[topk]['tp'][i] += 1

        for i, cnt in enumerate(self.collector[topk]['fn']): # false negative
            if i not in indexes and i in target:
                self.collector[topk]['fn'][i] += 1

    def collect_result(self, pred, target):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        for i in range(pred.size(0)):
            self.total += 1
            indexes = np.argsort(data[i])[::-1]
            for k in self.topns:
                idx = indexes[:k]
                self.collect(idx, target[i], 'top%d'%k)

    def compute_one_recall(self, tp, fn):
        empty = 0
        recall = np.zeros(tp.shape)
        for i, num in enumerate(tp):
            # ground truth number = true_positive(tp) + false_negative(fn)
            if tp[i] + fn[i] == 0:
                empty += 1
                continue
            else:
                recall[i] = float(tp[i]) / float(tp[i] + fn[i])
        sorted_recall = sorted(recall)[::-1]
        return 100 * sum(sorted_recall) / (len(sorted_recall) - empty)

    def compute_recall(self):
        for key, top in self.collector.items():
            self.recall[key] = self.compute_one_recall(top['tp'], top['fn'])


    def show_result(self, batch_idx=None):
        print('----------- Category Prediction ----------')
        if batch_idx is not None:
            print('Batch[%d]' % batch_idx)
        else:
            print('Total')

        self.compute_recall()
        print('[ Recall Rate ]')
        for k in self.topns:
            print('top%d = %.2f' % (k, self.recall['top%d' % k]))
