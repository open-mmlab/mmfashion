import numpy as np
import torch


class AttrPredictor(object):

    def __init__(self, cfg, tops_type=[3]):
        """Create the empty array to count true positive(tp),
            true negative(tn), false positive(fp) and false negative(fn).

        Args:
            class_num : number of classes in the dataset
            tops_type : default calculate top3, top5 and top10
        """

        attr_cloth_file = open(cfg.attr_cloth_file).readlines()
        self.attr_idx2name = {}
        for i, line in enumerate(attr_cloth_file[2:]):
            self.attr_idx2name[i] = line.strip('\n').split()[0]

        self.tops_type = tops_type

    def print_attr_name(self, pred_idx):
        for idx in pred_idx:
            print(self.attr_idx2name[idx])

    def show_prediction(self, pred):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        for i in range(pred.size(0)):
            indexes = np.argsort(data[i])[::-1]
            for topk in self.tops_type:
                idxes = indexes[:topk]
                print('[ Top%d Attribute Prediction ]' % topk)
                self.print_attr_name(idxes)
        
    def show_json(self, pred):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        res = []

        for i in range(pred.size(0)):
            indexes = np.argsort(data[i])[::-1]
            for topk in self.tops_type:
                idxes = indexes[:topk]
                for num in range(topk):
                    res.append({
                        "label": self.attr_idx2name[idxes[num]],
                        "confidence": float(data[i][indexes[num]])
                    })
        return res
