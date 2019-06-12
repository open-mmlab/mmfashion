import numpy as np
import torch

class Calculator(object):
    def __init__(self, num_classes, tops_type=[3,5,10]):
        """ create the empty array to count 
        true positive(tp), true negative(tn), false positive(fp) and false negative(fn);
        Args:
        class_num : number of classes in the dataset
        tops_type : default calculate top3, top5 and top10
        """
        self.collector = dict()
        self.total = 0 # the number of total predictions
        for i in tops_type:
            tp, tn, fp, fn = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
            self.collector['top%s'% (str(i))] = dict()
            self.collector['top%s'% (str(i))]['tp'] = tp
            self.collector['top%s'% (str(i))]['tn'] = tn
            self.collector['top%s'% (str(i))]['fp'] = fp
            self.collector['top%s'% (str(i))]['fn'] = fn
        
        """ precision = true_positive/(true_positive+false_positive)"""
        self.precision = dict()
        
        """ accuracy = (true_positive+true_negative)/total_precision"""
        self.accuracy = dict()

    def collect(self, indexes, target, top):
        for i, t in enumerate(target):
            if t==1:
               if i in indexes:
                  top['tp'][i]+=1
               else:
                  top['fn'][i]+=1
            if t==0:
               if i in indexes:
                  top['fp'][i]+=1
               else:
                  top['tn'][i]+=1
    
    def collect_result(self, pred, target):
        if isinstance(pred, torch.Tensor):
           data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
           data = pred
        else:
           raise TypeError('type {} cannot be calculated.'.format(
                            type(data)))

        for i in range(pred.size(0)):
            self.total += 1
            indexes = np.argsort(data[i])[::-1]
            idx3, idx5, idx10 = indexes[:3], indexes[:5], indexes[:10]

            self.collect(idx3, target[i], self.collector['top3'])
            self.collect(idx5, target[i], self.collector['top5'])
            self.collect(idx10, target[i], self.collector['top10'])
    
    def compute_one_precision(self, tp, fp):
        empty = 0
        precision = np.zeros(tp.shape)
        for i, num in enumerate(tp):
            if tp[i]+fp[i] == 0 :
               empty += 1
               continue
            else:
               precision[i] = float(tp[i])/float(tp[i]+fp[i])
        return 100*float(np.sum(precision))/ (len(precision)-empty)


    def compute_precision(self):
        for key, top in self.collector.items():
            self.precision[key] = self.compute_one_precision(top['tp'], top['fp'])
        return self.precision
        
    def compute_one_accuracy(self, tp, tn):
        empty = 0
        accuracy = np.zeros(tp.shape)
        for i, num in enumerate(tp):
            accuracy[i] = float(tp[i]+tn[i])/float(self.total)
        return 100*float(np.sum(accuracy))/len(accuracy)

    def compute_accuracy(self):
        for key, top in self.collector.items():
            self.accuracy[key] = self.compute_one_accuracy(top['tp'], top['tn'])

    def show_result(self, batch_idx=None):
        if batch_idx is not None:
           print('\n')
           print('Batch[%d]' % batch_idx)
        else:
           print('Total')
           print('Batch[%d]' % batch_idx)

        self.compute_precision() 
        print('[Precision] top3 = %.2f, top5 = %.2f, top10 = %.2f' 
              % (self.precision['top3'], 
                 self.precision['top5'], 
                 self.precision['top10']))

        self.compute_accuracy()
        print('[Accuracy] top3 = %.2f, top5 = %.2f, top10 = %.2f' 
              % (self.accuracy['top3'], 
                 self.accuracy['top5'], 
                 self.accuracy['top10']))


         
