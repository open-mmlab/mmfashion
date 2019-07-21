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


    def evaluate(self, distances, labels, nrof_folds=10):
        # Calculate evaluation metrics
        thresholds = np.arange(0, 30, 0.01)
        tpr, fpr, accuracy = calculate_roc(thresholds, distances,
                                           labels, nrof_folds=nrof_folds)
        thresholds = np.arange(0, 30, 0.001)
        val, val_std, far = calculate_val(thresholds, distances,
                                          labels, 1e-3, nrof_folds=nrof_folds)
        return tpr, fpr, accuracy, val, val_std, far        


    def calculate_roc(self, thresholds, distances, labels, nrof_folds=10):
        nrof_pairs = min(len(labels), len(distances))
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds,nrof_thresholds))
        fprs = np.zeros((nrof_folds,nrof_thresholds))
        accuracy = np.zeros((nrof_folds))

        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                 _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
                _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])

            tpr = np.mean(tprs,0)
            fpr = np.mean(fprs,0)
        return tpr, fpr, accuracy


    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/dist.size
        return tpr, fpr, acc


    def calculate_val(self, thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
        nrof_pairs = min(len(labels), len(distances))
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)

        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                 _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
            if np.max(far_train)>=far_target:
               f = interpolate.interp1d(far_train, thresholds, kind='slinear')
               threshold = f(far_target)
            else:
               threshold = 0.0

            val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

        val_mean = np.mean(val)
        far_mean = np.mean(far)
        val_std = np.std(val)
        return val_mean, val_std, far_mean


    def calculate_val_far(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        if n_diff == 0:
           n_diff = 1
        if n_same == 0:
           return 0,0
        val = float(true_accept) / float(n_same)
        far = float(false_accept) / float(n_diff)
        return val, far 
