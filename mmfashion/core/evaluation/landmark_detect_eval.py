import numpy as np
from numpy.linalg import norm as norm
import scipy.io as sip
from scipy.spatial.distance import cdist as cdist


class NormalizedErrorEvaluator(object):
     def __init__(self, img_size, landmark_num):
         self.w = img_size[0]
         self.h = img_size[1]
         self.landmark_num = landmark_num
     
     def compute_distance(self, pred_lms, lms):
         for i, pred_lm in enumerate(pred_lms):
             gt_lm = np.array([lms[i][0]/self.w, lms[i][1]/self.h], dtype=np.float)
             pred_lm_data = np.array([pred_lm[0]/self.w, pred_lm[1]/self.h], dtype=np.float)
                         
             dist = norm(gt_lm-pred_lm_data)

         avg_dist = dist/self.landmark_num
         return avg_dist


     def compute_normalized_error(self,
                                  pred_vis, pred_lm, 
                                  vis, landmark):
         batch_size = pred_lm.size(0)
         pred_lm_np = pred_lm.cpu().detach().numpy()
         landmark_np = landmark.cpu().detach().numpy()
         pred_vis = pred_vis.cpu().detach().numpy()
         vis = vis.cpu().detach().numpy()

         pred_lm_np = np.reshape(pred_lm_np.astype(np.float), (batch_size,self.landmark_num,2))
         landmark_np = np.reshape(landmark_np.astype(np.float), (batch_size,self.landmark_num,2))
         pred_vis = np.reshape(pred_vis, (batch_size, self.landmark_num, 1))
         vis = np.reshape(vis, (batch_size, self.landmark_num, 1))
         
         dist = self.compute_distance(pred_vis*pred_lm_np, vis*landmark_np)
         return dist
 
                     
