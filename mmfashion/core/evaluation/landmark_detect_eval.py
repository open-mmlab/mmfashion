import numpy as np
from numpy.linalg import norm as norm
import scipy.io as sip
from scipy.spatial.distance import cdist as cdist


class LandmarkDetectorEvaluator(object):
     def __init__(self, img_size, landmark_num):
         self.w = img_size[0]
         self.h = img_size[1]
         self.landmark_num = landmark_num
     
     def compute_distance(self, pred_lms, gt_lms, dist=10):
         """ compute the percentage of detected landmarks
         Args:
             pred_lms(list): predicted landmarks [[pred_lm1_x, pred_lm1_y], [pred_lm2_x, pred_lm2_y], ...]
             gt_lms(list): ground truth landmarks [[gt_lm1_x, gt_lm1_y],[gt_lm2_x, gt_lm2_y], ...]
         """
         detected = 0
         total = pred_lms.shape[0] * pred_lms.shape[1]
         print('total', total)
         norm_error_list = []

         for i, (pred_lms_per_image, gt_lms_per_image) in enumerate(zip(pred_lms, gt_lms)):

             #print('pred_lms_per_image', pred_lms_per_image)
             #print('gt_lms_per_image', gt_lms_per_image)

             gt_lm_arr = np.array([gt_lms_per_image[0]/self.w, gt_lms_per_image[1]/self.h], dtype=np.float)
             pred_lm_arr = np.array([pred_lms_per_image[0]/self.w, pred_lms_per_image[1]/self.h], dtype=np.float)
                         
             dist = norm(gt_lm_arr-pred_lm_arr)
             norm_error_list.append(dist)

             for j, (pred_lm, gt_lm) in enumerate(zip(pred_lms_per_image, gt_lms_per_image)):

                 # compute left normalized error and right normalized error
                 
                 dist = norm(pred_lm - gt_lms[i][j])                 
                 if dist<=10:
                    detected += 1 

         avg_dist = sum(norm_error_list)/self.landmark_num
         det_percent = 100*float(detected) / total
         return avg_dist, det_percent


     def evaluate_landmark_detection(self, pred_vis, pred_lm, vis, landmark):
         """ Evaluate landmark detection.
         Args:
             pred_vis (tensor): predicted landmark visibility
             pred_lm (tensor): predicted landmarks
             vis (tensor): ground truth of landmark visibility
             landmarks(tensor): ground truth of landmarks

         Returns:
             dist: average value of landmark detection normalized error per image
             detected_lm_percent: average value of detected landmarks per image
         """
         batch_size = pred_lm.size(0)
         pred_lm_np = pred_lm.cpu().detach().numpy()
         landmark_np = landmark.cpu().detach().numpy()
         pred_vis = pred_vis.cpu().detach().numpy()
         vis = vis.cpu().detach().numpy()

         pred_lm_np = np.reshape(pred_lm_np.astype(np.float), (batch_size,self.landmark_num,2))
         landmark_np = np.reshape(landmark_np.astype(np.float), (batch_size,self.landmark_num,2))

         # pred_vis_prob >= 0.5, view as True 
         pred_vis = np.reshape(pred_vis, (batch_size, self.landmark_num, 1))         
         vis = np.reshape(vis, (batch_size, self.landmark_num, 1))
         
         
         normalized_error, det_percent = self.compute_distance(pred_vis*pred_lm_np, vis*landmark_np)
         return normalized_error, det_percent
 

     def compute_vis_prediction_accuracy(self, pred_vis, vis):
         """ compute the percentage of detected landmarks
         Args:
             pred_vis(list): predicted landmark visibility [[lm1_pred, lm2_pred, ...], ...]
             vis(list): ground truth landmark visibility [[lm1_gt, lm2_gt, ...], ...]
         """
         batch_size = pred_vis.shape[0]
         correct = 0
         total = pred_vis.shape[0]*pred_vis.shape[1]
         
         for i, pred_row in enumerate(pred_vis):
             for j, per_pred in enumerate(pred_row):
                 if per_pred>=0.5 and vis[i][j]>=0.5:
                    correct += 1
         acc = float(correct*100) / total
         return acc
