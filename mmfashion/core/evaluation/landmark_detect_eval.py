import numpy as np


class LandmarkDetectorEvaluator(object):

    def __init__(self,
                 img_size,
                 landmark_num,
                 prob_threshold=0.3,
                 dist_threshold=10,
                 demo=False):
        self.w = img_size[0]
        self.h = img_size[1]
        self.landmark_num = landmark_num
        self.prob_threshold = prob_threshold
        self.dist_threshold = dist_threshold
        self.demo = demo

    def compute_distance(self, pred_lms, gt_lms):
        """Compute the percentage of detected landmarks,
            if pixel distance <= dist_threshold, such landmark is detected.
        Args:
            pred_lms(list): predicted landmarks
                [[pred_lm1_x, pred_lm1_y], [pred_lm2_x, pred_lm2_y], ...]
            gt_lms(list): ground truth landmarks
                [[gt_lm1_x, gt_lm1_y],[gt_lm2_x, gt_lm2_y], ...]
        """
        detected = 0  # the number of detected landmarks
        valid = 0  # the number of valid/visible landmarks
        norm_error_list = []

        for i, (pred_lms_per_image,
                gt_lms_per_image) in enumerate(zip(pred_lms, gt_lms)):
            if self.demo:
                print(self.img_idx_to_name[i])
                print('pred', pred_lms_per_image)
                print('gt', gt_lms_per_image)
                print('\n')

            for j, (pred_lm, gt_lm) in enumerate(
                    zip(pred_lms_per_image, gt_lms_per_image)):
                # compute normalized error per landmark
                gt_lm_x = float(gt_lm[0]) / self.w
                gt_lm_y = float(gt_lm[1]) / self.h
                pred_lm_x = float(pred_lm[0]) / self.w
                pred_lm_y = float(pred_lm[1]) / self.h

                gt_lm_arr = np.array([gt_lm_x, gt_lm_y])
                pred_lm_arr = np.array([pred_lm_x, pred_lm_y])
                norm_error = np.linalg.norm(gt_lm_arr - pred_lm_arr)
                norm_error_list.append(norm_error)

                # compute the pixel distance per landmark
                dist = np.linalg.norm(pred_lm - gt_lm)
                if dist <= self.dist_threshold:
                    detected += 1

                valid += 1

        avg_norm_error = sum(norm_error_list) / len(norm_error_list)
        det_percent = 100 * float(detected) / valid
        return avg_norm_error, det_percent

    def evaluate_landmark_detection(self, pred_vis, pred_lm, vis, landmark):
        """Evaluate landmark detection.

        Args:
            pred_vis (tensor): predicted landmark visibility.
            pred_lm (tensor): predicted landmarks.
            vis (tensor): ground truth of landmark visibility.
            landmarks(tensor): ground truth of landmarks.

        Returns:
            dist: average value of landmark detection normalized error per
                image.
            detected_lm_percent: average value of detected landmarks per image.
        """
        batch_size = pred_lm.size(0)
        pred_lm_np = pred_lm.cpu().detach().numpy()
        landmark_np = landmark.cpu().detach().numpy()
        pred_vis = pred_vis.cpu().detach().numpy()
        vis = vis.cpu().detach().numpy()

        pred_lm_np = np.reshape(
            pred_lm_np.astype(np.float), (batch_size, self.landmark_num, 2))
        landmark_np = np.reshape(
            landmark_np.astype(np.float), (batch_size, self.landmark_num, 2))

        # pred_vis_prob >= self.prob_threshold, view as True
        pred_vis_prob = np.reshape(pred_vis,
                                   (batch_size, self.landmark_num, 1))
        pred_vis_bool = pred_vis_prob >= self.prob_threshold
        pred_vis = pred_vis_bool * 1

        vis = np.reshape(vis, (batch_size, self.landmark_num, 1))
        pred_vis = np.reshape(pred_vis, (batch_size, self.landmark_num, 1))

        normalized_error, det_percent = self.compute_distance(
            vis * pred_lm_np, vis * landmark_np)
        return normalized_error, det_percent

    def compute_vis_prediction_accuracy(self, pred_vis, vis):
        """Compute the percentage of detected landmarks.

        Args:
            pred_vis(list): predicted landmark visibility
                [[lm1_pred, lm2_pred, ...], ...]
            vis(list): ground truth landmark visibility
                [[lm1_gt, lm2_gt, ...], ...]
        """
        correct = 0
        total = pred_vis.shape[0] * pred_vis.shape[1]

        for i, pred_row in enumerate(pred_vis):
            for j, per_pred in enumerate(pred_row):
                if per_pred >= 0.5 and vis[i][j] >= 0.5:
                    correct += 1
        acc = float(correct * 100) / total
        return acc
