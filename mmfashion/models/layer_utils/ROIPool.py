import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from models.config import cfg
import numpy as np


def extract_rois(landmarks):

    batch_rois = []

    for k, cors in enumerate(landmarks):
        for i, cor in enumerate(cors):
            if i % 2 == 0:  #x
                x = 14 * (cor.data.tolist()[0] / 224.0)
                y = 14 * (cors[i + 1].data.tolist()[0] / 224.0)

            else:
                continue
            x1 = max(x - 3, 0)
            x2 = min(x + 4, 13)
            y1 = max(y - 3, 0)
            y2 = min(y + 4, 13)

            if x1 == 0:
                x2 = 6
            if y1 == 0:
                y2 = 6
            if x2 == 13:
                x1 = 7
            if y2 == 13:
                y1 = 7
            one_rois = np.array([x1, y1, x2, y2], dtype=np.int)
            if i == 0:
                rois = one_rois

            else:

                rois = np.append(rois, one_rois)
                rois.reshape(-1)  # change to 1D
        batch_rois.append(rois)
    return batch_rois


def ROIPooling(in_features, landmarks, fc):

    batch_rois = extract_rois(landmarks)

    batch_size = len(batch_rois)

    for k, rois in enumerate(batch_rois):
        for i, cor in enumerate(rois):
            if i % 4 == 0:  #x1
                x1 = cor
                y1 = rois[i + 1]
                x2 = rois[i + 2]
                y2 = rois[i + 3]
            else:
                continue

            if x2 - x1 < 7:
                x2 = x2 + 1
            if y2 - y1 < 7:
                y2 = y2 + 1
            out_features = in_features[k][:, x1:x2, y1:y2].contiguous()

            out_feature_r = out_features.view(-1)

            second_branch_out = fc(out_feature_r)

            if i == 0:
                second_fc_output = second_branch_out
            else:
                second_fc_output = torch.cat(
                    (second_fc_output, second_branch_out), 0)
        second_fc_output = second_fc_output.view(-1)

        if k == 0:
            roi_output = second_fc_output
        else:
            roi_output = torch.cat((roi_output, second_fc_output))

    roi_output = roi_output.view((batch_size, 4096))

    return roi_output
