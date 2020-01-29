import torch.nn as nn

from ..registry import LANDMARKFEATUREEXTRACTOR


@LANDMARKFEATUREEXTRACTOR.register_module
class LandmarkFeatureExtractor(nn.Module):

    def __init__(self, inchannels, feature_dim, landmarks):
        super(LandmarkFeatureExtractor, self).__init__()
        self.linear = nn.Linear(inchannels, landmarks * feature_dim)
        self.landmarks = landmarks
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.landmarks, self.feature_dim)
        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.01)
