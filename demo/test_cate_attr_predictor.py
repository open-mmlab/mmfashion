from __future__ import division
import argparse
import json
import torch

from flask import Flask, render_template, request

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor

app = Flask(__name__)

cfg = Config.fromfile('configs/category_attribute_predict/global_predictor_resnet.py')
# global attribute predictor will not use landmarks
# just set a default value
landmark_tensor = torch.zeros(8)
model = build_predictor(cfg.model)
load_checkpoint(model, 'checkpoint/resnet50.pth', map_location='cpu')
model.cuda()
landmark_tensor = landmark_tensor.cuda()
model.eval()

@app.route("/api", methods=["POST"])
def apiv1():

    input_path = request.form['path']
    img_idx = request.form['img_idx']

	# predict probabilities for each attribute
    img_tensor = get_img_tensor(input_path, True)

    attr_prob, cate_prob = model(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
    attr_predictor = AttrPredictor(cfg.data.test)
    cate_predictor = CatePredictor(cfg.data.test)

    res = {
        "timeUsed": 0.063, "predictions": {
            "image_"+str(img_idx): {
                "attributes": attr_predictor.show_json(attr_prob),
                "category": cate_predictor.show_json(cate_prob)
            }
        }, "success": True
    }

    return res

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80, debug=True)