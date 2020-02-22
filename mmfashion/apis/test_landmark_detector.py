from __future__ import division

from mmcv.parallel import MMDataParallel

from ..core import LandmarkDetectorEvaluator
from ..datasets import build_dataloader
from .env import get_root_logger


def test_landmark_detector(model,
                           dataset,
                           cfg,
                           distributed=False,
                           validate=False,
                           logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start testing predictor
    if distributed:  # to do
        _dist_test(model, dataset, cfg, validate=validate)
    else:
        _non_dist_test(model, dataset, cfg, validate=validate)


def _non_dist_test(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    print('dataloader built')

    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    evaluator = LandmarkDetectorEvaluator(cfg.img_size, cfg.landmark_num)
    error_list, det_percent_list = [], []

    for batch_idx, testdata in enumerate(data_loader):
        img = testdata['img']
        landmark = testdata['landmark_for_regression']
        vis = testdata['vis']

        pred_vis, pred_lm = model(img, return_loss=False)
        det_error, det_lm_percent = evaluator.evaluate_landmark_detection(
            pred_vis, pred_lm, vis, landmark)
        if batch_idx % 20 == 0:
            print('Batch idx {:d}, normalized error = {:.4f}, '
                  'det. percent = {:.2f}'.format(batch_idx, det_error,
                                                 det_lm_percent))
            error_list.append(det_error)
            det_percent_list.append(det_lm_percent)

    print('Fashion Landmark Detection Normalized Error: {:.4f}, '
          'Detected Percent: {:.2f}'.format(
              sum(error_list) / len(error_list),
              sum(det_percent_list) / len(det_percent_list)))


def _dist_test(model, dataset, cfg, validate=False):
    raise NotImplementedError
