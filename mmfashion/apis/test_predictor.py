from __future__ import division

from mmcv.parallel import MMDataParallel

from .env import get_root_logger
from ..core import AttrCalculator
from ..datasets import build_dataloader


def test_predictor(model,
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

    attr_calculator = AttrCalculator(cfg)

    for batch_idx, testdata in enumerate(data_loader):
        imgs = testdata['img']
        landmark = testdata['landmark']
        attr = testdata['attr']

        attr_pred = model(imgs, attr, landmark=landmark, return_loss=False)

        attr_calculator.collect_result(attr_pred, attr)

        if batch_idx % cfg.print_interval == 0:
            attr_calculator.show_result(batch_idx)

    attr_calculator.show_result()


def _dist_test(model, dataset, cfg, validate=False):
    raise NotImplementedError
