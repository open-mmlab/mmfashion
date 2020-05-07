import argparse

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Fashion Detection and Segmentation')
    parser.add_argument(
        '--config',
        help='mmfashion config file path',
        default='configs/mmfashion/mask_rcnn_r50_fpn_1x.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='demo/01_4_full.jpg')


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config_file, args.checkpoint, device='cuda:0')

    # test a single image and show the results
    img = args.input
    result = inference_detector(model, img)

    # visualize the results in a new window
    # or save the visualization results to image files
    show_result(
        img, result, model.CLASSES, out_file=img.split('.')[0] + '_result.jpg')
