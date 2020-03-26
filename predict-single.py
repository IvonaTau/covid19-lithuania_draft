import os
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np

import boxconfig
import mrcnn.model as modellib
from mrcnn import visualize
import tensorflow as tf
from mrcnn.model import log


def visualize_predictions(results, input_img, rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    r = results[0]
    image_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    visualize.display_instances(image_rgb, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'boxes'], r['scores'], ax=ax,
                                    title="Predictions")
    # plt.show()


class SegmentationModel:
    def __init__(self, weights, device, config):
        with tf.device(device):
            self.model = modellib.MaskRCNN(
                mode="inference", model_dir=weights, config=config)
            self.model.load_weights(weights, by_name=True)

    def predict(self, image):
        image = np.array(image)
        return self.model.detect([image], verbose=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='/home/itautkute/mask_rcnn_arvato/barcode_detection/data/images_arvato/val/013_Color.png')
    parser.add_argument("--model_dir", default='/home/itautkute/mask_rcnn_arvato/barcode_detection/data/logs/boxes20200306T1614/mask_rcnn_boxes_0030.h5')
    parser.add_argument("--device", default="/gpu:0")
    args = parser.parse_args()

    # Load config
    config = boxconfig.InferenceConfig()
    config.display()



    # Load model
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))
    model = SegmentationModel(args.model_dir, args.device, config)

    image = cv2.imread(args.image_dir)

    results = model.predict(image)
    print(results)
