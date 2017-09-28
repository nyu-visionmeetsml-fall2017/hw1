
import sys, os, math, random
sys.path.append('../SSD-Tensorflow')

# Tensorflow and other
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# SSD dependencies
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

# Kinect dependencies
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import cv2

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Lines 27~72 credited to SSD project.

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

print 'DId it work?'

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Code to join SSD code and kinect data stream.

import matplotlib.patches as patches

plt.ion()

fig, ax = plt.subplots(1)
plt.show()

global depth, rgb
while True:
	# Get a fresh frame
	(depth,_), (rgb,_) = get_depth(), get_video()

	rclasses, rscores, rbboxes = process_image(rgb)

	im = np.array(rgb)

	height = len(rgb)
	width = len(rgb[0])

	print 'DETECTED', len(rbboxes)
	ax.cla()
	ax.imshow(im)
	for ii, box in enumerate(rbboxes):
		# bx0, by0, bx1, by1 = box.tolist()
		by0, bx0, by1, bx1 = box.tolist()
		bx0, bx1 = width * bx0, width * bx1
		by0, by1 = height * by0, height * by1

		print '     -', ii, rclasses[ii], rscores[ii]
		print '       ', bx0, bx1, by0, by1

		rect = patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, linewidth=1,edgecolor='r',facecolor='none')
		plt.text(bx0, by1, 'Class: %d - %.1f%%' % (rclasses[ii], rscores[ii] * 100), color='cyan')

		ax.add_patch(rect)

	plt.pause(.01)
	# raw_input(':')

	# print len(rgb), len(rgb[0]), len(rgb[0][0])

	# raw_input(':')
	# cv2.waitKey(5)

# raw_input(':')