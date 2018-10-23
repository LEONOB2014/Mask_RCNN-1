#coding:utf-8

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# import coco
# from coco import coco
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.model import  log
from mrcnn import visualize
import yaml
import cv2
import glob


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
class_names = ['BG', '119']

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = len(class_names)  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 800

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)
# model_path = model.find_last()[1]
model_path = "/home/han/Mask_RCNN/samples/shapes/logs/shapes20181023T1743/mask_rcnn_shapes_0030.h5"
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



# class_names = ['BG' , '5', '6', '9', '10', '11', '16', '20', '22', '24', '26', '30', '33', '35', '37', '39', '40', '41', '42', '44', '46', '61', '63', '64', '65', '66', '67', '72', '73', '74', '75', '77', '78', '82', '83', '84', '85', '86', '88', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '104', '105', '106', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159']




def predict(img_name , model):
	image = skimage.io.imread(img_name)
	image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
	results = model.detect([image] , verbose=1)
	
	# boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates
	boxes = results[0]['rois']
	# masks: [height, width, num_instances]
	masks = results[0]['masks']
	# class_ids: [num_instances]
	class_ids = results[0]['class_ids']

	N = boxes.shape[0]
	print('N:' , N)
	if not N:
		print("No object")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
		
	if N != 1:
		print('该张图片检测出多个物体')
		return None , None , None
	else:
		label_img = np.zeros(masks.shape, dtype=np.uint8)
		label_img[np.where(masks == 1)] = 255
		label = class_names[class_ids[0]]
		
		colors = visualize.random_colors(N , bright=True)
		
		mask_image = visualize.apply_mask(image , masks[:,:,0] , colors[0])
		
		
		
		return label_img , label , mask_image
	


# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
file_name = "/home/han/Desktop/hhahah/5-71fc4fc6-d665-11e8-9536-cc3d82bf7b29-2.png"
image = skimage.io.imread(file_name)
print('shape of image:' , np.shape(image))

# Run detection
results = model.detect([image], verbose=1)
print('results:' , results)

mask = results[0]['masks']
print('shape of mask:' , np.shape(mask))

label_img = np.zeros(mask.shape , dtype=np.uint8)
label_img[np.where(mask == 1)] = 255

# cv2.imshow("label_img" , label_img)
# cv2.waitKey(0)
#
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                             class_names, r['scores'])

# if __name__ == '__main__':
# 	data_dir = "/home/han/Desktop/hhahah"
# 	img_lsts = glob.glob(os.path.join(data_dir , "*.png"))
#
# 	for i in img_lsts:
# 		print(i)
# 		label_img , label , mask_image = predict(i , model)
# 		if type(label_img) == type(None):
# 			continue
#
# 		cv2.imshow(str(label), label_img)
# 		cv2.imshow("mask_image" , mask_image)
# 		cv2.waitKey(1000)

	

