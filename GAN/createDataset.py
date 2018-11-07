import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


imdir = 'BirdImages/'
img_arr = []


def preprocess_img(img):
	imr = img
	imr = cv2.resize(imr, (256,256), interpolation=cv2.INTER_AREA)

	return imr


for im in tqdm(os.listdir(imdir)):
	img = plt.imread(os.path.join(imdir, im))
	im_processed = preprocess_img(img)
	img_arr.append(im_processed)

img_arr = np.array(img_arr)
np.save('bird_imgs.npy', img_arr)
