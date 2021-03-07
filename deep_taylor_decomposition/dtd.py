#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import *
from model import saliency_mapping as sa_map
from torch.autograd import Variable


sample_dir = "saliency_map_sample"
test_size = 2000
batch_size = 1
heatmap_scale = 5000


def test():
	logger = logging.getLogger()
	logger.setLevel("DEBUG")
	formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
	stream_handler = logging.StreamHandler()
	stream_handler.setLevel("INFO")
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	logging.info('data loading')
	data_raw = torch.load("dataTensor.pt")
	data_raw = data_raw[-test_size:]

	data_raw = data_raw[:10]

	data_X = data_raw[:, :, :-1].transpose(1, 2)
	data_Y = data_raw[:, :, -1]

	logging.info('prepare model')
	model = torch.load("sepsis_predict.pt").cpu()
	model.train(False)
	module_list = sa_map.model_flattening(model)
	act_store_model = sa_map.ActivationStoringNet(module_list)
	DTD = sa_map.DTD()

	logging.info('testing with saliency mapping start')

	test_count = 0
	with torch.no_grad():
		while test_count + batch_size <= data_X.shape[0]:
			input_data = data_X[test_count:test_count + batch_size]
			input_data = Variable(input_data)
			# target = Variable(data_Y[test_count:test_count + batch_size])
			test_count += batch_size

			module_stack, output = act_store_model(input_data)

			logging.info('sample saliency map generation')
			saliency_map = DTD(module_stack, output, 336, 'TCN')
			# saliency_map = torch.sum(saliency_map, dim=1)
			saliency_map_sample = saliency_map[0].detach().numpy()
			saliency_map_sample = np.maximum(0, saliency_map_sample) * 255 * heatmap_scale
			saliency_map_sample = np.minimum(255, saliency_map_sample)
			saliency_map_sample = np.uint8(saliency_map_sample)
			saliency_heatmap = cv2.applyColorMap(saliency_map_sample, cv2.COLORMAP_BONE)

			heatmap_name = f"sepsis_{int(test_count / batch_size)}th_sample.png"
			cv2.imwrite(os.path.join(sample_dir, heatmap_name), saliency_heatmap)
			sample_origin = input_data.cpu().data[0]
			origin_name = f"sepsis_{int(test_count / batch_size)}th_origin.png"
			save_image(sample_origin, os.path.join(sample_dir, origin_name))

	logging.info('test finish')


if __name__ == '__main__':
	test()
