import torch
import numpy as np
import os
from tqdm import tqdm


def read_psv(f, m):
	lines = f.readlines()
	b = len(m) - len(lines)
	for i in range(1, len(lines)):
		line = lines[i][:-1].split('|')
		line = np.array(line, dtype=np.float32)
		m[i + b] = line[:]


def nan_process(m, alg=0):
	# 输入m矩阵尺寸为 (20000, 336, 41)
	if alg == 0:
		# fill NaN value with 0 
		m[np.isnan(m)] = 0
	elif alg == 1:
		# fill NaN value with col average
		for i in range(m.shape[2]):
			c_mean = np.nanmean(m[:, :, i])
			m[:, :, i][np.isnan(m[:, :, i])] = c_mean
	return m


def data_generator(dataset='B', nan_alg=0):
	"""
	读入数据并分割为训练集、验证集、测试集，默认使用setB

	Args:
		dataset: 使用的数据集         'A'为竞赛数据集A，'B'为竞赛数据集B，'M'为MIMIC-III数据集
		nan_alg: 处理空值使用的算法    0为填充0，1为填充平均值

	>> note:
		setA数据文件索引范围1~20643，19000之后有序号缺失，共计20337个病人的记录，最长时间序列长度336
			导出矩阵尺寸为 X:(20337, 336, 40) Y:(20337, 336)
		setB数据文件索引范围100001~120000，共计20000个病人的记录，最长时间序列长度336
			导出矩阵尺寸为 X:(20000, 336, 40) Y:(20000, 336)
	"""

	file_name_pattern = 'p{:0=6}.psv'
	if dataset == 'A':
		data_dir = "../data/training/"
		file_index = list(range(1, 20644))
	elif dataset == 'B':
		data_dir = "../data/training_setB/"
		file_index = list(range(100001, 120001))
	elif dataset == 'M':
		pass

	np.random.seed(1234888)
	np.random.shuffle(file_index)

	split_ratio = np.array([0.7, 0.15, 0.15])  # train_set 0.7, valid_set 0.15, test_set 0.15
	split_len = np.round(split_ratio * len(file_index)).astype(np.int16)

	# read file
	if os.path.isfile("dataTensor.pt"):
		data = torch.load("dataTensor.pt")
	else:
		index = 0
		data = np.zeros((len(file_index), 336, 41))
		for j in tqdm(file_index):
			file_name = file_name_pattern.format(j)
			try:
				datafile = open(data_dir + file_name)
				read_psv(datafile, data[index])
				datafile.close()
			except:
				print(file_name, " doesn't exist")
			index += 1
		data = nan_process(data, nan_alg)
		data = torch.Tensor(data)
		torch.save(data, "dataTensor.pt")
	train_set = data[:split_len[0]]
	valid_set = data[split_len[0]:split_len[0] + split_len[1]]
	test_set = data[split_len[0] + split_len[1]:]

	return train_set, valid_set, test_set
