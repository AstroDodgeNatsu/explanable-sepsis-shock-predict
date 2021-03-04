import numpy as np
import os
import sys
import csv
from tqdm import tqdm


def create_csv1(path, data, t, j):
	with open(path, 'w', newline='') as f:
		csv_write = csv.writer(f, delimiter='|')
		header = ['SepsisLabel']
		csv_write.writerow(header)
		for i in range(t):
			list = [data[1][j]]
			j += 1
			csv_write.writerow(list)


def create_csv2(path, data, t, j):
	with open(path, 'w', newline='') as f:
		csv_write = csv.writer(f)
		header = ['PredictedProbability|PredictedLabel']
		csv_write.writerow(header)
		for i in range(t):
			# list = [data[2][j]]
			if data[2][j] > 0.5:
				list = [str(data[2][j]) + '|1']
			else:
				list = [str(data[2][j]) + '|0']
			csv_write.writerow(list)
			j += 1


def revised_path_name(path, suffix_pre, suffix_later):
	path0 = path
	path1 = path0 + '/'
	sys.path.append(path1)
	# print(sys.path)

	files = os.listdir(path0)
	# files = os.listdir('.')
	# print('files', files)

	for filename in files:
		portion = os.path.splitext(filename)
		if portion[1] == suffix_pre:
			newname = portion[0] + suffix_later
			filenamedir = path1 + filename
			newnamedir = path1 + newname

			# os.rename(filename,newname)
			os.rename(filenamedir, newnamedir)


array = np.load("../sepsis_predict_comp_data/test_data.npy")
# 创建CSV 文件保存预测结果
for m in tqdm(range(3000)):
	path1 = "labels/" + str(m + 1) + ".psv"
	path2 = "predictions/" + str(m + 1) + ".psv"
	j = 0
	for i in array[m][0]:
		if i != 0:
			break
		j += 1
	# print(j)
	t = 336 - j
	create_csv1(path1, array[m], t, j)
	create_csv2(path2, array[m], t, j)
	revised_path_name("labels/", ".csv", ".psv")
	revised_path_name("predictions/", ".csv", ".psv")
