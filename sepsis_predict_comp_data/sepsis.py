from tcn import TCN
from data_generator import data_generator
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from time import time

if not torch.cuda.is_available():
	raise Exception("CUDA is not available")

# ---------------- 玄学部分 ---------------------

dropout = 0.1
clip = 0.0  # 防止梯度爆炸的梯度保护临界值
optim = 'Adam'
lr = 1e-3
epochs = 50  # 最大epoch
nhid = 80  # 每个隐层的参数量
levels = 3  # 时间卷积隐层数量
kernel_size = 5  # 卷积核尺寸
log_interval = 2000  # 记录log的间隔

# ------------------ 固定参数 -------------------

input_size = 40
output_size = 336
batch_size = 200

nan_alg = 1

seed = 6783
model_version = "5"

# -----------------------------------------------

# data
train_set, valid_set, test_set = data_generator(nan_alg=nan_alg)
train_set, valid_set, test_set = train_set.cuda(), valid_set.cuda(), test_set.cuda()

# model
torch.manual_seed(seed)
n_channels = [nhid] * levels

model = TCN(input_size, output_size, n_channels, kernel_size, dropout=dropout)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr)

# loss_function = nn.BCELoss()
loss_function = nn.MSELoss()
# def loss_function(input, target):
# 	o = input.flatten()
# 	x = torch.stack((o, 1-o), dim=1)
# 	return nn.CrossEntropyLoss(x, target.flatten())


def evaluate(X_data, name='Eval'):
	model.eval()
	eval_idx_list = np.arange(len(X_data), dtype="int32")
	np.random.shuffle(eval_idx_list)
	total_loss = 0.0
	count = 0
	with torch.no_grad():
		while count <= len(X_data) - batch_size:
			data_line = X_data[count: count + batch_size].transpose(1, 2)
			x, y = Variable(data_line[:, :-1]), Variable(data_line[:, -1])
			x, y = x.cuda(), y.cuda()
			output = model(x)
			loss = loss_function(output, y)
			total_loss += loss.item()
			count += output.size(0)
		eval_loss = total_loss / count * batch_size
		print(name + " loss: {:.5f}".format(eval_loss))
		return eval_loss


def train(_):
	model.train()
	total_loss = 0
	count = 0
	index = 0
	while index < len(train_set):
		data_batch = train_set[index:index + batch_size].transpose(1, 2)
		index += batch_size
		x, y = Variable(data_batch[:, :-1]), Variable(data_batch[:, -1])
		x, y = x.cuda(), y.cuda()
		optimizer.zero_grad()
		output = model(x)
		loss = loss_function(output, y)
		if torch.isnan(loss):
			raise Exception("loss is nan")
		total_loss += loss.item()
		count += output.size(0)
		if clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		loss.backward()
		optimizer.step()
		if index % log_interval == 0:
			cur_loss = total_loss / count * batch_size
			print("Epoch {:2d} | lr {:.8f} | loss {:.6f}".format(ep, lr, cur_loss))
			total_loss = 0.0
			count = 0


def extract():
	count = 0
	outputs = np.empty((0, 336))
	with torch.no_grad():
		data = test_set.transpose(1, 2)
		while count <= len(test_set) - batch_size:
			x = Variable(data[count: count + batch_size, :-1]).cuda()
			output = model(x).cpu().numpy()
			outputs = np.vstack((outputs, output))
			count += batch_size
	y = data[:, -1].cpu().numpy()
	timeline = data[:, -2].cpu().numpy()
	out = np.stack((timeline, y, outputs), axis=0)
	out = np.swapaxes(out, axis1=0, axis2=1)
	out[:, -1] *= 10
	out[:, -1][out[:, -1] > 1] = 1
	np.save("test_data", out)
	return out


best_vloss = 1e4
vloss_list = []
model_name = "sepsis_predict_" + model_version + ".pt"
t0 = time()
for ep in range(1, epochs + 1):
	train(ep)
	vloss = evaluate(valid_set, name='Validation')
	tloss = evaluate(test_set, name='Test')
	if vloss < best_vloss:
		with open(model_name, "wb") as f:
			torch.save(model, f)
			print("Saved model!\n")
		best_vloss = vloss
	if ep > 10 and vloss > max(vloss_list[-3:]):
		lr /= 10
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	vloss_list.append(vloss)

# state_name = "sepsis_predict_state_" + model_version + ".pt"
# torch.save(model.state_dict(), state_name)

print('-' * 90)
model = torch.load(model_name)
tloss = evaluate(test_set)

t1 = time()
print('time cost: {:.2f}s'.format(t1 - t0))

extract()
