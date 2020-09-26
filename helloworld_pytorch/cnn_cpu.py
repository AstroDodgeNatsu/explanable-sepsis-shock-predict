import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable  # 获取变量
from torch.utils import data  # 获取迭代数据
from torchvision.datasets import mnist  # 获取数据集

# 数据集的预处理
data_tf = torchvision.transforms.Compose(
	[
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.5], [0.5])
	]
)
data_path = r'.\MNIST_DATA_PyTorch'

# 获取数据集
# 如果本地没有数据集（第一次运行本代码），解除下面这一行的注释以下载数据集
# mnist.MNIST(data_path,train=True,transform=data_tf,download=True)
train_data = mnist.MNIST(data_path, train=True, transform=data_tf, download=False)
test_data = mnist.MNIST(data_path, train=False, transform=data_tf, download=False)

# 获取迭代数据
train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)


# 定义网络结构
# CNN网络结构	输入shape			卷积核		激活函数	输出图像
# conv1			[128,1,28,28]		[3,3,1,16]	ReLU		[128, 16, 14, 14]
# conv2			[128, 16, 14, 14]	[3,3,16,32]	ReLU		[128, 32, 7, 7]
# conv3			[128, 32, 7, 7]		[3,3,32,64]	ReLU		[128, 64, 4, 4]
# conv4			[128, 64, 4, 4]		[3,3,64,64]	ReLU		[128, 64, 2, 2]
class CNNnet(torch.nn.Module):
	def __init__(self):
		super(CNNnet, self).__init__()
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1,
							out_channels=16,
							kernel_size=3,
							stride=2,
							padding=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU()
		)
		self.conv2 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, 3, 2, 1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU()
		)
		self.conv3 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, 3, 2, 1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU()
		)
		self.conv4 = torch.nn.Sequential(
			torch.nn.Conv2d(64, 64, 2, 2, 0),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU()
		)
		self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
		self.mlp2 = torch.nn.Linear(100, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.mlp1(x.view(x.size(0), -1))
		x = self.mlp2(x)
		return x


# 实例化
model = CNNnet()
print(model)

# 定义损失和优化器
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练网络
loss_count = []
for epoch in range(2):
	for i, (x, y) in enumerate(train_loader):
		batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
		batch_y = Variable(y)  # torch.Size([128])
		# 获取最后输出
		out = model(batch_x)  # torch.Size([128,10])
		# 获取损失
		loss = loss_func(out, batch_y)
		# 使用优化器优化损失
		opt.zero_grad()  # 清空上一步残余更新参数值
		loss.backward()  # 误差反向传播，计算参数更新值
		opt.step()  # 将参数更新值施加到net的parmeters上
		if i % 20 == 0:
			loss_count.append(loss)
			print('{}:\t'.format(i), loss.item())
			torch.save(model, r'.\log_CNN')
		if i % 100 == 0:
			for a, b in test_loader:
				test_x = Variable(a)
				test_y = Variable(b)
				out = model(test_x)
				# print('test_out:\t',torch.max(out,1)[1])
				# print('test_y:\t',test_y)
				accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
				print('accuracy:\t', accuracy.mean())
				break
plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count, label='Loss')
plt.legend()
plt.show()

# 测试网络
model = torch.load(r'.\log_CNN')
accuracy_sum = []
for i, (test_x, test_y) in enumerate(test_loader):
	test_x = Variable(test_x)
	test_y = Variable(test_y)
	out = model(test_x)
	# print('test_out:\t',torch.max(out,1)[1])
	# print('test_y:\t',test_y)
	accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
	accuracy_sum.append(accuracy.mean())
	print('accuracy:\t', accuracy.mean())

print('总准确率：\t', sum(accuracy_sum) / len(accuracy_sum))
# 精确率图
print('总准确率：\t', sum(accuracy_sum) / len(accuracy_sum))
plt.figure('Accuracy')
plt.plot(accuracy_sum, 'o', label='accuracy')
plt.title('Pytorch_CNN_Accuracy')
plt.legend()
plt.show()
