import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

from mnist_test import Net, train, test
from innvestigator import InnvestigateModel

kwargs = {}
batch_size = 20
test_batch_size = 1

train_loader = DataLoader(
	datasets.MNIST('./data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
	datasets.MNIST('./data', train=False, transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])),
	batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net()
model.load_state_dict(torch.load("./mymodel"))
inn_model = InnvestigateModel(model, lrp_exponent=2,
							  method="e-rule",
							  beta=.5)

heatmap = 0
for data, target in test_loader:
	model_prediction, heatmap = inn_model.innvestigate(in_tensor=data)
	break
plt.imshow(heatmap[0][0])
plt.show()
