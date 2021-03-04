import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

from sepsis_predict_comp_data.tcn import TCN
from sepsis_predict_comp_data.data_generator import data_generator
from innvestigator import InnvestigateModel


# ---------------- 基本参数 --------------------

dropout = 0.1
nhid = 80  # 每个隐层的参数量
levels = 3  # 时间卷积隐层数量
kernel_size = 5  # 卷积核尺寸

input_size = 40
output_size = 336

nan_alg = 1
seed = 6783
model_version = "1"

# model
torch.manual_seed(seed)
n_channels = [nhid] * levels

# ----------------------------------------------

train_set, valid_set, test_set = data_generator(nan_alg=1)
data = test_set[0:1].transpose(1, 2)[:, :-1]

model = TCN(input_size, output_size, n_channels, kernel_size, dropout=dropout)
model.train(False)
model.load_state_dict(torch.load("../sepsis_predict_comp_data/sepsis_predict_state_" + model_version + ".pt"))
inn_model = InnvestigateModel(model, lrp_exponent=2, method="e-rule", beta=.5)

prediction, relevance = inn_model.innvestigate(in_tensor=data)

plt.imshow(relevance[0][0])
plt.show()
