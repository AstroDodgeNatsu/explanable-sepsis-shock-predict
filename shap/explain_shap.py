import numpy as np
from matplotlib import pyplot as plt
import torch
import shap

# ===============================================================
# ===                                                         ===
# ===   !!! Before run this file, Add the code below to the   ===
# ===   end of file: [shap]/explainers/_deep/deep_pytorch.py  ===
# ===   [shap] means the root path of your python shap lib    ===
# ===   such like "D:\anaconda3\Lib\site-packages\shap"       ===
# ===                                                         ===
# ===   code for supplement :                                 ===
# ===       op_handler['Chomp1d'] = passthrough               ===
# ===                                                         ===
# ===============================================================

model = torch.load('sepsis_predict.pt').cuda()

data_raw = torch.load("dataTensor.pt").cuda()
data_X = data_raw[:, :, :-1].transpose(1, 2)
X_train = data_X[:100]
X_test = data_X[-2000:-1995]

e = shap.DeepExplainer(model, X_train)
shap_values = e.shap_values(X_test)

np.save('shap_values', shap_values[0, 0])
# shap.image_plot(shap_values, X_test)

plt.imshow(shap_values[0, 0])
plt.colorbar()
plt.show()
