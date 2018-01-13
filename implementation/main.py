import torch

from models.ResNet18 import pretrained_res18


useGPU = False



# # # # # # # # # # # # # # # #
# Load Classifier Models
# # # # # # # # # # # # # # # #


print('Loading base classifier model and robust model')
model_base = pretrained_res18()
model_robust = pretrained_res18()



# # # # # # # # # # # # # # # #
# Initialize Attacker
# # # # # # # # # # # # # # # #



