from model_class import Model 
import torch.device as device

#config file path
config_file = 'yolo5x.yaml'

#setting the device for torch and ONNX
device = device('cpu')

yolov5_xlarge_model = Model(config_file).to(device)
print('Loaded the model...')