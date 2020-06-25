from model_class import Model 
import torch
import os

#config file path
config_file = 'models/yolov5x.yaml'

#setting the device for torch and ONNX
device = torch.device('cpu')

yolov5_xlarge_model = Model(config_file).to(device)
yolov5_xlarge_model.eval()
print('Loaded the model...')




### Converting the model to ONNX format...
print('converting the model to ONNX...')

yolov5_xlarge_model.model[-1].export = True

dummy_input = torch.randn(1, 3, 640, 640, device='cpu')
torch.onnx.export(yolov5_xlarge_model, dummy_input, 
	'onnx/yolov5.onnx',
	verbose=True,
	opset_version=11,
	input_names = ['input'],
	output_names = ['output'])

print('ONNX conversion done ....')

#creating the directory to save the weights
if not os.path.exists('weights'):
	os.makedirs('weights')

print('tracing the model...')
traced_yolo = torch.jit.trace(yolov5_xlarge_model, dummy_input)

traced_yolo.save("weights/yolo_jit.pt")
print('model saved in jit...')

