from onnx_coreml import convert
import onnx


model = onnx.load('yolov5.onnx')
coreml_model = convert(model=model,image_input_names=["input"],
	image_output_names=["ouput"],
	minimum_ios_deployment_target="13")
coreml_model.save('yolov5x.mlmodel')
