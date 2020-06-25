import onnx
import onnx.shape_inference

print('Loading the ONNX model...')
model = onnx.load("yolov5.onnx")
print('Done...')
output = [node.name for node in model.graph.output]
input_all =  [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input =  list(set(input_all)  - set(input_initializer))
print('printing the node names...')
print('Inputs: ', net_feed_input)
print('Outputs: ', output)


print('checking the model...')
onnx.checker.check_model(model)

#printing the human readable graph
print('print the readable graph...')
#print(onnx.helper.printable_graph(model.graph))