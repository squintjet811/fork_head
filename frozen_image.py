
from keras import backend as K

K.set_learning_phase(0)

import tensorflow as tf
from tensorflow.python.framework import graph_io
from keras.models import load_model


model = load_model('M_VGG/C_T.h5')
print(model.outputs)
print(model.inputs)
print(model.input_names)

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

print("input names: ",input_names)
print("output names: ", output_names)
#pdb.set_trace()

def freeze_session(session, keep_var_names = None, output_names = None, clear_devices = True):
		from tensorflow.python.framework.graph_util import convert_variables_to_constants
		graph = session.graph
		with graph.as_default():
			freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
			output_names = output_names or []
			output_names += [v.op.name for v in tf.global_variables()]
			#graph -> graphDef protobuff

			input_graph_def = graph.as_graph_def()
			if clear_devices:
				for node in input_graph_def.node:
					node.device = ""
			frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)

			return frozen_graph


frozen_graph = freeze_session(K.get_session(), output_names = [out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text = False)




import tensorflow.contrib.tensorrt as trt
trt_graph = trt.create_inference_graph(
	input_graph_def = frozen_graph,
	outputs = output_names,
	max_batch_size = 2,
	max_workspace_size_bytes = 1 << 25,
	precision_mode = 'FP32',
	minimum_segment_size = 50
	)

all_nodes = len([1 for n in frozen_graph.node])
print("numer of all nodes in a frozen graph : ", all_nodes)

trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numer of trt engine nodes in TensorRt GRAPH: ", trt_engine_nodes)

all_nodes2 = len([1 for n in trt_graph.node])
print("numer of all nodes in TensorRT graph: ", all_nodes2)

graph_io.write_graph(trt_graph, "./model/", 
					"trt_graph.pb", as_text = False)

import tensorflow as tf




