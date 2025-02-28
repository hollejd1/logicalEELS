import os

import tensorflow as tf
from tensorflow import keras

import onnx
import tf2onnx
import numpy as np

# tf_model_file = './models/resNet50'
tf_model_file = '/workspace/temEdge-deploy/models/20240416/X_encoder'
onnx_model_file = './models/temp.onnx'
trt_model_file = './models/temp.trt'
BATCH_SIZE=16

cmd = 'python3 -m tf2onnx.convert --saved-model '+tf_model_file+' --output '+onnx_model_file
os.system(cmd)

onnx_model = onnx.load_model(onnx_model_file)

inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

onnx.save_model(onnx_model, onnx_model_file)

cmd='trtexec --onnx='+onnx_model_file+' --saveEngine='+trt_model_file
os.system(cmd)
