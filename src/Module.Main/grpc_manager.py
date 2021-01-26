"""
COde responsible for processing and handling connection with trained modules via gRPC.
- Benedykt Kościński
"""
import grpc
import tensorflow as tf
from tensorboard.compat.proto import types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


"""
Preparing gRPC request with serialized data for analysis.
"""
def grpc_prepare(host, input_name, model_name, img):
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3
    channel = grpc.insecure_channel(host, options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = model_name
    grpc_request.model_spec.signature_name = 'serving_default'
    grpc_request.inputs[input_name].CopyFrom(tf.make_tensor_proto(img, shape=img.shape, dtype=types_pb2.DT_FLOAT))
    return stub, grpc_request


"""
Send request with serialized data to module. 
"""
def grpc_predict(transformed_frames, address, input_name, output_name, model_name):
    stub, grpc_request = grpc_prepare(address, input_name, model_name, transformed_frames)
    result = stub.Predict(grpc_request, 10)
    return result.outputs[output_name].float_val
