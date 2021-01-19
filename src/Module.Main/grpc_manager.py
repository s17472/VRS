from tensorboard.compat.proto import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import tensorflow as tf


def grpc_prep(host, input_name, model_name, img):
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3
    channel = grpc.insecure_channel(host, options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = model_name
    grpc_request.model_spec.signature_name = 'serving_default'
    grpc_request.inputs[input_name].CopyFrom(tf.make_tensor_proto(img, shape=img.shape, dtype=types_pb2.DT_FLOAT))
    return stub, grpc_request


def grpc_predict(stub, grpc_request, output_name):
    result = stub.Predict(grpc_request, 10)
    result = result.outputs[output_name].float_val
    return result
