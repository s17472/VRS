import json
import logging

import seqlog
from config import CAM_ADDRESS, SEQ_ADDRESS

seqlog.log_to_seq(
    server_url=SEQ_ADDRESS,
    api_key="API Key",
    level=logging.INFO,
    batch_size=1,
    auto_flush_timeout=10,
    override_root_logger=True,
    json_encoder_class=json.encoder.JSONEncoder
)


def f(*predictions):
    weights = {'FGN': 0.4,
               'VRN': 0.6,
               'DIDN': 0.2}

    def norm(value):
        return 1 if value > 1 else value

    v = sum([p * weights[name] for name, p in predictions])
    return norm(v)


def get_probablity_string(prediction):
    if prediction < 0.7:
        return "Low"
    else:
        return "High"


def log(module_type, ts, source, prediction):
    logging.info(
        "{ModuleType}: Prediction: {Prediction}%, violence probability - {Probability}, timestamp: {Timestamp}",
        ModuleType=module_type, Prediction=round(prediction * 100, 2), Probability=get_probablity_string(prediction),
        Timestamp=str(ts), Source=source)


def log_final_result(ts, source, predictions):
    prediction = f(*predictions)
    logging.info(
        "{ModuleType}: Prediction: {Prediction}%, violence probability - {Probability}, timestamp: {Timestamp}",
        ModuleType='VRS', Prediction=round(prediction * 100, 2), Probability=get_probablity_string(prediction),
        Timestamp=str(ts), Source=source, Action="Detection")


def log_all(ts, source, *predictions):
    for name, p in predictions:
        log(name, ts, source, p)

    log_final_result(ts, source, predictions)


def log_msg(msg, *parameters):
    logging.info(msg, *parameters)


def log_start(source, fgn, vrn, didn):
    logging.info("Starting VRS on {Source}. Modules enabled: FGN={FGN}, VRN={VRN}, DIDN={DIDN}",
                 Source=source, FGN=fgn, VRN=vrn, DIDN=didn, Action="Start")


def log_stop():
    logging.warn("Stopping VRS on {Source}", Source=CAM_ADDRESS, Action="Stop")
