import json
import logging
import seqlog

from config import SEQ_PATH, CAM_IP

seqlog.log_to_seq(
    server_url=SEQ_PATH,
    api_key="My API Key",
    level=logging.INFO,
    batch_size=1,
    auto_flush_timeout=10,  # seconds
    override_root_logger=True,
    json_encoder_class=json.encoder.JSONEncoder  # Optional; only specify this if you want to use a custom JSON encoder
)


def f(*preds):
    weights = {'FGN': 0.6,
               'VRN': 0.4,
               'DIDN': 0.2}
    def norm(value):
        return 1 if value > 1 else value

    v = sum([p * weights[name] for name, p in preds])
    return norm(v)


def get_probablity_string(prediction):
    if prediction < 0.5:
        return "Low"
    elif prediction < 0.7:
        return "Medium"
    elif prediction < 0.9:
        return "High"
    else:
        return "Very high"


def log(module_type, ts, source, prediction):
    logging.info("{ModuleType}: Prediction: {Prediction}%, violence probability - {Probability}, timestamp: {Timestamp}",
                 ModuleType=module_type, Prediction=round(prediction*100, 2), Probability=get_probablity_string(prediction),
                 Timestamp=str(ts), Source=source)


def log_papa(ts, source, predictions):
    prediction = f(*predictions)
    logging.info("{ModuleType}: Prediction: {Prediction}%, violence probability - {Probability}, timestamp: {Timestamp}",
                 ModuleType='VRS', Prediction=round(prediction*100, 2), Probability=get_probablity_string(prediction),
                 Timestamp=str(ts), Source=source, Action="Detection")

def log_all(ts, source, *predictions):
    for name, p in predictions:
        log(name, ts, source, p)

    log_papa(ts, source, predictions)

def log_msg(msg, *parametrs):
    logging.info(msg, *parametrs)

def log_start(source, fgn, vrn, didn):
    logging.info("Starting VRS on {Source}. Modules enabled: FGN={FGN}, VRN={VRN}, DIDN={DIDN}",
                 Source=source, FGN=fgn, VRN=vrn, DIDN=didn, Action="Start")

def log_stop():
    logging.warn("Stopping VRS on {Source}", Source=CAM_IP, Action="Stop")
