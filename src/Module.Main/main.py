import multiprocessing
import threading
import time
from datetime import datetime

import cv2
from config import (CAM_IP, DIDN_ENABLED, DIDN_FRAME_COUNT, FGN_ENABLED,
                    FGN_FRAME_COUNT, VRN_ENABLED, VRN_FRAME_COUNT)
from didn import get_prediction_didn, reshape_didn, transform_didn
from fgn import get_prediction_fgn, reshape_fgn, transform_fgn
from log import log_all, log_start, log_stop
from video_cap import available_frames, collect_frames
from vrn import predict_vrn, reshape_vrn, transform_vrn


def predict(frames, reshape, transform, prediction):
    reshaped_frames = reshape(frames)
    transformed_frames = transform(reshaped_frames)
    prediction = prediction(transformed_frames)
    return prediction


def main():
    cap = cv2.VideoCapture(CAM_IP)
    pool = multiprocessing.Pool()
    threading.Thread(target=collect_frames, args=(cap,)).start()
    log_start(CAM_IP, FGN_ENABLED, VRN_ENABLED, DIDN_ENABLED)

    while True:
        if len(available_frames) >= 64:
            ts = datetime.now()
            
            threads = dict()
            if FGN_ENABLED:
                threads["FGN"] = pool.apply_async(func=predict, args=(available_frames[-FGN_FRAME_COUNT:], reshape_fgn, transform_fgn, get_prediction_fgn,))
            if VRN_ENABLED:
                threads["VRN"] = pool.apply_async(func=predict, args=(available_frames[-VRN_FRAME_COUNT:], reshape_vrn, transform_vrn, predict_vrn,))
            if DIDN_ENABLED:
                threads["DIDN"] = pool.apply_async(func=predict, args=(available_frames[-DIDN_FRAME_COUNT:], reshape_didn, transform_didn, get_prediction_didn,))

            predictions = []
            for name, t in threads.items():
                t.wait()
                predictions.append((name, t.get()))

            log_all(ts, CAM_IP, *predictions)

        else:
            print(f'Frames collected: {len(available_frames)}')
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Exiting...")
        log_stop()
