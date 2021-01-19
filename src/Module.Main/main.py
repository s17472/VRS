import threading
import time

import cv2

from config import CAM_IP, FGN_FRAME_COUNT, DIDN_FRAME_COUNT, VRN_FRAME_COUNT, FGN_ENABLED, VRN_ENABLED, DIDN_ENABLED
from didn import transform_didn, get_prediction_didn, reshape_didn
from vrn import reshape_vrn, transform_vrn, get_prediction_vrn
from log import log_all, log_start, log_stop
from fgn import reshape_fgn, transform_fgn, get_prediction_fgn
from video_cap import get_frames_range, collect_frames, frames_now
import multiprocessing
import env_file
from tabulate import tabulate


def fgn(frames):
    reshaped_frames = reshape_fgn(frames)
    transformed_frames = transform_fgn(reshaped_frames)
    prediction = get_prediction_fgn(transformed_frames)
    return prediction


def vrn(frames):
    reshaped_frames_vrn = reshape_vrn(frames)
    transformed_frames_vrn = transform_vrn(reshaped_frames_vrn)
    prediction = get_prediction_vrn(transformed_frames_vrn)
    return prediction


def didn(frames):
    reshaped_frames_didn = reshape_didn(frames)
    transformed_frames_didn = transform_didn(reshaped_frames_didn)
    prediction = get_prediction_didn(transformed_frames_didn)
    return prediction


def print_conf():
    conf = env_file.get('.env.example')
    print("SYSTEM CONFIGURATION")
    print(tabulate(conf.items(), headers=['Name', 'Value'], tablefmt='orgtbl'))


def main():
    cap = cv2.VideoCapture(CAM_IP)
    pool = multiprocessing.Pool()
    threading.Thread(target=collect_frames, args=(cap,)).start()
    log_start(CAM_IP, FGN_ENABLED, VRN_ENABLED, DIDN_ENABLED)

    while True:
        if len(frames_now) >= 64:
            from datetime import datetime
            ts = datetime.now()
            if FGN_ENABLED:
                prediction_fgn = pool.apply_async(func=fgn, args=(frames_now[-FGN_FRAME_COUNT:],))
            if VRN_ENABLED:
                prediction_vrn = pool.apply_async(func=vrn, args=(frames_now[-VRN_FRAME_COUNT:],))
            if DIDN_ENABLED:
                prediction_didn = pool.apply_async(func=didn, args=(frames_now[-DIDN_FRAME_COUNT:],))

            predictions = []
            if FGN_ENABLED:
                prediction_fgn.wait()
                predictions.append(("FGN", prediction_fgn.get()))
            if VRN_ENABLED:
                prediction_vrn.wait()
                predictions.append(("VRN", prediction_vrn.get()))
            if DIDN_ENABLED:
                prediction_didn.wait()
                predictions.append(("DIDN", prediction_didn.get()))

            log_all(ts, CAM_IP, *predictions)

        else:
            print(len(frames_now))
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Exiting...")
        log_stop()