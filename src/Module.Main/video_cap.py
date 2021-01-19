frames_now = []

def get_frames_range(cap, count, reshape_func = None):
    frames = []
    while len(frames) < count:
        suc, frame = cap.read()

        if not suc:
            return False, None, None

        if reshape_func is not None:
            frame = reshape_func(frame)

        frames.append(frame)

    from datetime import datetime
    dt_object = datetime.now()

    return True, frames, dt_object


def get_frames_range2(cap, count):
    frames = []
    while len(frames) < count:
        suc, frame = cap.read()

        if not suc:
            return False, None, None

        frames.append(frame)

    from datetime import datetime
    dt_object = datetime.now()

    return True, frames, dt_object


def collect_frames(cap):
    while True:
        if len(frames_now) > 64:
            frames_now.pop(0)
        suc, frame = cap.read()
        if not suc:
            break

        frames_now.append(frame)


