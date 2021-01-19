frames_now = []


def collect_frames(cap):
    while True:
        if len(frames_now) > 64:
            frames_now.pop(0)
        suc, frame = cap.read()
        if not suc:
            break

        frames_now.append(frame)
