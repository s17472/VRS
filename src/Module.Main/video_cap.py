available_frames = []


def collect_frames(cap):
    while True:
        if len(available_frames) > 64:
            available_frames.pop(0)
        suc, frame = cap.read()
        if not suc:
            break

        available_frames.append(frame)
