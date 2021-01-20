import cv2
import numpy as np
import config


def normalize(data):
    """
    Z-score normalization of data (how far from the mean a data point is)
    Args:
        data: data to normalize

    Returns:
        normalized data
    """
    mean = np.mean(data)
    # standard deviation
    std = np.std(data)
    return (data - mean) / std


def random_flip(video, prob: float):
    """
    Randomly flips the video
    Args:
        video: video data
        prob: probability of the flip

    Returns:
        flipped video data
    """
    s = np.random.rand()
    if s < prob:
        video = np.flip(m=video, axis=2)
    return video


def uniform_sampling(video, target_frames=config.FRAMES_NO):
    """
    Sampling FRAMES_NO frames uniformly from the entire video
    Args:
        video: video data
        target_frames: number of frames

    Returns:
        sampled video
    """
    # get total frames of input video and calculate sampling interval
    len_frames = int(len(video))
    interval = int(np.ceil(len_frames / target_frames))

    # init empty list for sampled video and
    sampled_video = []
    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])

    num_pad = target_frames - len(sampled_video)
    padding = []
    if num_pad > 0:
        for i in range(-num_pad, 0):
            try:
                padding.append(video[i])
            except:
                padding.append(video[0])
        sampled_video += padding

    return np.array(sampled_video, dtype=np.float32)


def color_jitter(video):
    """
    Transform color of the image
    Args:
        video: video data

    Returns:
        transformed video data
    """
    # range of s-component: 0-1
    # range of v component: 0-255
    s_jitter = np.random.uniform(-0.2, 0.2)
    v_jitter = np.random.uniform(-30, 30)
    for i in range(len(video)):
        hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
        s = hsv[..., 1] + s_jitter
        v = hsv[..., 2] + v_jitter
        s[s < 0] = 0
        s[s > 1] = 1
        v[v < 0] = 0
        v[v > 255] = 255
        hsv[..., 1] = s
        hsv[..., 2] = v
        video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return video


def get_optical_flow(video):
    """
    Gets optical flow from video
    Args:
        video: video data

    Returns:
        list of optical flows
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (config.SIZE, config.SIZE, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # add into list
        flows.append(flow)
    # padding the last frame as empty array
    flows.append(np.zeros((config.SIZE, config.SIZE, 2)))

    return np.array(flows, dtype=np.float32)


def normalize_respectively(data):
    """
    Normalize for each channel
    Args:
        data: data to normmalize

    Returns:
        normalized data
    """
    data[..., :3] = normalize(data[..., :3])
    data[..., 3:] = normalize(data[..., 3:])

    return data


def reshape(frame):
    """
    Reshape of the frame
    Args:
        frame: frame to reshape

    Returns:
        reshaped frame
    """
    frame = cv2.resize(frame, (config.SIZE, config.SIZE), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.reshape(frame, (config.SIZE, config.SIZE, 3))
    return frame


def set_optical_flow(frames, flows):
    """
    Marge rgb channel with optical flow
    Args:
        frames: rgb frames
        flows: optical flows

    Returns:
        list of rgb + optical flow
    """
    result = np.zeros((len(flows), config.SIZE, config.SIZE, 5))
    result[..., :3] = frames
    result[..., 3:] = flows

    return result


def video_2_npy(file_path):
    """
    Convert video to np array
    Args:
        file_path: path to the video

    Returns:
        transformed np array
    """
    # load video
    cap = cv2.VideoCapture(file_path)
    len_frames = int(cap.get(7))

    # extract frames from video
    try:
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            frame = reshape(frame)
            frames.append(frame)
    except:
        print("Error: ", file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()
    # get the optical flow of video
    flows = get_optical_flow(frames)
    result = set_optical_flow(frames, flows)

    return result
