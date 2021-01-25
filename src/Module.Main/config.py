from decouple import config

# Defaults are for development purposes.
FGN_ENABLED = config('FGN_ENABLED', default=True, cast=bool)
VRN_ENABLED = config('VRN_ENABLED', default=True, cast=bool)
DIDN_ENABLED = config('DIDN_ENABLED', default=True, cast=bool)

CAM_IP = config('CAM_IP', default='http://localhost:8080/mjpg/video.mjpg')
SEQ_PATH = config('SEQ_PATH', default='http://localhost:5341/')
FGN_PATH = config('FGN_PATH', default="localhost:8520")
VRN_PATH = config('VRN_PATH', default="localhost:8500")
DIDN_PATH = config('DIDN_PATH', default="localhost:8510")

FGN_FRAME_COUNT = config('FGN_FRAME_COUNT', default=64, cast=int)
VRN_FRAME_COUNT = config('VRN_FRAME_COUNT', default=2, cast=int)
DIDN_FRAME_COUNT = config('DIDN_FRAME_COUNT', default=1, cast=int)

