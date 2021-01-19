from decouple import config

FGN_FRAME_COUNT = config('FGN_FRAME_COUNT', default=64, cast=int)
VRN_FRAME_COUNT = config('VRN_FRAME_COUNT', default=2, cast=int)
DIDN_FRAME_COUNT = config('DIDN_FRAME_COUNT', default=1, cast=int)
CAM_IP = config('CAM_IP', default='http://72.89.63.90:8083/mjpg/video.mjpg')

FGN_ENABLED = config('FGN_ENABLED', default=True, cast=bool)
VRN_ENABLED = config('VRN_ENABLED', default=True, cast=bool)
DIDN_ENABLED = config('DIDN_ENABLED', default=True, cast=bool)

SEQ_PATH = config('SEQ_PATH', default='http://seq:5341/')
FGN_PATH = config('FGN_PATH', default="vrs-module-fgn:8500")
VRN_PATH = config('VRN_PATH', default="vrs-module-vrn:8500")
DIDN_PATH = config('DIDN_PATH', default="vrs-module-didn:8500")
