FROM tensorflow/serving

ENV MODEL_NAME=vrn
ENV MODEL_BASE_NAME=vgg_base
COPY models.config /models
COPY model/vrn /models/${MODEL_NAME}
COPY model/vgg_base /models/${MODEL_BASE_NAME}
EXPOSE 8500