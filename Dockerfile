FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /ML
COPY . /ML
VOLUME /ML

CMD ["sleep", "infinity"]