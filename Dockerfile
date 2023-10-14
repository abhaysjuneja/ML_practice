FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /ML
COPY . /ML
VOLUME /ML

RUN apt-get -y update
RUN apt-get -y install git

CMD ["sleep", "infinity"]