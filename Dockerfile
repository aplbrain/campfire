FROM tensorflow/tensorflow:latest-gpu
#FROM python:3.8
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"


WORKDIR /root/campfire
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install python3.8-venv
RUN apt install -y git
RUN python3 -m venv /opt/venv

RUN pip3 install --upgrade pip
COPY ./requirements.txt /root/campfire
RUN . /opt/venv/bin/activate && pip3 install -r requirements.txt

RUN . /opt/venv/bin/activate && pip3 install --upgrade  "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN . /opt/venv/bin/activate && pip3 install git+https://github.com/google-research/sofima

COPY . /root/campfire
COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY credentials /root/.aws/credentials
COPY intern.cfg /root/.intern/intern.cfg
RUN . /opt/venv/bin/activate && pip3 install -e ./neuvue-client
RUN . /opt/venv/bin/activate && pip3 install -e ./intern
COPY neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg
RUN pip3 freeze

CMD . /opt/venv/bin/activate && python3 drive.py agents nvq gpu agents_prod_functional Tip_detect_prod_v2
