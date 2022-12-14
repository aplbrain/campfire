FROM tensorflow/tensorflow:latest-gpu
#FROM python:3.8
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"

WORKDIR /root/campfire
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git python3.8-venv -yqq
RUN python3 -m venv /opt/venv

RUN pip3 install --upgrade pip
COPY ./requirements.txt /root/campfire
RUN . /opt/venv/bin/activate && pip3 install -r requirements.txt

RUN . /opt/venv/bin/activate && pip3 install --upgrade  "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN . /opt/venv/bin/activate && pip3 install git+https://github.com/google-research/sofima
RUN . /opt/venv/bin/activate && pip3 install git+https://github.com/aplbrain/neuvue-client.git
RUN . /opt/venv/bin/activate && pip3 install git+https://github.com/jhuapl-boss/intern.git

COPY . /root/campfire
COPY secrets/cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY secrets/credentials /root/.aws/credentials
COPY secrets/intern.cfg /root/.intern/intern.cfg
COPY secrets/neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg

# Test Environment variables (comment out for kubernetes image)
# ENV MODE agents
# ENV ARG2 nvq
# ENV ARG3 gpu
# ENV ARG4 agents_prod_v6
# ENV ARG5 Tip_detect_prod_v1

CMD . /opt/venv/bin/activate && python3 drive.py $MODE $ARG2 $ARG3 $ARG4 $ARG5
