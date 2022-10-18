# # FROM tensorflow/tensorflow:latest-gpu
FROM python:3.8
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"

ADD ./requirements.txt /root/campfire/requirements.txt
# ADD ./neuvue-client /root/campfire/neuvue-client
ADD ./intern /root/campfire/intern
# COPY neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg
# COPY credentials /root/.aws/credentials
COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY intern.cfg /root/.intern/intern.cfg

WORKDIR /root/campfire
RUN pip3 install --upgrade pip

RUN pip3 install -e ./intern
# RUN pip3 install -e ./neuvue-client

RUN pip3 install -r requirements.txt
COPY . /root/campfire

CMD . /opt/venv/bin/activate && python3 drive.py agents nvq cpu agents_prod_v6 Tip_detect_prod_v1
