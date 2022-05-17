FROM python:3.9
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"

COPY . /root/campfire
COPY neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg
COPY credentials /root/.aws/credentials
COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json

WORKDIR /root/campfire
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/aplbrain/neuvue-client.git

CMD [ "python3", "drive.py", 1, "radius=(100,100,10)", "resolution=(8,8,40)", "unet_bound_mult=4", "ep=sqs", "save=sqs","device=cpu"]
