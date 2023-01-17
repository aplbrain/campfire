FROM python:3.8
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"
COPY . /root/campfire
WORKDIR /root/campfire
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements_tips.txt

COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY credentials /root/.aws/credentials
RUN pip3 install -e ./neuvue-client
COPY neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg
CMD [ "python3", "drive.py", "agents", "-1", "nvq", "True", "Tip_detect_defects_v6"]
