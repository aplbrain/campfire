FROM python:3.8
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"
COPY requirements_tips.txt /root/campfire/requirements_tips.txt
WORKDIR /root/campfire
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements_tips.txt

COPY . /root/campfire
COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY credentials /root/.aws/credentials
RUN pip3 install -e ./neuvue-client
COPY neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg

COPY soma_table.p /root/campfire/soma_table.p
COPY root_ids.p /root/campfire/root_ids.p

CMD [ "python3", "drive.py", "tips", "-1", "nvq", "True", "Tip_detect_defects_v7"]
