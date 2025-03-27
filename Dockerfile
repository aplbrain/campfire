FROM python:3.11
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"
COPY requirements_tips.txt /root/campfire/requirements_tips.txt
WORKDIR /root/campfire
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements_tips.txt

COPY . /root/campfire
RUN pip3 install -e ./neuvue-client
COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY cave_datastack_to_server_map.json /root/.cloudvolume/secrets/cave_datastack_to_server_map.json
COPY credentials /root/.aws/credentials
COPY neuvuequeue.cfg /root/.neuvuequeue/neuvuequeue.cfg

COPY soma_table.p /root/campfire/soma_table.p
#COPY root_ids.p /root/campfire/root_ids.p

CMD [ "python3", "drive.py", "tips", "5", "nvq", "True", "tip_detect_defects_h01_v1"]
