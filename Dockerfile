# FROM python:3-alpine
FROM python:3.9
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"

COPY . /root/campfire
COPY cave-secret.json /root/.cloudvolume/secrets/cave-secret.json
COPY credentials /root/.aws/credentials

WORKDIR /root/campfire
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD [ "python3", "drive.py", "sqs", "-1", "delete=True","return_skel=s3"]