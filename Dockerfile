FROM node:16.10-alpine
LABEL maintainer "Justin Joyce <Justin.Joyce@jhuapl.edu>"

# RUN apk add --no-cache su-exec tini
# EXPOSE 80
# ENV NODE_CONFIG_DIR=/etc/neuvuequeue
# VOLUME [ "/etc/neuvuequeue" ]

COPY ./campfire ~/campfire

CMD [ "node", "/opt/neuvuequeue/build/bin/neuvuequeue.js" ]
ENTRYPOINT [ "/sbin/tini", "--" ]
CMD [ "node", "/opt/neuvuequeue/build/bin/neuvuequeue.js" ]