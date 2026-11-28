# Made on the NAS by ibkr-nas.sh when the registry does not answer: the nas-images release's base (the system and
# IB Gateway: downloaded only when it changed) and the bridge's files (bridge.jar and its two scripts). Only copies:
# nothing is compiled or fetched here. The same image as docker/ibkr/Dockerfile.ibkr's last stage.
ARG BASE
FROM $BASE
COPY bridge.jar /app/bridge.jar
COPY ibkr-start.sh /app/ibkr-start.sh
COPY run.sh /app/scripts/run.sh
ARG VERSION=unknown
ENV BRIDGE_VERSION=$VERSION
