#!/bin/sh
# Runs one of the native tools (ofm_strategy, ofm_eod, ofm_scan) in the Alpine image ofm-tools (docker/native/Dockerfile)
# on the working directory, as the calling user, with the network of the host. Only the variables the tools read are
# passed in: EODData's key and base, the store's token and folder, the market, the mail settings and the engine's X_*.
#   ofm ofm_eod shares --exchange LSE ...
set -e
IMAGE="${OFM_IMAGE:-ofm-tools}"
set -- docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work \
  -e EODDATA_API_KEY -e EODDATA_BASE -e EOD_TOKEN -e EOD_STORE -e MARKET \
  -e SMTP_SERVER -e SMTP_PORT -e SMTP_USERNAME -e SMTP_PASSWORD -e MAIL_TO -e MAIL_FROM \
  $(env | sed -n 's/^\(X_[A-Z0-9_]*\)=.*/-e \1/p') "$IMAGE" "$@"
exec "$@"
