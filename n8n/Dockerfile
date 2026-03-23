FROM alpine:3.22 AS mqtt-tools

RUN apk add --no-cache mosquitto-clients

FROM n8nio/n8n:latest

USER root

COPY --from=mqtt-tools /usr/bin/mosquitto_sub /usr/local/bin/mosquitto_sub
COPY --from=mqtt-tools /usr/lib/libmosquitto.so.1 /usr/lib/libmosquitto.so.1
COPY --from=mqtt-tools /usr/lib/libcjson.so.1 /usr/lib/libcjson.so.1
COPY --from=mqtt-tools /usr/lib/libssl.so.3 /usr/lib/libssl.so.3
COPY --from=mqtt-tools /usr/lib/libcrypto.so.3 /usr/lib/libcrypto.so.3
COPY --from=mqtt-tools /usr/lib/libcares.so.2 /usr/lib/libcares.so.2

USER node
