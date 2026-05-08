import os
import signal
import sys
import time

import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "mosquitto")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_FILTER = os.getenv("MQTT_TOPIC_FILTER", "home/sensors/#,home/actuators/+/state")
QOS = int(os.getenv("MQTT_QOS", "0"))

RUNNING = True


def handle_signal(_signum, _frame):
    global RUNNING
    RUNNING = False


def on_connect(client, _userdata, _flags, reason_code, _properties=None):
    if reason_code != 0:
        print(f"[mqtt-retainer] connect failed: {reason_code}", flush=True)
        return

    # noLocal avoids receiving our own republished messages on MQTT v5 brokers.
    options = mqtt.SubscribeOptions(qos=QOS, noLocal=True)
    # Allow multiple comma-separated topic filters in the env var.
    filters = [t.strip() for t in TOPIC_FILTER.split(",") if t.strip()]
    for f in filters:
        client.subscribe(f, options=options)
    print(f"[mqtt-retainer] connected to {BROKER}:{PORT}, subscribed to {', '.join(filters)}", flush=True)


def on_message(client, _userdata, msg):
    # Skip broker metadata topics.
    if msg.topic.startswith("$SYS/"):
        return

    # If already retained, keep it as-is to avoid unnecessary duplicates.
    if msg.retain:
        return

    info = client.publish(msg.topic, payload=msg.payload, qos=msg.qos, retain=True)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"[mqtt-retainer] publish failed for topic {msg.topic}, rc={info.rc}", flush=True)


def main():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mqtt-retainer", protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.loop_start()

    while RUNNING:
        time.sleep(0.5)

    client.loop_stop()
    client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
