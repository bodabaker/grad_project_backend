import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import paho.mqtt.client as mqtt
import typer

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TIMEOUT_SECONDS = float(os.getenv("MQTT_TIMEOUT_SECONDS", "3"))

REGISTRY_PATH = Path(__file__).resolve().parent / "devices" / "registry.json"

app = typer.Typer(add_completion=False)
SOFT_FAIL = True


def _fail(message: str, **details: object) -> None:
    payload = {"ok": False, "error": message}
    payload.update(details)
    typer.echo(json.dumps(payload))
    raise typer.Exit(code=0 if SOFT_FAIL else 1)


def _available_devices(registry: Dict) -> list[str]:
    actuators = registry.get("actuators", {})
    sensors = registry.get("sensors", {})
    return sorted([*actuators.keys(), *sensors.keys()])


def _valid_actions_for_device(device: str, registry: Dict) -> list[str]:
    actuators = registry.get("actuators", {})
    sensors = registry.get("sensors", {})

    if device in sensors:
        return ["read"]
    if device in actuators:
        return list(actuators[device].get("actions", []))
    return []


def _load_registry() -> Dict:
    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        _fail(f"Registry file not found: {REGISTRY_PATH}")
    except json.JSONDecodeError as exc:
        _fail(f"Invalid registry.json: {exc}")
    return {}


def _make_client(client_id: str) -> mqtt.Client:
    return mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)


def _publish(topic: str, payload: str) -> None:
    client = _make_client("homectl_publish")
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        info = client.publish(topic, payload, qos=0, retain=False)
        info.wait_for_publish(timeout=TIMEOUT_SECONDS)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            _fail(f"Failed to publish to topic {topic}")
    except Exception as exc:
        _fail(f"MQTT publish error: {exc}")
    finally:
        client.loop_stop()
        client.disconnect()


def _read_single(topic: str) -> str:
    payload_event = threading.Event()
    data: Dict[str, str] = {}

    def on_message(_client: mqtt.Client, _userdata: Optional[dict], msg: mqtt.MQTTMessage) -> None:
        data["value"] = msg.payload.decode("utf-8", errors="replace").strip()
        payload_event.set()

    client = _make_client("homectl_read_single")
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe(topic)
        client.loop_start()
        if not payload_event.wait(timeout=TIMEOUT_SECONDS):
            _fail(f"Timed out waiting for sensor data on topic: {topic}")
        return data.get("value", "")
    except Exception as exc:
        _fail(f"MQTT read error: {exc}")
    finally:
        client.loop_stop()
        client.disconnect()


def _collect_topics(topic_to_name: Dict[str, str]) -> Dict[str, str]:
    payload_event = threading.Event()
    received: Dict[str, str] = {}
    total = len(topic_to_name)

    def on_message(_client: mqtt.Client, _userdata: Optional[dict], msg: mqtt.MQTTMessage) -> None:
        topic = msg.topic
        if topic in topic_to_name and topic not in received:
            received[topic] = msg.payload.decode("utf-8", errors="replace").strip()
            if len(received) >= total:
                payload_event.set()

    client = _make_client("homectl_list")
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        for topic in topic_to_name:
            client.subscribe(topic)
        client.loop_start()
        payload_event.wait(timeout=TIMEOUT_SECONDS)
    except Exception as exc:
        _fail(f"MQTT list error: {exc}")
    finally:
        client.loop_stop()
        client.disconnect()

    return received


def _format_list_value(device: str, raw_value: str) -> str:
    if device != "rgb_light":
        return raw_value

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value

    if isinstance(parsed, dict):
        # Keep grep-friendly one-line output while preserving RGB structured state.
        return json.dumps(parsed, separators=(",", ":"), sort_keys=True)

    return raw_value


def _map_actuator_payload(device: str, action: str) -> str:
    action = action.strip()
    normalized = action.lower()

    if device != "rgb_light":
        return action

    if normalized == "off":
        return "b 0"

    if normalized == "on":
        return "b 100"

    brightness_match = re.fullmatch(r"brightness\s+(\d{1,3})", action, flags=re.IGNORECASE)
    if brightness_match:
        value = int(brightness_match.group(1))
        if 0 <= value <= 100:
            return f"b {value}"
        _fail("rgb_light brightness must be between 0 and 100")

    color_match = re.fullmatch(r"color\s+(#[0-9a-fA-F]{6})", action, flags=re.IGNORECASE)
    if color_match:
        color = color_match.group(1).upper()
        return f"c {color}"

    _fail(
        "Invalid action for rgb_light",
        valid_actions=["on", "off", "brightness <0-100>", "color <#RRGGBB>"],
    )
    return ""


def _validate_action(device: str, action: str, registry: Dict) -> Tuple[str, bool]:
    actuators = registry.get("actuators", {})
    sensors = registry.get("sensors", {})

    if device in actuators:
        if device == "rgb_light":
            payload = _map_actuator_payload(device, action)
            return payload, False

        valid_actions = set(actuators[device].get("actions", []))
        normalized = action.strip().lower()
        if normalized not in valid_actions:
            _fail(
                f"Invalid action for {device}: {action}",
                valid_actions=_valid_actions_for_device(device, registry),
            )
        return normalized, False

    if device in sensors:
        if action.strip().lower() != "read":
            _fail(
                f"Invalid action for sensor {device}: {action}",
                valid_actions=_valid_actions_for_device(device, registry),
            )
        return "read", True

    _fail(f"Unknown device: {device}", available_devices=_available_devices(registry))
    return "", False


def _run_list(registry: Dict) -> None:
    sensors = registry.get("sensors", {})
    actuators = registry.get("actuators", {})

    sensor_order = list(sensors.keys())
    actuator_order = list(actuators.keys())

    topic_to_name: Dict[str, str] = {}
    for name in sensor_order:
        topic_to_name[sensors[name]["topic"]] = name
    for name in actuator_order:
        topic_to_name[actuators[name]["state_topic"]] = name

    values_by_topic = _collect_topics(topic_to_name)

    for name in sensor_order:
        topic = sensors[name]["topic"]
        typer.echo(f"{name}={values_by_topic.get(topic, '')}")

    for name in actuator_order:
        topic = actuators[name]["state_topic"]
        value = values_by_topic.get(topic, "")
        typer.echo(f"{name}={_format_list_value(name, value)}")


@app.callback(invoke_without_command=True)
def main(
    device: Optional[str] = typer.Option(None, "--device", help="Device name"),
    action: Optional[str] = typer.Option(None, "--action", help="Action to execute"),
    list_: bool = typer.Option(False, "--list", help="List all sensor and actuator states"),
    soft_fail: bool = typer.Option(
        True,
        "--soft-fail/--strict",
        help="Default soft-fail mode returns structured error JSON with exit code 0; use --strict for exit code 1 on errors",
    ),
) -> None:
    global SOFT_FAIL
    SOFT_FAIL = soft_fail

    registry = _load_registry()

    if list_:
        if device or action:
            _fail("Use either --list or --device/--action, not both")
        _run_list(registry)
        return

    if not device or not action:
        _fail("Usage: homectl --device <device_name> --action <action> OR homectl --list")

    payload_or_action, is_sensor_read = _validate_action(device, action, registry)

    if is_sensor_read:
        topic = registry["sensors"][device]["topic"]
        value = _read_single(topic)
        typer.echo(json.dumps({"ok": True, "device": device, "action": "read", "value": value}))
        return

    topic = registry["actuators"][device]["command_topic"]
    _publish(topic, payload_or_action)
    typer.echo(json.dumps({"ok": True, "device": device, "action": action.strip()}))


if __name__ == "__main__":
    app()
