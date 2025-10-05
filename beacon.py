#!/usr/bin/env python3
import json, socket, time, sys

PORT = 18830
NAME = "face-broker"
BROKER_PORT = 1883

def host_ip():
    """Return primary LAN IP (no deps)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # doesn't actually connect; just asks OS which src IP it'd use
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"[beacon] host_ip error: {e}", file=sys.stderr)
        return None

def main():
    print(f"[beacon] starting NAME={NAME} PORT={PORT} BROKER_PORT={BROKER_PORT}")
    sys.stdout.flush()

    # Listen for WHO_IS
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind(("", PORT))
    print(f"[beacon] listening on UDP 0.0.0.0:{PORT}")
    sys.stdout.flush()

    # Broadcast socket
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    last_adv = 0.0
    while True:
        # periodic global broadcast
        now = time.time()
        if now - last_adv > 2.0:
            ip = host_ip()
            if ip:
                msg = json.dumps({"name": NAME, "ip": ip, "port": BROKER_PORT}).encode()
                try:
                    tx.sendto(msg, ("255.255.255.255", PORT))
                    print(f"[beacon] sent GLOBAL -> 255.255.255.255:{PORT} {msg}")
                except Exception as e:
                    print(f"[beacon] broadcast error: {e}", file=sys.stderr)
            last_adv = now

        # reply to WHO_IS with unicast
        rx.settimeout(0.2)
        try:
            data, addr = rx.recvfrom(512)
            txt = data.decode(errors="ignore")
            # very permissive parse to avoid needing json lib failures
            if '"type"' in txt and '"WHO_IS"' in txt and '"name"' in txt and NAME in txt:
                ip = host_ip()
                if ip:
                    msg = json.dumps({"name": NAME, "ip": ip, "port": BROKER_PORT}).encode()
                    rx.sendto(msg, addr)
                    print(f"[beacon] reply -> {addr}: {msg}")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[beacon] rx error: {e}", file=sys.stderr)
            time.sleep(0.2)

if __name__ == "__main__":
    main()

