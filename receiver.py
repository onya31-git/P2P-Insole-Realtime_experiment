import asyncio
import json
import signal
import sys
import pickle
import socket
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp

import sensor

# =========================
# 設定
# =========================

# SSEのエンドポイント（例）
# 例: "http://<server>:<port>/stream"
SSE_URL = "http://163.143.136.103:5001/stream"

# 2台のデバイス識別（payload["dn"] or obj["dn"] を想定）
DN_LEFT = "B8F862C6FE30"
DN_RIGHT = "B78G98CO9GFFU"

# 保存先
exp_name = "./exp/0707/"

# 可視化/後段互換（以前のreceiver.pyと同じ）
LOCAL_IP = "127.0.0.1"
LOCAL_PORT_1 = 53000
LOCAL_PORT_2 = 53001

# =========================
# グローバル状態（既存構造に合わせる）
# =========================

data_list_l = []
data_list_r = []

# 「最新の生JSON（bytes）」を保持して、左右そろったらpickle送信する
_last_raw_left: Optional[bytes] = None
_last_raw_right: Optional[bytes] = None

# aiohttpタスク停止用
_stop_event = asyncio.Event()


# =========================
# ユーティリティ
# =========================

def sse_json_to_sensordata(obj: Dict[str, Any]) -> Optional[sensor.SensorData]:
    """
    SSEの data: {...} のJSONを、sensor.SensorData に変換する。
    あなたの例では、payloadの中に ts, p, mag, gyro, acc が入っている。

    - SensorData.timestamp     <- payload["ts"]
    - SensorData.pressure_sensors <- payload["p"]
    - SensorData.magnetometer  <- payload["mag"]
    - SensorData.gyroscope     <- payload["gyro"]
    - SensorData.accelerometer <- payload["acc"]
    """
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return None

    ts = payload.get("ts")
    p = payload.get("p")
    mag = payload.get("mag")
    gyro = payload.get("gyro")
    acc = payload.get("acc")

    if ts is None or p is None or mag is None or gyro is None or acc is None:
        return None

    # 既存のSensorDataに合わせて型を揃える
    try:
        timestamp = float(ts)
        pressure = [float(x) for x in p]
        magnetometer = (float(mag[0]), float(mag[1]), float(mag[2]))
        gyroscope = (float(gyro[0]), float(gyro[1]), float(gyro[2]))
        accelerometer = (float(acc[0]), float(acc[1]), float(acc[2]))
    except Exception:
        return None

    return sensor.SensorData(
        timestamp=timestamp,
        pressure_sensors=pressure,
        magnetometer=magnetometer,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
    )


def save_on_exit():
    now = datetime.now()
    file_name_l = exp_name + str(now.strftime("%Y%m%d_%H%M%S") + "_left" + ".csv")
    file_name_r = exp_name + str(now.strftime("%Y%m%d_%H%M%S") + "_right" + ".csv")

    print(f"\nExiting gracefully. Sensor data saved to {file_name_l}, {file_name_r}.")
    sensor.save_sensor_data_to_csv(data_list_l, file_name_l)
    sensor.save_sensor_data_to_csv(data_list_r, file_name_r)


def _install_signal_handlers():
    def _handler(*_):
        # asyncio側に停止を伝える
        _stop_event.set()

    # Unix系：SIGINT, SIGTERM
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# =========================
# SSE パーサ（aiohttp）
# =========================

async def sse_consume(url: str, session: aiohttp.ClientSession):
    """
    SSEを購読して、event:update の data JSON を順次処理する。
    """
    global _last_raw_left, _last_raw_right

    headers = {"Accept": "text/event-stream"}

    # 可視化用UDPソケット（以前と同じ）
    sock_vis = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    count = 0
    backoff = 1.0

    while not _stop_event.is_set():
        try:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()

                event_name: Optional[str] = None
                data_lines = []

                async for raw in resp.content:
                    if _stop_event.is_set():
                        break

                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

                    # 空行で1イベント確定
                    if line == "":
                        if data_lines:
                            data_str = "\n".join(data_lines)

                            if event_name == "update":
                                try :
                                    obj = json.loads(data_str)
                                except json.JSONDecodeError:
                                    event_name, data_lines = None, []
                                    continue

                                # dnは obj["dn"] でも payload["dn"] でも来るので両対応
                                dn = obj.get("dn")
                                payload = obj.get("payload", {})
                                if dn is None and isinstance(payload, dict):
                                    dn = payload.get("dn")

                                if not isinstance(dn, str):
                                    event_name, data_lines = None, []
                                    continue

                                sd = sse_json_to_sensordata(obj)
                                if sd is None:
                                    event_name, data_lines = None, []
                                    continue

                                # 既存の左右バッファへ
                                raw_bytes = data_str.encode("utf-8")

                                if dn == DN_LEFT:
                                    data_list_l.append(sd)
                                    _last_raw_left = raw_bytes
                                elif dn == DN_RIGHT:
                                    data_list_r.append(sd)
                                    _last_raw_right = raw_bytes
                                else:
                                    # 2台以外が混ざる可能性があるなら無視
                                    event_name, data_lines = None, []
                                    continue

                                # 左右が揃ったら、以前と同じpickle形式で送る
                                if _last_raw_left is not None and _last_raw_right is not None:
                                    data2 = pickle.dumps((_last_raw_left, _last_raw_right))
                                    sock_vis.sendto(data2, (LOCAL_IP, LOCAL_PORT_1))
                                    sock_vis.sendto(data2, (LOCAL_IP, LOCAL_PORT_2))

                                count += 1
                                if count % 200 == 0:
                                    print(
                                        f"\rReceived {count} updates "
                                        f"(L={len(data_list_l)} R={len(data_list_r)}) "
                                        f"Press Ctrl+C to stop.",
                                        end=""
                                    )

                        # 次イベントへ
                        event_name, data_lines = None, []
                        continue

                    # コメント行は無視
                    if line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:"):].lstrip())

            # 正常に抜けた（サーバが切断など）→再接続
            backoff = 1.0

        except Exception as e:
            # 切断や一時エラーは通常運用で起きる
            print(f"\nstream error: {e} -> reconnect in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    sock_vis.close()


# =========================
# エントリポイント
# =========================

async def main():
    _install_signal_handlers()

    timeout = aiohttp.ClientTimeout(sock_connect=5, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print("Connecting to SSE stream...")
        task = asyncio.create_task(sse_consume(SSE_URL, session))

        # Ctrl+C 等で _stop_event が立つのを待つ
        await _stop_event.wait()

        # 終了処理
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    save_on_exit()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 念のため（signalで拾えない環境対策）
        save_on_exit()
        sys.exit(0)
