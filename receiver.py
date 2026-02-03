import asyncio
import json
import signal
import sys
import pickle
import socket
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp
import shutil

import sensor

# =========================
# 設定
# =========================

# SSEのエンドポイント（例）
# 例: "http://<server>:<port>/stream"
SSE_URL = "http://163.143.136.103:5001/stream"

# 2台のデバイス識別（payload["dn"] or obj["dn"] を想定）
DN_LEFT = "3030F9284F54"
DN_RIGHT = "3030F92685D4"

# 保存先
exp_name = "./exp/0707/"

# 可視化/後段互換（以前のreceiver.pyと同じ）
LOCAL_IP = "127.0.0.1"
LOCAL_PORT_1 = 53000
LOCAL_PORT_2 = 53001

# --- ログ設定 ---
LOG_EVERY_N = 100          # N件に1回、サマリを表示（高頻度なら 50~200 推奨）
SHOW_RAW_ON_LOG = False   # True にするとログ行に生JSONの先頭を付ける（長いので注意）
RAW_HEAD_CHARS = 200      # 生JSONを表示する場合の先頭文字数

# dn の扱い：Trueなら payload["dn"] を優先（あなたのログで不整合があったため）
PREFER_PAYLOAD_DN = True

# =========================
# グローバル状態（既存構造に合わせる）
# =========================

data_list_l = []
data_list_r = []

# 「最新の生JSON（bytes）」を保持して、左右そろったらpickle送信する
_last_data_left: Optional[sensor.SensorData] = None
_last_data_right: Optional[sensor.SensorData] = None

# aiohttpタスク停止用
_stop_event = asyncio.Event()

# カウント類（ログ用）
_count_total = 0
_count_left = 0
_count_right = 0
_count_other = 0

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

def extract_dn(obj: Dict[str, Any]) -> Optional[str]:
    payload = obj.get("payload", {})
    if PREFER_PAYLOAD_DN and isinstance(payload, dict) and isinstance(payload.get("dn"), str):
        return payload.get("dn")
    if isinstance(obj.get("dn"), str):
        return obj.get("dn")
    if isinstance(payload, dict) and isinstance(payload.get("dn"), str):
        return payload.get("dn")
    return None


def log_update(dn: str, obj: Dict[str, Any], raw_str: str):
    """
    画面を埋めないために「1行上書き」するログ。
    - 通常ログは同一行を更新
    - エラー/警告は別途改行して残す（下で対応）
    """
    payload = obj.get("payload", {}) if isinstance(obj.get("payload"), dict) else {}
    sn = payload.get("sn")
    ts = payload.get("ts")
    p = payload.get("p") or []
    gyro = payload.get("gyro") or []
    acc = payload.get("acc") or []

    msg = (
        f"[update] dn={dn} sn={sn} ts={ts} "
        f"p_len={len(p)} gyro={gyro} acc={acc} "
        f"(L={_count_left} R={_count_right} other={_count_other} total={_count_total})"
    )

    # 端末幅に合わせて切り詰め（長文で折り返されるのを防ぐ）
    width = shutil.get_terminal_size((120, 20)).columns
    if len(msg) > width - 1:
        msg = msg[: max(0, width - 2)] + "…"

    # 行をクリアしてから上書き（前の行が長い場合にゴミが残らないようにする）
    clear = " " * (width - 1)
    sys.stdout.write("\r" + clear)
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


def save_on_exit():
    now = datetime.now()
    file_name_l = exp_name + str(now.strftime("%Y%m%d_%H%M%S") + "_left" + ".csv")
    file_name_r = exp_name + str(now.strftime("%Y%m%d_%H%M%S") + "_right" + ".csv")

    print(f"\nExiting gracefully. Sensor data saved to {file_name_l}, {fe_name_r}.")
    sensor.save_sensor_data_to_csv(data_list_l, file_name_l)
    sensor.save_sensor_data_to_csv(data_list_r, file_name_r)


def _install_signal_handlers():
    def _handler(*_):
        _stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


async def sse_consume(url: str, session: aiohttp.ClientSession):
    global _last_data_left, _last_data_right
    global _count_total, _count_left, _count_right, _count_other

    headers = {"Accept": "text/event-stream"}
    sock_vis = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

                    # 空行でイベント確定
                    if line == "":
                        if data_lines:
                            data_str = "\n".join(data_lines)

                            if event_name == "update":
                                try:
                                    obj = json.loads(data_str)
                                except json.JSONDecodeError as e:
                                    print(f"[warn] JSON decode error: {e} head={data_str[:120]!r}")
                                    event_name, data_lines = None, []
                                    continue

                                dn = extract_dn(obj)
                                if dn is None:
                                    print(f"[warn] dn not found. head={data_str[:120]!r}")
                                    event_name, data_lines = None, []
                                    continue

                                sd = sse_json_to_sensordata(obj)
                                if sd is None:
                                    print(f"[warn] invalid payload format. dn={dn} head={data_str[:120]!r}")
                                    event_name, data_lines = None, []
                                    continue

                                _count_total += 1
                                # ログ：間引き
                                if _count_total % LOG_EVERY_N == 0:
                                    log_update(dn, obj, data_str)

                                if dn == DN_LEFT:
                                    _last_data_left = sd
                                    _count_left += 1
                                elif dn == DN_RIGHT:
                                    _last_data_right = sd
                                    _count_right += 1
                                else:
                                    _count_other += 1
                                    # 2台以外が混ざる可能性があるなら、ここでcontinueでもOK
                                    continue

                                # 左右両方のデータが揃ったら、ペアとして送信し、リセットする
                                if _last_data_left is not None and _last_data_right is not None:
                                    # ペアができたので、これを送信・保存する
                                    left_to_send, right_to_send = _last_data_left, _last_data_right

                                    # 保存リストに追加
                                    data_list_l.append(left_to_send)
                                    data_list_r.append(right_to_send)

                                    data2 = pickle.dumps((left_to_send, right_to_send))
                                    sock_vis.sendto(data2, (LOCAL_IP, LOCAL_PORT_1))
                                    sock_vis.sendto(data2, (LOCAL_IP, LOCAL_PORT_2))

                                    # 送信後、ペアをリセットして次のペアを待つ
                                    _last_data_left = None
                                    _last_data_right = None

                        event_name, data_lines = None, []
                        continue

                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:"):].lstrip())

            backoff = 1.0

        except Exception as e:
            print(f"\n[error] stream error: {e} -> reconnect in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    sock_vis.close()


async def main():
    _install_signal_handlers()

    timeout = aiohttp.ClientTimeout(sock_connect=5, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print("Connecting to SSE stream...")
        task = asyncio.create_task(sse_consume(SSE_URL, session))

        await _stop_event.wait()

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
        save_on_exit()
        sys.exit(0)