import socket
import signal
import sys
import sensor
import pickle
from datetime import datetime

# left:1370 right: 1371 glove:1381
# Define port
PORT_L = 13250
PORT_R = 13251
data_list_l = []
data_list_r = []
exp_name = './exp/0707/'



# Define a signal handler to capture interrupt signals
def signal_handler(sig, frame):
    # Get the current time
    now = datetime.now()

    # Format the time
    file_name_l = exp_name + str(now.strftime("%Y%m%d_%H%M%S") + "_left" + ".csv")
    file_name_r = exp_name + str(now.strftime("%Y%m%d_%H%M%S") + "_right" + ".csv")

    print(f'\nExiting gracefully. Sensor data saved to {file_name_l}, {file_name_r}.')
    sensor.save_sensor_data_to_csv(data_list_l, file_name_l)
    sensor.save_sensor_data_to_csv(data_list_r, file_name_r)
    sys.exit(0)

# Configure the signal handler to listen for SIGINT (usually Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Function to receive broadcast data
def receive_broadcast():
    udp_ip = ""  # Listening broadcast address
    udp_port_l = PORT_L
    udp_port_r = PORT_R
    count = 0
    # Create UDP socket
    # Socket for receiving data
    sock_l = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_r = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind port and IP
    sock_l.bind((udp_ip, udp_port_l))
    sock_r.bind((udp_ip, udp_port_r))

    # Socket for visualization
    local_ip = "127.0.0.1"
    local_port = 53000
    local_port2 = 53001
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("Connecting to sensor...")
    while True:
        data_l, addr_l = sock_l.recvfrom(1024)  # Buffer size is 1024 bytes
        data_r, addr_r = sock_r.recvfrom(1024)  # Buffer size is 1024 bytes

        data2 = pickle.dumps((data_l, data_r))
        sock2.sendto(data2, (local_ip, local_port))
        sock2.sendto(data2, (local_ip, local_port2))
        
        data_list_l.append(sensor.parse_sensor_data(data_l))
        data_list_r.append(sensor.parse_sensor_data(data_r))
        count += 1
        if count % 100 == 0:
            print(f"\rReceived {count} packets from {addr_l}, {addr_r}, Press Ctrl+C to stop receiving.", end='')

if __name__ == "__main__":
    receive_broadcast()
