"""
Check if a port is ready to be binded

python src/tools/check_port.py --port 10000

nc -vz localhost 10000
"""

import socket, time

import click


@click.command()
@click.option("--port", default=0, type=int)
def create_socket_connection(port: int = 0):
    # Ask the OS if the port is ready to be binded
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.listen(1)  # <-- start listening
        host, port = s.getsockname()
        print("listening on", host, port)
        while True:
            conn, addr = s.accept()  # waits for nc to connect
            with conn:
                print("connected by", addr)
                time.sleep(1)  # keep it open


if __name__ == "__main__":
    create_socket_connection()
