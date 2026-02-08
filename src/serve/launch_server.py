"""
We start a detached process to run sglang server, and then exit.
https://docs.sglang.ai/basic_usage/send_request.html

This is for VLLM only.
python scripts/tools/download_model_from_storage.py -i workspace/hf_downloads/Qwen/Qwen3-1.7B -o saves/Qwen3-1.7B

model_path="saves/Qwen3-1.7B"
VLLM_HOST_IP=localhost VLLM_LOOPBACK_IP=localhost \
    python src/serve/launch_server.py \
    "${model_path}" \
    --host "localhost" \
    --port 30000 \
    --data-parallel-address localhost \
    -dp 8 \
    -tp 1

pkill -f VLLM
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import psutil

import requests


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)


def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """

    kill_process_tree(process.pid)


def kill_process_tree(
    parent_pid, include_parent: bool = True, skip_pid: Optional[int] = None
):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# NOTE: solely use setLevel do not output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


DEFAULT_WAIT_FOR_SERVER_TIMEOUT = 600


def parse_port_from_argv(args, logger=None):
    port = None
    for idx, arg in enumerate(args):
        if arg.startswith("--port"):
            try:
                port = int(arg.split("=")[1])
            except Exception as e:
                if isinstance(e, (IndexError, ValueError)):
                    if logger:
                        logger.debug(f"Invalid try to parse port from {arg}")
                else:
                    raise e
            if idx == len(args) - 1:
                continue
            try:
                port = int(args[idx + 1])
            except ValueError:
                if logger:
                    logger.debug(f"Invalid try to parse port from {args[idx + 1]}")
    return port


def main():
    args = sys.argv[1:]
    logger.info(f"Server args: {args}")

    port = parse_port_from_argv(sys.argv[1:], logger=logger)
    if port is None:
        err_msg = "Port must be specified with `--port PORT`"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    command = f"vllm serve {' '.join(args)}"
    logger.info(f"Start server process:\n{command}")
    server_process = execute_shell_command(f"vllm serve {' '.join(args)}")

    logger.info(
        f"Start server process: {server_process.pid}, waiting for the response..."
    )
    server_url = f"http://localhost:{port}"
    try:
        timeout = DEFAULT_WAIT_FOR_SERVER_TIMEOUT
        logger.info(f"Waiting for server to become ready within {timeout} s")
        wait_for_server(server_url, timeout=DEFAULT_WAIT_FOR_SERVER_TIMEOUT)
    except TimeoutError:
        terminate_process(server_process)
        raise RuntimeError(
            f"Server did not become ready within timeout period: {DEFAULT_WAIT_FOR_SERVER_TIMEOUT} s"
        )
    logger.info(f"Server is ready! {server_url}")

    # After the server is up, we can kill the process
    # terminate_process(server_process)


# Copied from sglang/python/sglang/utils.py:wait_for_server
def wait_for_server(base_url, timeout=None, timer_print_interval=5) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    timer_print_cnt = 0
    while True:
        try:
            # NOTE: Fix the bug of timeout not working
            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")

            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time_lapse = time.perf_counter() - start_time
                print(f"Server is ready after {time_lapse:.0f} s: {base_url}")
                time.sleep(5)
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
            is_print = timer_print_cnt % timer_print_interval == 0
            timer_print_cnt = (timer_print_cnt + 1) % timer_print_interval
            time_lapse = time.perf_counter() - start_time
            if is_print:
                print(
                    f"Waiting for {time_lapse:.0f} s (timeout: {timeout:.0f} s).",
                    end="\r",
                    flush=True,
                )


if __name__ == "__main__":
    main()
