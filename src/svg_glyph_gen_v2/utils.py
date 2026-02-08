import inspect
import json
import logging
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Union


def load_jsonl(jsonl_path_or_dir, logger=None, decode_jsonl=True):
    jsonl_path_or_dir = Path(jsonl_path_or_dir)
    if jsonl_path_or_dir.is_file():
        return _load_jsonl(jsonl_path_or_dir, decode_jsonl)

    jsonl_files = sorted(jsonl_path_or_dir.glob("*.jsonl"))
    if logger:
        logger.info(f"Found {len(jsonl_files)} jsonl files from {jsonl_path_or_dir}.")
    data = []
    for jsonl_file in jsonl_files:
        data += _load_jsonl(jsonl_file, decode_jsonl)
    if logger:
        logger.info(f"Loaded {len(data)} records from {jsonl_path_or_dir}.")
    return data


def _load_jsonl(jsonl_path, decode_jsonl=True):
    with open(jsonl_path, "r") as f:
        if decode_jsonl:
            return [json.loads(line.strip()) for line in f]
        else:
            return [line.strip() for line in f]


def write_jsonl(
    data: list[Union[str, dict]],
    output_path,
    chunk_size: int = 5000,
    logger=None,
    encode_jsonl=True,
) -> None:
    """Save the merged data to JSONL format in chunks."""
    if logger is not None:
        logger.info(
            f"Saving {len(data)} records to: {output_path} (chunk size: {chunk_size})"
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get base filename without extension
    base_path = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1]

    # Calculate number of chunks needed
    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    if total_chunks == 1:
        # If only one chunk needed, save with original filename
        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                # NOTE: do not use json.dump() here, it will escapes all non-ASCII characters into \uXXXX
                if encode_jsonl:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    f.write(record + "\n")

        if logger is not None:
            logger.info(f"Successfully saved {len(data)} records to {output_path}")
    else:
        # Save in multiple chunks
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(data))
            chunk_data = data[start_idx:end_idx]

            # Create chunk filename
            chunk_filename = f"{base_path}_chunk_{chunk_idx + 1:03d}{extension}"

            with open(chunk_filename, "w", encoding="utf-8") as f:
                for record in chunk_data:
                    if encode_jsonl:
                        # NOTE: do not use json.dump() here, it will escapes all non-ASCII characters into \uXXXX
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        f.write(record + "\n")

            if logger is not None:
                logger.info(
                    f"Saved chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_data)} records to {chunk_filename}"
                )

        if logger is not None:
            logger.info(
                f"Successfully saved all {len(data)} records across {total_chunks} chunks"
            )


@contextmanager
def timer_display(logger=None):
    """Context manager for showing elapsed time"""
    stop_event = threading.Event()

    def show_time():
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            sys.stdout.write(f"\rElapsed: {elapsed:.1f}s")
            sys.stdout.flush()
            time.sleep(0.1)
        if logger is not None:
            logger.info(f"Elapsed: {time.time() - start_time:.1f}s")

    timer_thread = threading.Thread(target=show_time)
    timer_thread.start()

    try:
        yield
    finally:
        stop_event.set()
        timer_thread.join()
        print()  # New line


def setup_logger(output_dir, name=None):
    """Setup logger to write to both stdout and log file

    Args:
        output_dir: Directory where logs will be stored
        name: Logger name (defaults to caller's __name__)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = inspect.currentframe()
    error_cnt = 3
    while inspect.getfile(frame) == __file__:
        frame = frame.f_back
        error_cnt -= 1
        if error_cnt == 0:
            raise RuntimeError("Could not find caller's __name__")

    if name is None:
        name = frame.f_globals.get("__name__", __name__)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    script_file_name = Path(inspect.getfile(frame)).stem
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_file_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Setup up logger to: {log_file}")

    return logger


def check_dir_non_empty(output_dir, logger=None):
    output_dir = Path(output_dir)

    if output_dir.exists():
        # check if output_dir is empty not
        output_dir_glob = output_dir.glob("*")
        is_non_empty = False
        for i in output_dir_glob:
            # ignore the logs folder
            if i.name == "logs":
                continue
            is_non_empty = True
            break

        if is_non_empty:
            if logger is not None:
                logger.info(f"Output dir {output_dir} already exists, will be skipped.")
            return True

    if logger is not None:
        logger.info(f"Output dir {output_dir} does not exist, will be created.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return False


def prepare_output_dir_and_logger(
    *, output_dir, overwrite, output_log_dir=None, logger_name=None
):
    should_skip = False

    output_dir = Path(output_dir)
    if output_log_dir is None:
        output_log_dir = output_dir
    logger = setup_logger(output_log_dir, name=logger_name)

    if check_dir_non_empty(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
            # recreate output dir and its logger
            logger = setup_logger(output_dir, name=logger_name)
            logger.warning(
                f"Output dir {output_dir} is not empty. Overwriting with `--overwrite`."
            )
        else:
            logger.warning(f"Output dir {output_dir} is not empty. Skipping.")
            should_skip = True

    return should_skip, logger


def _count_lines(filename):
    count = 0
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            count += chunk.count(b"\n")
    return count


def count_lines(filename_or_dir):
    filename_or_dir = Path(filename_or_dir)
    if not filename_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {filename_or_dir}")

    if os.path.isfile(filename_or_dir):
        return _count_lines(filename_or_dir)
    else:
        return sum(_count_lines(f) for f in Path(filename_or_dir).glob("*.jsonl"))


def _load_jsonl_by_generator(jsonl_path, decode_jsonl=True):
    with open(jsonl_path, "r") as f:
        for line in f:
            yield json.loads(line.strip()) if decode_jsonl else line.strip()


def load_jsonl_by_generator(jsonl_path_or_dir, logger=None, decode_jsonl=True):
    jsonl_path_or_dir = Path(jsonl_path_or_dir)
    if not jsonl_path_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {jsonl_path_or_dir}")

    if jsonl_path_or_dir.is_file():
        yield from _load_jsonl(jsonl_path_or_dir, decode_jsonl)
        return

    jsonl_files = sorted(jsonl_path_or_dir.glob("*.jsonl"))
    if logger:
        logger.info(f"Found {len(jsonl_files)} jsonl files from {jsonl_path_or_dir}.")

    count = 0
    for jsonl_file in jsonl_files:
        if logger:
            logger.info(f"Loading {jsonl_file}...")
        for record in _load_jsonl_by_generator(jsonl_file, decode_jsonl):
            count += 1
            yield record

    if logger:
        logger.info(f"Loaded {count} records from {jsonl_path_or_dir}.")
