# save as test_torchrun.py
"""
torchrun --standalone --nproc-per-node 8 src/tools/check_torchrun_nccl.py

torchrun --nproc-per-node 8 --master_addr "localhost" --master-port 10000 src/tools/check_torchrun_nccl.py
"""
import os, socket
from datetime import timedelta

import torch
import torch.distributed as dist


def log_env(prefix="env"):
    master_addr = os.getenv("MASTER_ADDR")
    master_port = os.getenv("MASTER_PORT")
    rdzv_endpoint = os.getenv("TORCHELASTIC_RENDEZVOUS_ENDPOINT")
    print(
        f"[{prefix}] MASTER_ADDR={master_addr} MASTER_PORT={master_port} "
        f"RDZV_ENDPOINT={rdzv_endpoint}"
    )


def init_dist(backend="nccl", device_id=0):
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", device_id),
    )


def main(backend="nccl"):
    # Helpful NCCL defaults for single-node debugging; comment out if not needed
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    # os.environ.setdefault("NCCL_IB_DISABLE", "1")  # single node: avoid IB path
    # If you really want to pin to loopback for bootstrap, uncomment:
    # os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")

    # Required: put each rank on its own GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    log_env("pre-init")
    init_dist(backend=backend, device_id=local_rank)

    # Recompute after init (for clarity in logs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    host = socket.gethostname()
    print(
        f"[rank {rank}/{world_size}] host={host} backend={backend} local_rank={local_rank} device={device}"
    )

    # Tiny all-reduce
    t = torch.tensor([rank], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f"[rank {rank}/{world_size}] reduced_sum={int(t)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    # keep CPU side deterministic/lightweight
    torch.set_num_threads(1)
    try:
        main(backend="nccl")
    except Exception as e:
        # Quick isolation: if NCCL/CUDA init fails, try CPU-only GLOO
        print(f"[warn] NCCL path failed with: {e!r}. Falling back to GLOO (CPU).")
        main(backend="gloo")
