"""Distributed training helpers (DDP, NCCL)."""

import inspect
import os

import torch
import torch.distributed as dist

_DEFAULT_DEVICE_ID = None


def init_distributed_mode(args):
    global _DEFAULT_DEVICE_ID
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    kwargs = {"backend": "nccl", "init_method": "env://"}
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        kwargs["device_id"] = args.local_rank
    dist.init_process_group(**kwargs)
    _DEFAULT_DEVICE_ID = args.local_rank
    dist.barrier()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        if _DEFAULT_DEVICE_ID is not None:
            dist.barrier(device_ids=[_DEFAULT_DEVICE_ID])
        else:
            dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
