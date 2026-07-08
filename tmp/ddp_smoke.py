import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.multiprocessing.spawn import spawn

from cs336_systems.dist.ddp import DDP


def worker(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29612"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(rank)  # different init per rank, like the real test
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 5))

    ddp = DDP(deepcopy(model))

    torch.manual_seed(1234)
    all_x, all_y = torch.randn(20, 10), torch.randn(20, 5)

    baseline = deepcopy(model)
    for p_b, p_d in zip(baseline.parameters(), ddp.parameters()):
        p_b.data.copy_(p_d.data)  # sync baseline to post-broadcast weights

    opt_d = torch.optim.SGD(ddp.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(baseline.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    for step in range(3):
        opt_b.zero_grad()
        loss_fn(baseline(all_x), all_y).backward()
        opt_b.step()

        opt_d.zero_grad()
        lo = rank * 10
        loss_fn(ddp(all_x[lo : lo + 10]), all_y[lo : lo + 10]).backward()
        ddp.finish_gradient_synchronization()
        opt_d.step()

        if rank == 0:
            ok = all(
                torch.allclose(p_b, p_d, atol=1e-6)
                for p_b, p_d in zip(baseline.parameters(), ddp.parameters())
            )
            print(f"step {step}: {'params match baseline' if ok else 'MISMATCH'}")
            assert ok

    if rank == 0:
        print(f"handles after sync: {len(ddp.work_handles)} (expect 0)")

    dist.destroy_process_group()


if __name__ == "__main__":
    spawn(worker, args=(2,), nprocs=2, join=True)
