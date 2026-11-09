import time, torch
from torch.utils.flop_counter import FlopCounterMode
from data import MODS


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure(model, dims, device, n_lat=200):
    """Params, GFLOPs per sample (FlopCounterMode, batch=1, eval) and latency (ms/sample, batch=1)."""
    model.eval()
    x = model.example_input(device)
    av = torch.ones(1, len(MODS), device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, av)
    gflops = fc.get_total_flops() / 1e9
    with torch.no_grad():
        for _ in range(20):
            model(x, av)
        if device.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n_lat):
            model(x, av)
        if device.type == "cuda": torch.cuda.synchronize()
    return {"params": count_params(model), "gflops": gflops, "latency_ms": 1000 * (time.perf_counter() - t) / n_lat}
