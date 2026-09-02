"""Fixed-duration throughput probe: immune to barrier/straggler artifacts.

Each worker runs the kernel in a tight launch loop for DURATION seconds and
counts completed launches. Every BATCH launches it also takes one CUDA-event
timed sample, so fidelity is measured *under sustained* concurrency rather than
in a barrier-aligned block that ramps down.
"""
import argparse, json, os, statistics as st, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import build, barrier   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bdir", required=True)
    ap.add_argument("--duration", type=float, default=15.0)
    a = ap.parse_args()

    sys.path.insert(0, "/localhome/local-tianzheng/KernelBench/src")
    import torch
    mish, mm = build()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    n = 32 * 1024 * 1024
    x = torch.randn(n, device=dev, dtype=torch.float32)
    A = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    B = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    A2 = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    B2 = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    torch.cuda.synchronize(dev)

    cases = [("mish_32M", mish.mish_cuda, [x], 200),
             ("matmul_1024", mm.mm_cuda, [A, B], 100),
             ("matmul_2048", mm.mm_cuda, [A2, B2], 20)]

    out = {"degree": a.degree, "slot": a.slot, "duration": a.duration,
           "pid": os.getpid(), "cases": {}}

    for cname, fn, args, batch in cases:
        for _ in range(10):                       # warm up
            fn(*args)
        torch.cuda.synchronize(dev)
        if not barrier(a.bdir, "t_%s" % cname, a.slot, a.degree):
            out.setdefault("errors", []).append("barrier timeout " + cname)
            continue

        ev, launches = [], 0
        t0 = time.time()
        while time.time() - t0 < a.duration:
            for _ in range(batch):
                fn(*args)
            torch.cuda.synchronize(dev)
            launches += batch
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(dev)
            s.record(); fn(*args); e.record()
            torch.cuda.synchronize(dev)
            ev.append(s.elapsed_time(e)); launches += 1
        el = time.time() - t0
        evs = sorted(ev)
        out["cases"][cname] = {
            "launches": launches, "elapsed_sec": round(el, 3),
            "launches_per_sec": launches / el,
            "ev_n": len(ev),
            "ev_median_ms": st.median(ev), "ev_mean_ms": st.fmean(ev),
            "ev_min_ms": evs[0], "ev_max_ms": evs[-1],
            "ev_p90_ms": evs[int(0.9 * (len(evs) - 1))],
            "ev_p99_ms": evs[int(0.99 * (len(evs) - 1))],
        }

    with open(os.path.join(a.outdir, "w%02d.json" % a.slot), "w") as f:
        json.dump(out, f, indent=1)
    print("slot %d done" % a.slot)


if __name__ == "__main__":
    main()
