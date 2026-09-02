"""Sweep concurrency degree on one GPU and collect per-degree kernel timings."""
import json, os, shutil, subprocess, sys, time

BASE = "/localhome/local-tianzheng/.claude/jobs/83bd07b4/tmp/conc"
REPO = "/localhome/local-tianzheng/KernelBench"
PY = REPO + "/.venv/bin/python"
GPU = os.environ.get("PROBE_GPU", "1")
DEGREES = [int(d) for d in os.environ.get("PROBE_DEGREES", "1,3,6,9,12,15").split(",")]
TRIALS = int(os.environ.get("PROBE_TRIALS", "100"))
BLOCKS = int(os.environ.get("PROBE_BLOCKS", "12"))

env = dict(os.environ)
env["CUDA_HOME"] = os.path.expanduser("~/opt/cuda-12.8")
env["PATH"] = env["CUDA_HOME"] + "/bin:" + REPO + "/.venv/bin:" + env.get("PATH", "")
env["CUDA_VISIBLE_DEVICES"] = GPU
env["TORCH_EXTENSIONS_DIR"] = BASE + "/ext"
env["KB_GPU_EVAL_LOCK"] = "0"           # probe must NOT interlock with the live arms' lock
os.makedirs(env["TORCH_EXTENSIONS_DIR"], exist_ok=True)


def smi():
    q = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    u, m = q.stdout.strip().split(",")
    return int(u), int(m)


def live_arms():
    r = subprocess.run("ps -eo pid= -o cmd= | grep evolve_kb_batch | grep -v grep | awk '{print $1}'",
                       shell=True, capture_output=True, text=True)
    n = 0
    for p in r.stdout.split():
        try:
            e = open("/proc/%s/environ" % p, "rb").read().decode("utf8", "replace")
        except Exception:
            continue
        if "CUDA_VISIBLE_DEVICES=%s\x00" % GPU in e:
            n += 1
    return n // 2          # each arm shows as 2 pids (parent + spawn helper)


print("[prebuild] compiling extensions once (shared TORCH_EXTENSIONS_DIR)", flush=True)
t0 = time.time()
r = subprocess.run([PY, BASE + "/worker.py", "--build-only", "--outdir", BASE, "--bdir", BASE],
                   env=env, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:]); print(r.stderr[-3000:]); sys.exit(1)
print("[prebuild] ok in %.1fs" % (time.time() - t0), flush=True)

results = {"gpu": GPU, "trials": TRIALS, "blocks": BLOCKS, "degrees": {}}
for deg in DEGREES:
    outdir = "%s/out_d%02d" % (BASE, deg)
    bdir = "%s/bar_d%02d" % (BASE, deg)
    for d in (outdir, bdir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    arms0 = live_arms()
    print("[deg %2d] launching %d workers (background arms on GPU%s: %d)" % (deg, deg, GPU, arms0), flush=True)
    t0 = time.time()
    procs = [subprocess.Popen([PY, BASE + "/tput_worker.py", "--degree", str(deg), "--slot", str(i),
                               "--outdir", outdir, "--bdir", bdir,
                               "--duration", os.environ.get("PROBE_DURATION","15")],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i in range(deg)]
    samples = []
    while any(p.poll() is None for p in procs):
        time.sleep(2.0)
        try:
            samples.append(smi())
        except Exception:
            pass
        if time.time() - t0 > 1800:
            for p in procs:
                p.kill()
            break
    wall = time.time() - t0
    for i, p in enumerate(procs):
        o, e = p.communicate()
        if p.returncode != 0:
            print("  slot %d rc=%d\n%s\n%s" % (i, p.returncode, o[-1500:], e[-1500:]), flush=True)
    ws = []
    for fn in sorted(os.listdir(outdir)):
        ws.append(json.load(open(os.path.join(outdir, fn))))
    util = [s[0] for s in samples] or [0]
    mem = [s[1] for s in samples] or [0]
    results["degrees"][str(deg)] = {
        "workers": ws, "wall_sec": round(wall, 1), "n_returned": len(ws),
        "bg_arms_gpu": arms0,
        "util_mean_pct": round(sum(util) / len(util), 1), "util_max_pct": max(util),
        "mem_max_mib": max(mem),
    }
    print("[deg %2d] done in %.0fs, %d/%d workers returned, GPU util mean %.0f%% max %d%%, mem max %d MiB"
          % (deg, wall, len(ws), deg, results["degrees"][str(deg)]["util_mean_pct"],
             max(util), max(mem)), flush=True)
    with open(BASE + "/results_tput.json", "w") as f:
        json.dump(results, f, indent=1)
    time.sleep(3)

print("[done] wrote " + BASE + "/results_tput.json", flush=True)
