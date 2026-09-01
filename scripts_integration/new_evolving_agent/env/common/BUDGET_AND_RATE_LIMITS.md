# Checking the LLM budget and reading the rate-limit buckets

Written 2026-08-31 after the budget silently ran out mid-wave on 2026-08-30 and killed
24 arms. That failure was invisible until arms started dying, because nothing in this
repo ever looked at the budget. This is the missing check.

## TL;DR

```bash
python scripts_integration/new_evolving_agent/env/common/check_budget.py
# exit 0 = healthy, 1 = exhausted, 2 = the call itself failed
```

Run it **before every launch**. Launching into an exhausted budget is not a slow failure
— arms burn through problems in minutes producing nothing, which is exactly how the
2026-08-29/30 incidents destroyed 214 problems.

## How to read the budget

LiteLLM's management routes are **blocked for our virtual key**:

```
/key/info, /user/info, /spend/logs  ->  403
{"detail":"Virtual key is not allowed to call this route.
           Only allowed to call routes: ['llm_api_routes']"}
```

So there is no admin endpoint to query. But the budget rides on the **response headers of
any ordinary completion**, so a 1-token request is enough:

```
x-litellm-key-max-budget: 4500.0
x-litellm-key-spend:      3626.11
```

**The budget is KEY-LEVEL, not per-model.** The same numbers come back whichever model you
probe — one shared pool across gpt-oss, terra and qwen, and across every server using this
key. A wave on another host draws down the same balance.

### Probe with gpt-oss-120b — it is free

`check_budget.py` defaults to `gpt-oss-120b` deliberately. Model ids tell you the billing:

| alias | resolves to | costs money? |
|---|---|---|
| gpt-oss-120b | `nvidia/openai/gpt-oss-120b` | no (NVIDIA-hosted) |
| qwen3.6-27b | `nvidia/qwen/qwen3.6-27b` | no (NVIDIA-hosted) |
| **gpt-5.6-terra** | **`azure/openai/gpt-5.6-terra`** | **YES — Azure passthrough** |

Probing with terra would spend real money to ask how much money is left. Use gpt-oss.

### The headers are not always present

A probe can return `x-litellm-key-spend: 0.0` with **no max-budget header at all**.
Treat their absence as "retry", never as "zero spend". `check_budget.py` says so and
exits 2 rather than reporting a false healthy.

## What the rate-limit buckets mean

Every completion also returns two SEPARATE bucket families. They are per-model, and they
are not the budget — you can be rate-limited with plenty of budget left, and vice versa.

```
x-ratelimit-remaining-requests / -tokens                 <- GLOBAL bucket
x-ratelimit-remaining-priority_model-requests / -tokens  <- PRIORITY bucket
```

* **global** — the model's overall allowance across all callers on this endpoint.
* **priority** — the allowance for *our* priority tier on that model. This is the one that
  actually throttles us, and it is much tighter.

Each family has a **request** count and a **token** count, and either can bind
independently: you can have thousands of requests left but no tokens, or the reverse.

### Measured 2026-08-31 — the priority bucket is why qwen failed

| model | global req | priority req | priority tokens |
|---|---|---|---|
| gpt-oss-120b | 2984/3000 | 583/600 | 249,928/600,000 |
| gpt-5.6-terra | 14878/15000 | 2971/3000 | ~3.0M/3.0M |
| **qwen3.6-27b** | 537/600 | **56/120** | 1.45M/2.0M |

**qwen's priority request bucket is 120 against terra's 3000 — 25x tighter.** That is why
qwen threw `Priority-based rate limit exceeded` while its *global* bucket looked fine, and
why the errors clustered in bursts. It is a structural property of that model's tier, not
something that improves with time or with fewer arms.

`is_rate_limit_error()` matches these (429 + "rate limit"), so they are retried with
backoff — they cost throughput, not data.

## Estimating runway

Sample the spend twice and divide. Measured 2026-08-31 with 10 terra arms live
(7 on this host + 3 on another server, same key):

```
$1.24 spent in 101s  ->  $44.2/hour  ->  ~$4.4 per terra arm-hour
```

Only terra bills, so **burn scales with the terra arm count and ignores gpt-oss/qwen
entirely**. Runway = remaining / (4.4 x terra_arms). Adding terra arms shortens the runway
for the arms already running, because the pool is shared.

```bash
# two samples 100s apart
a=$(python .../check_budget.py | grep -oP 'spent \K[0-9.]+')
sleep 100
b=$(python .../check_budget.py | grep -oP 'spent \K[0-9.]+')
# burn/hour = (b-a)*36 ; runway_h = (4500-b) / burn
```

## The rule this incident produced

**Check the budget before launching, and re-check the rate periodically during a wave.**
A budget-exhausted call surfaces as `RateLimitError: Budget has been exceeded! Key=...`,
which `is_rate_limit_error()` retries 6x with backoff — correct behaviour, but retrying
cannot refill a budget. It only slows the failure, so the arm still burns its problems.
