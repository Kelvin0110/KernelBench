#!/usr/bin/env python3
"""Report the LLM key's budget and rate-limit headroom from ONE cheap API call.

The inference endpoint is a LiteLLM proxy. Its management routes (/key/info,
/user/info, /spend/logs) are BLOCKED for our virtual key --
  {"detail":"Virtual key is not allowed to call this route.
             Only allowed to call routes: ['llm_api_routes']"}
-- so the only channel is the response headers on a normal completion. A 1-token
request is enough; the headers come back regardless of the body.

    python check_budget.py [model_alias]

Exit 0 = healthy, 1 = budget exhausted or key rejected, 2 = call failed.
"""
import json, os, sys, urllib.request, urllib.error

BASE = "https://inference-api.nvidia.com/v1"
ALIAS = {"gpt-oss-120b": "nvidia/openai/gpt-oss-120b",
         "gpt-5.6-terra": "azure/openai/gpt-5.6-terra",
         "qwen3.6-27b": "nvidia/qwen/qwen3.6-27b"}

def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "../../../../.env"))
    except Exception:
        pass
    key = os.environ.get("NVIDIA_INF_API_KEY", "")
    if not key:
        print("NVIDIA_INF_API_KEY not set"); return 2
    model = ALIAS.get(sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-120b",
                      sys.argv[1] if len(sys.argv) > 1 else "nvidia/openai/gpt-oss-120b")
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            h = {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        msg = e.read()[:300].decode(errors="replace")
        print(f"HTTP {e.code}: {msg}")
        return 1 if ("budget" in msg.lower() or e.code in (401, 403)) else 2
    except Exception as e:
        print(f"call failed: {type(e).__name__}: {e}"); return 2

    def num(k):
        try: return float(h[k])
        except Exception: return None

    mx, sp = num("x-litellm-key-max-budget"), num("x-litellm-key-spend")
    print(f"model: {model}   priority tier: {h.get('x-litellm-priority','?')}")
    if mx and sp is not None:
        left = mx - sp
        print(f"BUDGET  spent {sp:.2f} of {mx:.2f}  ->  {left:.2f} left ({100*left/mx:.1f}%)")
        if left <= 0:
            print("  EXHAUSTED -- do not launch; arms burn through problems producing nothing")
            return 1
        if left / mx < 0.10:
            print("  WARNING under 10% remaining")
    else:
        print("BUDGET  headers absent on this response (they are not always returned; retry)")

    # The priority_model bucket is the one that produced qwen's
    # "Priority-based rate limit exceeded" errors -- it is far tighter than the global.
    for label, pfx in (("global   ", "x-ratelimit"),
                       ("priority ", "x-ratelimit-priority_model")):
        rr, lr = num(f"{pfx}-remaining-requests"), num(f"{pfx}-limit-requests")
        rt, lt = num(f"{pfx}-remaining-tokens"), num(f"{pfx}-limit-tokens")
        if lr or lt:
            parts = []
            if lr: parts.append(f"req {rr:.0f}/{lr:.0f} ({100*rr/lr:.0f}%)")
            if lt: parts.append(f"tok {rt:.0f}/{lt:.0f} ({100*rt/lt:.0f}%)")
            print(f"RATE {label} " + "  ".join(parts))
    return 0

if __name__ == "__main__":
    sys.exit(main())
