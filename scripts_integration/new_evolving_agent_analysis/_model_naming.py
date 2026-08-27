"""Model identity and run-name parsing, DERIVED rather than hardcoded.

Both conventions here belong to the launcher (`env/<HARDWARE>/launch_wave.sh`),
not to any particular model:

    run dir   runs_evolving/<model>/[median/]<run_name>_<stamp>
    run name  base_agent_<model with . and - replaced by _>_<tag>_itr30_GH200

Deriving them means a third or fourth model needs no edit here. The hardcoded
pairs this replaces silently MISLABELLED anything that was not gpt-oss or terra:
`"terra" if "terra" in name else "gpt-oss-120b"` scored a qwen arm as gpt-oss and
pooled it into the gpt-oss aligned intersection -- precisely the cross-model
contrast the surrounding comment calls invalid. A wrong label is worse than a
crash because the numbers still look plausible.
"""

from __future__ import annotations

import os
import re

_STAMP = r"(_\d{4}(_\d{2}){4})?$"


def model_slug(model: str) -> str:
    """`gpt-5.6-terra` -> `gpt_5_6_terra`, matching how RUN_PREFIX is built."""
    return re.sub(r"[.\-]", "_", model)


def model_from_root(root: str) -> str:
    """`runs_evolving/<model>/median` -> `<model>`.

    The path is authoritative: it is where the launcher was told to write, whereas
    the run name only carries a slug that several models could collide on.
    """
    parts = os.path.normpath(root).split(os.sep)
    if "runs_evolving" in parts:
        i = parts.index("runs_evolving")
        if i + 1 < len(parts):
            return parts[i + 1]
    return os.path.basename(os.path.normpath(root)) or "unknown"


def discover_roots(base: str = "runs_evolving",
                   skip: tuple[str, ...] = ("archived", "smoke")) -> list[str]:
    """Every `runs_evolving/<model>[/median]` that actually holds run dirs."""
    found: list[str] = []
    if not os.path.isdir(base):
        return found
    for model in sorted(os.listdir(base)):
        if model in skip:
            continue
        mdir = os.path.join(base, model)
        if not os.path.isdir(mdir):
            continue
        for cand in (os.path.join(mdir, "median"), mdir):
            if not os.path.isdir(cand):
                continue
            if any(re.search(r"_itr30_GH200", d) for d in os.listdir(cand)):
                found.append(cand)
                break
    return found


def arm_tag(run_name: str, model: str | None = None) -> str:
    """Strip run-name scaffolding, leaving the arm tag (`folding`, `l2`, ...).

    The bare truncation arm has no tag at all -- its name is `<prefix>_itr30_GH200`
    -- so it renders as `truncation` rather than an empty string.
    """
    n = re.sub(r"_itr30_GH200" + _STAMP, "", run_name)
    if model:
        pref = "base_agent_" + model_slug(model)
        if n == pref:
            return "truncation"
        if n.startswith(pref + "_"):
            return n[len(pref) + 1:] or "truncation"
    # Unknown model: drop only the fixed literal, never guess where the slug ends.
    n = re.sub(r"^base_agent_", "", n)
    return n or "truncation"


def arm_label(run_name: str, model: str | None = None) -> str:
    """`<model>/<tag>` -- unambiguous when several models appear in one table."""
    t = arm_tag(run_name, model)
    return f"{model}/{t}" if model else t
