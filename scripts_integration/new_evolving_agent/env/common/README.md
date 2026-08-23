# Shared launcher scripts

Byte-identical across every server folder in `env/`, so they live here once and
each `env/<HARDWARE>/` folder symlinks to them.

These five are safe to share precisely because **none of them sources
`hardware_env.sh`**. That file derives `$HARDWARE` from the *calling script's
parent directory name*, so a script that resolves hardware must physically live
in `env/<HARDWARE>/` -- moving it here would make it resolve `common` and fall
back to the wrong baseline.

The six that DO source `hardware_env.sh` therefore stay per-server:
`launch_arm_reps.sh`, `launch_merge_reps.sh`, `launch_nvcc_series.sh`,
`launch_run.sh`, `launch_wave.sh`, `resume_run.sh`.
