# GPU 1 -- context-management axis + L2 promotion
#
#   bash scripts_integration/new_evolving_agent/env/launch_wave.sh 1 \
#        scripts_integration/new_evolving_agent/env/wave_gpu1.spec dry-run
#
# NOTE: 4 arms here vs 5 on GPU 0, by design decision. The two groups therefore
# sit under slightly different eval-lock contention, and this group has no
# same-GPU truncation control -- compare these arms against GPU 0's baseline
# only with that caveat in mind.
#
# tag          | context-mode         | extra flags
markov         | markov_report        |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
l2             | truncation           | --enable-l2
