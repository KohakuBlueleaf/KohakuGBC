# The machine on pseudo engines

`scripts/bench/pseudo_machine.py`: the recorded PoC arms (`runs/traces/poc30`, 30 images each,
every job 4 times: 120 jobs) on replay engines with the recorded VLM and detector latencies
divided by 4. The floor is every VLM request the jobs ask, at its recorded latency, over the VLM
lane's capacity; efficiency is floor over wall time. VLM in flight is the lane's mean requests
in flight over its capacity while jobs are resident. Windows 11, Python 3.13, the native core.

| Configuration | Wall s | Floor s | Efficiency | VLM in flight | Dispatcher busy | Detect batch | Embed batch |
|---|---|---|---|---|---|---|---|
| text, VLM 32, waiting 32 | 35.25 | 33.03 | 0.937 | 0.933 | 0.154 | 1.39 | 4.31 |
| text, VLM 32, waiting 8 | 35.61 | 33.03 | 0.928 | 0.940 | 0.141 | 1.38 | 4.36 |
| text, VLM 32, waiting 128 | 35.64 | 33.03 | 0.927 | 0.935 | 0.138 | 1.41 | 4.41 |
| text, VLM 32, window 16 | 36.04 | 33.03 | 0.917 | 0.925 | 0.146 | 1.35 | 1.83 |
| text, VLM 64, waiting 64 | 19.34 | 16.52 | 0.854 | 0.845 | 0.269 | 2.31 | 4.52 |
| text, VLM 32, detector 1 slot | 35.62 | 33.03 | 0.927 | 0.939 | 0.143 | 1.94 | 4.66 |
| image query root | 48.53 | 44.12 | 0.909 | 0.909 | 0.134 | 1.27 | 4.60 |
| tags root | 32.42 | 30.35 | 0.936 | 0.907 | 0.153 | 1.72 | 4.59 |
| image and text root | 41.53 | 38.41 | 0.925 | 0.921 | 0.141 | 1.32 | 4.72 |
| entity batch 4 | 32.14 | 28.31 | 0.881 | 0.868 | 0.124 | 1.72 | 3.03 |
| leaf captions from the parent | 40.30 | 36.71 | 0.911 | 0.881 | 0.121 | 1.25 | 4.07 |

In every configuration all 120 graphs equal the PoC's and every recorded job (19 per
configuration) conforms. The VLM lane is the limit: its queue holds 36 to 155 requests on
average while about 60 images are resident, and the time below the floor is the run's start
and end, when fewer images have requests to send (the larger lane, with half the run time,
loses more to them). Admission's target and the window move the result by about 1%. The
dispatcher is busy 12 to 27% of the time with every latency divided by 4.
