# Participatory Budgeting Research 
Artem Ivaniuk, Rutgers EconCS Research Group
https://econcs.rutgers.edu/





Useful commands:

To run all specified experiments with normal utility:
`python3 run_requested_experiments.py all`

To run specific experiments with normal utility, example:
`python3 run_requested_experiments.py 1a 1b 1c 2 3a 3b 4 5a 5b 5c`

To run with cost-proportional utility, examples:
`python3 run_requested_experiments.py all --utility cost_proportional`
`python3 run_requested_experiments.py 1a 2 3a --utility cost_proportional`

Case 5 can be split across processes (`5a`, `5b`, `5c` — one alpha each) or run in one job as `5` (all three alphas).

To run counterexample:
`python3 -c "from counterexample_quality_dominance import run_counterexample_aggregate; run_counterexample_aggregate(m=8, k=4, n=2000, num_runs=100, seed_start=0, save_plot=True)"`

Parallel run commands (3 parallel), one process per case:
`printf "1a\n1b\n1c\n2\n3a\n3b\n4\n5\n" | xargs -n 1 -P 3 -I {} python3 run_requested_experiments.py {} --utility normal`
`printf "1a\n1b\n1c\n2\n3a\n3b\n4\n5\n" | xargs -n 1 -P 3 -I {} python3 run_requested_experiments.py {} --utility cost_proportional`
`printf "1a\n1b\n1c\n2\n3a\n3b\n4\n5\n" | xargs -n 1 -P 3 -I {} python3 run_requested_experiments.py {} --utility normal --save-agent-preferences`
`printf "1a\n1b\n1c\n2\n3a\n3b\n4\n5\n" | xargs -n 1 -P 3 -I {} python3 run_requested_experiments.py {} --utility cost_proportional --save-agent-preferences`