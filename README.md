# Participatory Budgeting Research 
Artem Ivaniuk, Rutgers EconCS Research Group
https://econcs.rutgers.edu/


To run all specified experiments:
`python3 run_requested_experiments.py all`

To run specific experiments, example:
`python3 run_requested_experiments.py 1a 1b 3a 3b 4 5`

To run counterexample:
`python3 -c "from counterexample_quality_dominance import run_counterexample_aggregate; run_counterexample_aggregate(m=8, k=4, n=2000, num_runs=100, seed_start=0, save_plot=True)"`