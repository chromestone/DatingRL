# Parallelization

These are my notes on parallelizing RLlib on my M1 MBP. After running a few tests I ended up with 6 runners and 16 envs per runner for DQN.

## envs_per_runner

```bash
# real	1m4.966s
# user	1m42.091s
# sys	0m9.893s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 6 --envs_per_runner 1
```

```bash
# real	0m23.801s
# user	0m25.593s
# sys	0m2.543s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 6 --envs_per_runner 4
```

```bash
# real	0m17.370s
# user	0m15.569s
# sys	0m1.517s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 6 --envs_per_runner 8
```

```bash
# real	0m15.228s
# user	0m11.410s
# sys	0m1.145s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 6 --envs_per_runner 16
```

```bash
# real	0m13.765s
# user	0m9.800s
# sys	0m0.923s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 6 --envs_per_runner 32
```

## num_runners

```bash
# real	0m28.295s
# user	0m21.571s
# sys	0m1.619s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 1 --envs_per_runner 16
```

```bash
# real	0m27.502s
# user	0m19.008s
# sys	0m1.465s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 2 --envs_per_runner 16
```

```bash
# real	0m17.808s
# user	0m13.160s
# sys	0m1.239s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 4 --envs_per_runner 16
```

```bash
# real	0m15.041s
# user	0m11.297s
# sys	0m1.132s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 6 --envs_per_runner 16
```

```bash
# real	0m14.344s
# user	0m10.838s
# sys	0m1.122s
time python3 train.py -a dqn -e running_rank --iters 10 --num_runners 8 --envs_per_runner 16
```
