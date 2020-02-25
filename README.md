Code Contributors: Kaixin Liu

If you have any questions, feel free to contact me. My emails is: lkx17@mails.tsinhua.edu.cn.

Plase cite our paper if you choose to use our code. 

```
@article{
  title={Instance Explainable Multi-Instance Learning for ROI of Various Data},
  author={Xu Zhao, Zihao Wang, Yong Zhang and Chunxiao Xing},
  booktitle={DASFAA},
  year={2020},
}
```

# Single Source Personalized PageRank

## Tested Environment
- Ubuntu
- C++ 11
- GCC 4.8
- Boost
- cmake

## Compile
```sh
$ cmake .
$ make
```

## Parameters
```sh
./fbrw action_name --algo <algorithm> [options]
```
- action:
    - query
    - topk
    - generate-ss-query: generate queries file
    - gen-exact-topk: generate ground truth by power-iterations method
    - gen-exact-self: generate ground truth by backward propagation
- algo: which algorithm you prefer to run
    - montecarlo: Monte Carlo
    - dne: Dynamic neighborhood expansion
    - fora_bippr: fora and bippr
    - fbrw: FBRW
- options
    - --prefix \<prefix\>
    - --epsilon \<epsilon\>
    - --dataset \<dataset\>
    - --query_size \<queries count\>
    - --k \<top k\>
    - --exact_ppr_path \<directory to place generated ground truth\>
    - --result_dir \<directory to place results\>

## Data
The data for DBLP, Pokec, Livejournal, Twitter are not included here for size limitation reason. You can find them online.

## Generate queries
Generate query files for the graph data. Each line contains a node id.

```sh
$ ./fbrw generate-ss-query --prefix <data-folder> --dataset <graph-name> --query_size <query count>
```

- Example:

```sh
$ ./fbrw generate-ss-query --prefix ./data/ --dataset pokec --query_size 500
```

## Query
Process queries.

```sh
$ ./fbrw query --algo <algo-name> --prefix <data-folder> --dataset <graph-name> --result_dir <output-folder> --epsilon <relative error> --query_size <query count>
```

- Example:

```sh
$ ./fbrw query --algo fbrw --prefix ./data/ --dataset pokec --epsilon 0.5 --query_size 20
```

## Top-K
Process top-k queries.

```sh
$ ./fbrw topk --algo <algo-name> --prefix <data-folder> --dataset <graph-name> --result_dir <output-folder> --epsilon <relative error> --query_size <query count> --k <k>
```

- Example

```sh
$ ./fbrw topk --algo fbrw --prefix ./data/ --dataset pokec --epsilon 0.5 --query_size 20 --k 500
```


## Exact PPR (ground truth)
Construct ground truth for the generated queries.

```sh
$ ./fbrw gen-exact-topk --prefix <data-folder> --dataset <graph-name> --k <k> --query_size <query count> --exact_ppr_path <folder to save exact ppr>
$ ./fbrw gen-exact-self --prefix <data-folder> --dataset <graph-name> --k <k> --query_size <query count> --exact_ppr_path <folder to save exact ppr>
```

- Example

```sh
$ mkdir ./exact
$ ./fbrw gen-exact-topk --prefix ./data/ --dataset pokec --k 500 --query_size 100 --exact_ppr_path ./exact/
$ ./fbrw gen-exact-self --prefix ./data/ --dataset pokec --k 500 --query_size 100 --exact_ppr_path ./exact/
```

```

