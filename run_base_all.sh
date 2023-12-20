datasets="cora citeseer pubmed computers photo coauthorcs coauthorphysics"
lams="0.2"
seeds="0 1 2 3 4 5 6 7 8 9"
bases="gat gin sage"
for dataset in $datasets; do
  for lam in $lams; do
    for seed in $seeds; do
      for base in $bases; do
        python main.py --dataset $dataset --lam $lam --seed $seed --device cuda:0 --base_model $base
      done
    done
  done
done
