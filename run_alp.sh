datasets="cora citeseer pubmed"
lams="0.2"
alps="0.5 1.0"
seeds="0 1 2 3 4 5 6 7 8 9"

for dataset in $datasets; do
  for lam in $lams; do
    for seed in $seeds; do
      for alp in $alps; do
        python main.py --dataset $dataset --lam $lam --seed $seed --device cuda:0 --alp $alp
      done
    done
  done
done
