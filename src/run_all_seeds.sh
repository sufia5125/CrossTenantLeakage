#!/bin/bash
set -e
rm -f ../results/main_multiseed.csv ../results/raw_queries.csv
for method in kmeans random; do
    for seed in 0 1 2 3 4; do
        echo "=== SEED $seed ($method) ==="
        python prep.py $seed $method
        python load_db.py
        python experiment.py $seed $method
    done
done
echo "All seeds and split methods complete."
