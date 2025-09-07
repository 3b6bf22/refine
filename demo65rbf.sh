modelseed_list=(402025 102025 202025 302025 3070 4080 5090 20250101)
M_list=(10 30 50 100 300 500 1000 3000 5000)
data_list=('adult' 'cifar10' 'protein' 'workloads')
epoch_list=(10 10 20 20)
bigm_list=(5000 5000 5000 1000)

for ((j=0; j<${#data_list[@]}; j++)); do
    data=${data_list[$j]}
    epoch=${epoch_list[$j]}
    bigm=${bigm_list[$j]}

    for M in "${M_list[@]}"; do
        start_time=$(date +%s)

        CUDA_VISIBLE_DEVICES=4 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 0))]} &
        CUDA_VISIBLE_DEVICES=5 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 1))]} &
        CUDA_VISIBLE_DEVICES=6 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 2))]} &
        CUDA_VISIBLE_DEVICES=7 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 3))]} &
        CUDA_VISIBLE_DEVICES=4 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 4))]} &
        CUDA_VISIBLE_DEVICES=5 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 5))]} &
        CUDA_VISIBLE_DEVICES=6 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 6))]} &
        CUDA_VISIBLE_DEVICES=7 python train.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $M --modelseed ${modelseed_list[$((4 * i + 7))]}

        wait

        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        echo "Data $data Modelseed $i N $N M $M completed in $elapsed_time seconds." >> demo65rbf.log
    done

    start_time=$(date +%s)

    CUDA_VISIBLE_DEVICES=4 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 0))]} &
    CUDA_VISIBLE_DEVICES=5 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 1))]} &
    CUDA_VISIBLE_DEVICES=6 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 2))]} &
    CUDA_VISIBLE_DEVICES=7 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 3))]} &
    CUDA_VISIBLE_DEVICES=4 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 4))]} &
    CUDA_VISIBLE_DEVICES=5 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 5))]} &
    CUDA_VISIBLE_DEVICES=6 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 6))]} &
    CUDA_VISIBLE_DEVICES=7 python ridge.py --model RFLAF --data $data --epochs $epoch --h 0.0625 --N 65 --M $bigm --modelseed ${modelseed_list[$((4 * i + 7))]} &

    wait

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Data $data ridge Modelseed $i N $N M $M completed in $elapsed_time seconds." >> demo65rbf.log

done