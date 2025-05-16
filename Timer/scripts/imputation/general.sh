export CUDA_VISIBLE_DEVICES=1

model_name=Timer
ckpt_path=checkpoints/Timer_imputation_1.0.ckpt
d_model=256
d_ff=512
e_layers=4
patch_len=24

Domain=('Energy' 'Healthcare' 'Environment' 'Nature' 'Traffic' 'Finance' 'Web' 'CloudOps')

declare -A DomainDataset=(
    ['Energy']='CAISO_AL NYISO_AL Wind'
    ['Healthcare']='Covid-19 illness'
    ['Environment']='weather AQShunyi AQWan' 
    ['Nature']='CzeLan ZafNoo'
    ['Traffic']='M_DENSE METR_LA NYCTAXI_Inflow NYCTAXI_Outflow'
    ['Finance']='Exchange_rate IBM_m MSFT_w NASDAQ_w'
    ['Web']='Wike2000 BizITObs-Application BizITObs-Service BizITObs-L2C'
    ['CloudOps']='Bitbrains-rnd100-cpuusage Bitbrains-rnd100-memoryusage'

)

declare -A DomainSeqlen=(
    ['Energy']='long long long'
    ['Healthcare']='short short'
    ['Environment']='long long long'
    ['Nature']='long long'
    ['Traffic']='long long short short' 
    ['Finance']='long short short short'
    ['Web']='short long long long'
    ['CloudOps']='short short'
)

declare -A DomainFreq=(
    ['Energy']='h h 15min'
    ['Healthcare']='d 7d'
    ['Environment']='10min h h'
    ['Nature']='30min 30min'
    ['Traffic']='h 5min h h'
    ['Finance']='d m 7d 7d'
    ['WebCloudOps']='d 10s 10s 5min'
    ['CloudOps']='15min 15min'
)

declare -A DomainChannel=(
    ['Energy']='34 11 7'
    ['Healthcare']='753 7'
    ['Environment']='21 11 11'
    ['Nature']='11 11'
    ['Traffic']='30 207 263 263'
    ['Finance']='8 5 5 12'
    ['Web_CloudOps']='2000 4 64 7'
    ['CloudOps']='100 100'
)

for domain in "${Domain[@]}"; do
    datasets=(${DomainDataset[$domain]})
    seq_lengths=(${DomainSeqlen[$domain]})
    frequencies=(${DomainFreq[$domain]})
    channels=(${DomainChannel[$domain]})

    for (( idx=0; idx<${#datasets[@]}; idx++ )); do
        dataset="${datasets[idx]}"
        seq_type="${seq_lengths[idx]}"
        freq="${frequencies[idx]}"
        channel="${channels[idx]}"
        common_params=(
            "--task_name" "imputation"
            "--is_training" "1"
            "--ckpt_path" "$ckpt_path" 
            "--seed" "0"
            "--root_path" "./Dataset/$domain/"
            "--data_path" "${dataset}.csv"
            "--data" "custom"
            "--model" "$model_name"
            "--features" "M"
            "--freq" "$freq"
            "--e_layers" "4"
            "--factor" "3"
            "--seq_len" "192" 
            "--label_len" "0"
            "--pred_len" "0"
            "--patch_len" "$patch_len"
            "--train_test" "0"
            "--batch_size" "32" # 16
            "--d_model" "$d_model"
            "--d_ff" "$d_ff"
            "--des" "Exp"
            "--itr" "1"
            "--use_ims"
            "--learning_rate" "0.001"
        )

        # for subset_rand_ratio in 0.05 0.1 0.2 1
        for subset_rand_ratio in 0.05 0.1 0.2
        do
        for mask_rate in 0.125 0.25 0.375 0.5
        do
            python run.py "${common_params[@]}" \
                --model_id "${dataset}_sr_${subset_rand_ratio}_mask_${mask_rate}" \
                --mask_rate $mask_rate \
                --subset_rand_ratio $subset_rand_ratio 
        done
        done

    done
done
