model_name=Timer

patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt


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
            "--task_name" "forecast"
            "--is_training" "0"
            "--ckpt_path" "$ckpt_path" 
            "--seed" "200"
            "--root_path" "./Dataset/$domain/"
            "--data_path" "${dataset}.csv"
            "--data" "custom"
            "--model" "$model_name"
            "--features" "M"
            "--freq" "$freq"
            "--e_layers" "8"
            "--factor" "3"
            "--des" "Exp"
            "--d_model" "1024"
            "--d_ff" "2048"
            "--batch_size" "256"
            "--learning_rate" "3e-5"
            "--num_workers" "4"
            "--patch_len" "$patch_len"
            "--train_test" "0"
            "--itr" "1"
            "--gpu" "0"
            "--use_ims"
        )


        if [[ "$seq_type" == "long" ]]; then
            ( IFS=$'\n'; echo "处理长序列: $domain - $dataset" )
            
            for subset_rand_ratio in 0.05 0.1 0.2 1
            do
                python run.py "${common_params[@]}" \
                    --model_id "${dataset}_512_96_sr_${subset_rand_ratio}" \
                    --seq_len 512 \
                    --label_len 384 \
                    --pred_len 96 \
                    --output_len 96 \
                    --subset_rand_ratio $subset_rand_ratio
            done

            for subset_rand_ratio in 0.05 0.1 0.2 1
            do
                python run.py "${common_params[@]}" \
                    --model_id "${dataset}_512_192_sr_${subset_rand_ratio}" \
                    --seq_len 512 \
                    --label_len 288 \
                    --pred_len 192 \
                    --output_len 192 \
                    --subset_rand_ratio $subset_rand_ratio
            done

            for subset_rand_ratio in 0.05 0.1 0.2 1
            do
                python run.py "${common_params[@]}" \
                    --model_id "${dataset}_512_336_sr_${subset_rand_ratio}" \
                    --seq_len 512 \
                    --label_len 144 \
                    --pred_len 336 \
                    --output_len 336 \
                    --subset_rand_ratio $subset_rand_ratio
            done

        elif [[ "$seq_type" == "short" ]]; then
            ( IFS=$'\n'; echo "处理短序列: $domain - $dataset" )
            
            for subset_rand_ratio in 0.05 0.1 0.2 1
            do
                python run.py "${common_params[@]}" \
                    --model_id "${dataset}_128_24_sr_${subset_rand_ratio}" \
                    --seq_len 128 \
                    --label_len 72 \
                    --pred_len 24 \
                    --output_len 24 \
                    --subset_rand_ratio $subset_rand_ratio
            done

            for subset_rand_ratio in 0.05 0.1 0.2 1
            do
                python run.py "${common_params[@]}" \
                    --model_id "${dataset}_128_48_sr_${subset_rand_ratio}" \
                    --seq_len 128 \
                    --label_len 48 \
                    --pred_len 48 \
                    --output_len 48 \
                    --subset_rand_ratio $subset_rand_ratio
            done

            for subset_rand_ratio in 0.05 0.1 0.2 1
            do
                python run.py "${common_params[@]}" \
                    --model_id "${dataset}_128_60_sr_${subset_rand_ratio}" \
                    --seq_len 128 \
                    --label_len 36 \
                    --pred_len 60 \
                    --output_len 60 \
                    --subset_rand_ratio $subset_rand_ratio
            done
        
        fi
    done
done
