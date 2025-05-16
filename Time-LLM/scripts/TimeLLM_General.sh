model_name=TimeLLM


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
            "--task_name" "long_term_forecast"
            "--is_training" "1"
            "--root_path" "./Dataset/$domain/"
            "--data_path" "${dataset}.csv"
            "--model" "$model_name"
            "--data" "custom"
            "--features" "M"
            "--freq" "$freq"
            "--factor" "3"
            "--enc_in" "$channel"
            "--dec_in" "$channel"
            "--c_out" "$channel"
            "--batch_size" "32"
            "--des" "Exp"
            "--llm_model" "BERT"
            "--llm_dim" "768"
            "--llm_layers" "12"
            "--train_epochs" "10"
            "--model_comment" "TimeLLM-$dataset"
        )

        if [[ "$seq_type" == "long" ]]; then
            ( IFS=$'\n'; echo "处理长序列: $domain - $dataset" )
            
            accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 12345 run_main.py "${common_params[@]}" \
                --model_id "${dataset}_512_96" \
                --seq_len 512 \
                --label_len 416 \
                --pred_len 96 

            accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 12345 run_main.py "${common_params[@]}" \
                --model_id "${dataset}_512_192" \
                --seq_len 512 \
                --label_len 320 \
                --pred_len 192

            accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 12345 run_main.py "${common_params[@]}" \
                --model_id "${dataset}_512_336" \
                --seq_len 512 \
                --label_len 176 \
                --pred_len 336

        elif [[ "$seq_type" == "short" ]]; then
            ( IFS=$'\n'; echo "处理短序列: $domain - $dataset" )
            
            accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 12345 run_main.py "${common_params[@]}" \
                --model_id "${dataset}_128_24" \
                --seq_len 128 \
                --label_len 104 \
                --pred_len 24 

            accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 12345 run_main.py "${common_params[@]}" \
                --model_id "${dataset}_128_48" \
                --seq_len 128 \
                --label_len 80 \
                --pred_len 48


            accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 12345 run_main.py "${common_params[@]}" \
                --model_id "${dataset}_128_60" \
                --seq_len 128 \
                --label_len 68 \
                --pred_len 60
        fi
    done
done
