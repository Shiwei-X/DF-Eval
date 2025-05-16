#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model=GPT4TS

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
            "--root_path" "./Dataset/$domain/"
            "--data_path" "${dataset}.csv"
            "--data" "custom"
            "--batch_size" "256"
            "--enc_in" "$channel"
            "--c_out" "$channel"
            "--freq" "$freq"
            "--percent" "100"
            "--gpt_layer" "6" 
            "--model" "$model"
            "--itr" "1" 
        )

        if [[ "$seq_type" == "long" ]]; then
            ( IFS=$'\n'; echo "处理长序列: $domain - $dataset" )
            
            python main.py "${common_params[@]}" \
                --model_id $dataset'_'$model'_'$gpt_layer \
                --seq_len 512 \
                --label_len 416 \
                --pred_len 96 

            python main.py "${common_params[@]}" \
                --model_id $dataset'_'$model'_'$gpt_layer \
                --seq_len 512 \
                --label_len 320 \
                --pred_len 192

            python main.py "${common_params[@]}" \
                --model_id $dataset'_'$model'_'$gpt_layer \
                --seq_len 512 \
                --label_len 176 \
                --pred_len 336

        elif [[ "$seq_type" == "short" ]]; then
            ( IFS=$'\n'; echo "处理短序列: $domain - $dataset" )
            
            python main.py "${common_params[@]}" \
                --model_id $dataset'_'$model'_'$gpt_layer \
                --seq_len 128 \
                --label_len 104 \
                --pred_len 24 

            python main.py "${common_params[@]}" \
                --model_id $dataset'_'$model'_'$gpt_layer \
                --seq_len 128 \
                --label_len 80 \
                --pred_len 48


            python main.py "${common_params[@]}" \
                --model_id $dataset'_'$model'_'$gpt_layer \
                --seq_len 128 \
                --label_len 68 \
                --pred_len 60
        fi
    done
done



