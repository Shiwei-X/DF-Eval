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

for domain in "${Domain[@]}"; do
    datasets=(${DomainDataset[$domain]})
    seq_lengths=(${DomainSeqlen[$domain]})
    for (( idx=0; idx<${#datasets[@]}; idx++ )); do
        dataset="${datasets[idx]}"
        seq_type="${seq_lengths[idx]}"
        # echo "shuju [$dataset]"
        if [[ "$seq_type" == "long" ]]; then
            ( IFS=$'\n'; echo "处理长序列: $domain - $dataset" )
            CUDA_VISIBLE_DEVICES=1 python training/train.py \
                --config training/configs/${domain}/${dataset}.yaml \
                --model-id amazon/chronos-t5-small \
                --context-length 512 \
                --prediction-length 96 \
                --output-dir "./checkpoint/" \
                --max-steps 1000 \
                --per-device-train-batch-size 8 

            CUDA_VISIBLE_DEVICES=1 python training/train.py \
                --config training/configs/${domain}/${dataset}.yaml \
                --model-id amazon/chronos-t5-small \
                --context-length 512 \
                --prediction-length 192 \
                --output-dir "./checkpoint/" \
                --max-steps 1000 \
                --per-device-train-batch-size 8
            
            CUDA_VISIBLE_DEVICES=1 python training/train.py \
                --config training/configs/${domain}/${dataset}.yaml \
                --model-id amazon/chronos-t5-small \
                --context-length 512 \
                --prediction-length 336 \
                --output-dir "./checkpoint/" \
                --max-steps 1000 \
                --per-device-train-batch-size 8

        elif [[ "$seq_type" == "short" ]]; then
            ( IFS=$'\n'; echo "处理短序列: $domain - $dataset" )
            CUDA_VISIBLE_DEVICES=1 python training/train.py \
                --config training/configs/${domain}/${dataset}.yaml \
                --model-id amazon/chronos-t5-small \
                --context-length 128 \
                --prediction-length 24 \
                --output-dir "./checkpoint/" \
                --max-steps 1000 \
                --per-device-train-batch-size 8 
            
            CUDA_VISIBLE_DEVICES=1 python training/train.py \
                --config training/configs/${domain}/${dataset}.yaml \
                --model-id amazon/chronos-t5-small \
                --context-length 128 \
                --prediction-length 48 \
                --output-dir "./checkpoint/" \
                --max-steps 1000 \
                --per-device-train-batch-size 8 

            CUDA_VISIBLE_DEVICES=1 python training/train.py \
                --config training/configs/${domain}/${dataset}.yaml \
                --model-id amazon/chronos-t5-small \
                --context-length 128 \
                --prediction-length 60 \
                --output-dir "./checkpoint/" \
                --max-steps 1000 \
                --per-device-train-batch-size 8 

        else
            echo "Unknown sequence type: $seq_type"
        fi

    done
done


    