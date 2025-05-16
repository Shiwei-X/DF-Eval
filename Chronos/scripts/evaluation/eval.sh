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
    for i in "${!datasets[@]}"; do
        dataset=${datasets[$i]}
        seq_type="${seq_lengths[idx]}"
        if [[ "$seq_type" == "long" ]]; then
            ( IFS=$'\n'; echo "处理长序列: $domain - $dataset" )
            for pl in 96 192 336; do
                yaml_file="evaluation/configs/${domain}/${dataset}-${pl}.yaml"
                # savepath_file="evaluation/results/chronos-t5-base-${dataset}-${pl}.csv"
                savepath_file="evaluation/results/chronos-t5-${dataset}-${pl}-small-1.csv"
                checkpoint_file="/home/xiongshiwei/chronos-forecasting/scripts/checkpoint/${dataset}_${pl}_small/checkpoint-final"
    
                python evaluation/eval.py $yaml_file $savepath_file \
                    --chronos-model-id $checkpoint_file \
                    --batch-size=16 \
                    --device=cuda:0 
            done
        elif [[ "$seq_type" == "short" ]]; then
            # 短序列配置
            ( IFS=$'\n'; echo "处理短序列: $domain - $dataset" )
            for pl in 24 48 60; do
                yaml_file="evaluation/configs/${domain}/${dataset}-${pl}.yaml"
                savepath_file="evaluation/results/chronos-t5-${dataset}-${pl}-small-1.csv"
                checkpoint_file="/home/xiongshiwei/chronos-forecasting/scripts/checkpoint/${dataset}_${pl}_small/checkpoint-final"
    
                python evaluation/eval.py $yaml_file $savepath_file \
                    --chronos-model-id $checkpoint_file \
                    --batch-size=16 \
                    --device=cuda:0
            done
        fi
    done
done
