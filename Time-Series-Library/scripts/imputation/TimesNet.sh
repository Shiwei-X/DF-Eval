#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name="TimesNet"

Domain=('Energy' 'Healthcare' 'Environment' 'Nature' 'Traffic' 'Finance' 'Web' 'CloudOps')

# 声明关联数组存储配置信息
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

# 主执行循环
for domain in "${Domain[@]}"; do
    # 转换字符串为数组
    datasets=(${DomainDataset[$domain]})
    seq_lengths=(${DomainSeqlen[$domain]})
    frequencies=(${DomainFreq[$domain]})
    channels=(${DomainChannel[$domain]})

    # 遍历数据集
    for (( idx=0; idx<${#datasets[@]}; idx++ )); do
        dataset="${datasets[idx]}"
        seq_type="${seq_lengths[idx]}"
        freq="${frequencies[idx]}"
        channel="${channels[idx]}"

        # 构造公共参数
        common_params=(
            "--task_name" "imputation"
            "--is_training" "1"
            "--root_path" "./dataset/$domain/"
            "--data_path" "${dataset}.csv"
            "--model" "$model_name"
            "--data" "custom"
            "--features" "M"
            "--seq_len" "192" 
            "--label_len" "0"
            "--pred_len" "0"
            "--e_layers" "2"
            "--d_layers" "1"
            "--factor" "3"
            "--enc_in" "$channel"
            "--dec_in" "$channel"
            "--c_out" "$channel"
            "--batch_size" "16"
            "--d_model" "64"
            "--d_ff" "64"
            "--des" "Exp"
            "--itr" "1"
            "--top_k" "3"
            "--learning_rate" "0.001"
        )

        for mask_rate in 0.125 0.25 0.375 0.5
        do
            python run.py "${common_params[@]}" \
                --model_id "${dataset}_mask_${mask_rate}" \
                --mask_rate $mask_rate 
        done
    done
done