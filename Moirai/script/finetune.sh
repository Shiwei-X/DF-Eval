export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"


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

for domain in "${Domain[@]}"; do
  datasets=(${DomainDataset[$domain]})
  for (( idx=0; idx<${#datasets[@]}; idx++ )); do
    dataset="${datasets[idx]}"
    # python -m cli.train -cp conf/finetune run_name=$dataset model=moirai_1.0_R_small data=$dataset val_data=$dataset
    # python -m cli.train -cp conf/finetune run_name=$dataset model=moirai_1.0_R_base data=$dataset val_data=$dataset
    python -m cli.train -cp conf/finetune run_name=$dataset model=moirai_1.0_R_large data=$dataset val_data=$dataset
  done
done