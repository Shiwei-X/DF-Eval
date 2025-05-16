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

# 定义每个数据集的 offset
declare -A Offset=(
  ['CAISO_AL']=6148
  ['NYISO_AL']=11974
  ['Wind']=34071
  ['Covid-19']=974
  ['illness']=676
  ['weather']=36887
  ['AQShunyi']=24545
  ['AQWan']=24545
  ['CzeLan']=13954
  ['ZafNoo']=13458
  ['M_DENSE']=12264
  ['METR_LA']=23990
  ['NYCTAXI_Inflow']=1529
  ['NYCTAXI_Outflow']=1529
  ['Exchange_rate']=5312
  ['IBM_m']=531
  ['MSFT_w']=1424
  ['NASDAQ_w']=541
  ['Wike2000']=554
  ['BizITObs-Application']=6184
  ['BizITObs-Service']=6185
  ['BizITObs-L2C']=22378
  ['Bitbrains-rnd100-cpuusage']=1306
  ['Bitbrains-rnd100-memoryusage']=1306
)


for domain in "${Domain[@]}"; do
  datasets=(${DomainDataset[$domain]})
  for dataset in "${datasets[@]}"; do
    echo "Processing $dataset from $domain ..."
    python -m uni2ts.data.builder.simple $dataset dataset/Dataset/$domain/$dataset.csv \
      --dataset_type wide \
      --offset ${Offset[$dataset]}
  done
done
