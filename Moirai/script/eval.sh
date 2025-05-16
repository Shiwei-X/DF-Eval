export CUDA_VISIBLE_DEVICES=0



# ========== CAISO_AL (long, test_length=1757) ==========
for pl in 96 192 336; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Energy Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=CAISO_AL \
    model=CAISO_AL \
    data=CAISO_AL_test \
    data.windows=$(( 1757 - pl )) \
    data.prediction_length=$pl
done

# ========== NYISO_AL (long, test_length=3421) ==========
for pl in 96 192 336; do
  echo "eval NYISO_AL with prediction length $pl"
  python -m cli.eval \
    run_name=NYISO_AL \
    model=NYISO_AL \
    data=NYISO_AL_test \
    data.windows=$(( 3421 - pl )) \
    data.prediction_length=$pl
done

# ========== Wind (long, test_length=9735) ==========
for pl in 96 192 336; do
  echo "eval Wind with prediction length $pl"
  python -m cli.eval \
    run_name=Wind \
    model=Wind \
    data=Wind_test \
    data.windows=$(( 9735 - pl )) \
    data.prediction_length=$pl
done

# ========== Covid-19 (short, test_length=278) ==========
for pl in 24 48 60; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Healthcare Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=Covid-19 \
    model=Covid-19 \
    data=Covid-19_test \
    data.windows=$(( 278 - pl )) \
    data.prediction_length=$pl
done

# ========== illness (short, test_length=193) ==========
for pl in 24 48 60; do
  echo "eval illness with prediction length $pl"
  python -m cli.eval \
    run_name=illness \
    model=illness \
    data=illness_test \
    data.windows=$(( 193 - pl )) \
    data.prediction_length=$pl
done

# ========== weather (long, test_length=10539) ==========
for pl in 96 192 336; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Environment Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=weather \
    model=weather \
    data=weather_test \
    data.windows=$(( 10539 - pl )) \
    data.prediction_length=$pl
done

# ========== AQShunyi (long, test_length=7013) ==========
for pl in 96 192 336; do
  echo "eval AQShunyi with prediction length $pl"
  python -m cli.eval \
    run_name=AQShunyi \
    model=AQShunyi \
    data=AQShunyi_test \
    data.windows=$(( 7013 - pl )) \
    data.prediction_length=$pl
done

# ========== AQWan (long, test_length=7013) ==========
for pl in 96 192 336; do
  echo "eval AQWan with prediction length $pl"
  python -m cli.eval \
    run_name=AQWan \
    model=AQWan \
    data=AQWan_test \
    data.windows=$(( 7013 - pl )) \
    data.prediction_length=$pl
done

# ========== CzeLan (long, test_length=3987) ==========
for pl in 96 192 336; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Nature Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=CzeLan \
    model=CzeLan \
    data=CzeLan_test \
    data.windows=$(( 3987 - pl )) \
    data.prediction_length=$pl
done

# ========== ZafNoo (long, test_length=3845) ==========
for pl in 96 192 336; do
  echo "eval ZafNoo with prediction length $pl"
  python -m cli.eval \
    run_name=ZafNoo \
    model=ZafNoo \
    data=ZafNoo_test \
    data.windows=$(( 3845 - pl )) \
    data.prediction_length=$pl
done

# ========== M_DENSE (long, test_length=3504) ==========
for pl in 96 192 336; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Traffic Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=M_DENSE \
    model=M_DENSE \
    data=M_DENSE_test \
    data.windows=$(( 3504 - pl )) \
    data.prediction_length=$pl
done

# ========== METR_LA (long, test_length=6854) ==========
for pl in 96 192 336; do
  echo "eval METR_LA with prediction length $pl"
  python -m cli.eval \
    run_name=METR_LA \
    model=METR_LA \
    data=METR_LA_test \
    data.windows=$(( 6854 - pl )) \
    data.prediction_length=$pl
done

# ========== NYCTAXI_Inflow (short, test_length=437) ==========
for pl in 24 48 60; do
  echo "eval NYCTAXI_Inflow with prediction length $pl"
  python -m cli.eval \
    run_name=NYCTAXI_Inflow \
    model=NYCTAXI_Inflow \
    data=NYCTAXI_Inflow_test \
    data.windows=$(( 437 - pl )) \
    data.prediction_length=$pl
done

# ========== NYCTAXI_Outflow (short, test_length=437) ==========
for pl in 24 48 60; do
  echo "eval NYCTAXI_Outflow with prediction length $pl"
  python -m cli.eval \
    run_name=NYCTAXI_Outflow \
    model=NYCTAXI_Outflow \
    data=NYCTAXI_Outflow_test \
    data.windows=$(( 437 - pl )) \
    data.prediction_length=$pl
done

# ========== Exchange_rate (long, test_length=1518) ==========
for pl in 96 192 336; do
 if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Finance Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=Exchange_rate \
    model=Exchange_rate \
    data=Exchange_rate_test \
    data.windows=$(( 1518 - pl )) \
    data.prediction_length=$pl
done

# ========== IBM_m (short, test_length=152) ==========
for pl in 24 48 60; do
  echo "eval IBM_m with prediction length $pl"
  python -m cli.eval \
    run_name=IBM_m \
    model=IBM_m \
    data=IBM_m_test \
    data.windows=$(( 152 - pl )) \
    data.prediction_length=$pl
done

# ========== MSFT_w (short, test_length=407) ==========
for pl in 24 48 60; do
  echo "eval MSFT_w with prediction length $pl"
  python -m cli.eval \
    run_name=MSFT_w \
    model=MSFT_w \
    data=MSFT_w_test \
    data.windows=$(( 407 - pl )) \
    data.prediction_length=$pl
done

# ========== NASDAQ_w (short, test_length=155) ==========
for pl in 24 48 60; do
  echo "eval NASDAQ_w with prediction length $pl"
  python -m cli.eval \
    run_name=NASDAQ_w \
    model=NASDAQ_w \
    data=NASDAQ_w_test \
    data.windows=$(( 155 - pl )) \
    data.prediction_length=$pl
done

# ========== Wike2000 (short, test_length=158) ==========
for pl in 24 48 60; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval Web Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=Wike2000 \
    model=Wike2000 \
    data=Wike2000_test \
    data.windows=$(( 158 - pl )) \
    data.prediction_length=$pl
done

# ========== BizITObs-Application (long, test_length=1767) ==========
for pl in 96 192 336; do
  echo "eval BizITObs-Application with prediction length $pl"
  python -m cli.eval \
    run_name=BizITObs-Application \
    model=BizITObs-Application \
    data=BizITObs-Application_test \
    data.windows=$(( 1767 - pl )) \
    data.prediction_length=$pl
done

# ========== BizITObs-Service (long, test_length=1767) ==========
for pl in 96 192 336; do
  echo "eval BizITObs-Service with prediction length $pl"
  python -m cli.eval \
    run_name=BizITObs-Service \
    model=BizITObs-Service \
    data=BizITObs-Service_test \
    data.windows=$(( 1767 - pl )) \
    data.prediction_length=$pl
done

# ========== BizITObs-L2C (long, test_length=6394) ==========
for pl in 96 192 336; do
  echo "eval BizITObs-L2C with prediction length $pl"
  python -m cli.eval \
    run_name=BizITObs-L2C \
    model=BizITObs-L2C \
    data=BizITObs-L2C_test \
    data.windows=$(( 6394 - pl )) \
    data.prediction_length=$pl
done

# ========== Bitbrains-rnd100-cpuusage (short, test_length=373) ==========
for pl in 24 48 60; do
  if [[ $pl -eq 96 || $pl -eq 24 ]]; then
    echo "Start eval CloudOps Domain" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result200.csv
  fi
  python -m cli.eval \
    run_name=Bitbrains-rnd100-cpuusage \
    model=Bitbrains-rnd100-cpuusage \
    data=Bitbrains-rnd100-cpuusage_test \
    data.windows=$(( 373 - pl )) \
    data.prediction_length=$pl
done

# ========== Bitbrains-rnd100-memoryusage (short, test_length=373) ==========
for pl in 24 48 60; do
  echo "eval Bitbrains-rnd100-memoryusage with prediction length $pl"
  python -m cli.eval \
    run_name=Bitbrains-rnd100-memoryusage \
    model=Bitbrains-rnd100-memoryusage \
    data=Bitbrains-rnd100-memoryusage_test \
    data.windows=$(( 373 - pl )) \
    data.prediction_length=$pl
done













## IBM数据集异常
# ========== IBM_m (short, test_length=152) ==========
# for pl in 24 48 60; do
#   echo "eval IBM_m with prediction length $pl" | tee -a /home/xiongshiwei/uni2ts/cli/conf/eval/result.csv
#   python -m cli.eval \
#     run_name=IBM_m \
#     model=IBM_m \
#     data=IBM_m_test \
#     data.windows=$(( 152 - pl )) \
#     data.prediction_length=$pl
# done