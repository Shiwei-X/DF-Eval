from forecasting import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seeds = [0, 100, 200]

pretrained_model = ["AutonLab/MOMENT-1-small", "AutonLab/MOMENT-1-base", "AutonLab/MOMENT-1-large"]

Domain = ['Energy', 'Healthcare', 'Environment', 'Nature', 'Traffic', 'Finance', 'Web', 'CloudOps']


DomainDataset = {
    'Energy': ['CAISO_AL', 'NYISO_AL', 'Wind'],
    'Healthcare': ['Covid-19', 'illness'],
    'Environment': ['weather', 'AQShunyi', 'AQWan'],
    'Nature': ['CzeLan', 'ZafNoo'],
    'Traffic': ['M_DENSE', 'METR_LA', 'NYCTAXI_Inflow', 'NYCTAXI_Outflow'],
    'Finance': ['Exchange_rate', 'IBM_m', 'MSFT_w', 'NASDAQ_w'],
    'Web': ['Wike2000', 'BizITObs-Application', 'BizITObs-Service', 'BizITObs-L2C', 'Bitbrains-rnd100-cpuusage', 'Bitbrains-rnd100-memoryusage'],
    'CloudOps': ['Bitbrains-rnd100-cpuusage', 'Bitbrains-rnd100-memoryusage']
}

DomainSeqlen = {
    'Energy': ['long', 'long', 'long'],
    'Healthcare': ['short', 'short'],
    'Environment': ['long', 'long', 'long'],
    'Nature': ['long', 'long'],
    'Traffic': ['long', 'long', 'short', 'short'],
    'Finance': ['long', 'short', 'short', 'short'],
    'Web': ['short', 'long', 'long', 'long'],
    'CloudOps': ['short', 'short']
}


for seed in seeds:
    output_file = f"forecasting_results_seed{seed}.txt"
    for model in pretrained_model:
        print(f"Finetuning on Pretrained model: {model}")
        for domain in Domain:
            datasets = DomainDataset[domain]
            seqlens = DomainSeqlen[domain]
            for dataset, seqlen in zip(datasets, seqlens):
                if seqlen == 'long':
                    for i in [96, 192, 336]:
                        run_forecasting_pipeline(model, f'Dataset/{domain}/{dataset}.csv', 512, i, seed, output_file)
                elif seqlen == 'short':
                    for i in [24, 48, 60]:
                        run_forecasting_pipeline(model, f'Dataset/{domain}/{dataset}.csv', 128, i, seed, output_file)      
