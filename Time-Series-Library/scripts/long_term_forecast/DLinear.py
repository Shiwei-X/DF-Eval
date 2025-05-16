import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "DLinear"

# Domain definitions
domains = ['Energy', 'Healthcare', 'Environment', 'Nature', 'Traffic', 'Finance', 'Web', 'CloudOps']

domain_dataset = {
    'Energy': ['CAISO_AL', 'NYISO_AL', 'Wind'],
    'Healthcare': ['Covid-19', 'illness'],
    'Environment': ['weather', 'AQShunyi', 'AQWan'],
    'Nature': ['CzeLan', 'ZafNoo'],
    'Traffic': ['M_DENSE', 'METR_LA', 'NYCTAXI_Inflow', 'NYCTAXI_Outflow'],
    'Finance': ['Exchange_rate', 'IBM_m', 'MSFT_w', 'NASDAQ_w'],
    'Web': ['Wike2000', 'BizITObs-Application', 'BizITObs-Service', 'BizITObs-L2C', 'Bitbrains-rnd100-cpuusage', 'Bitbrains-rnd100-memoryusage'],
    'CloudOps': ['Bitbrains-rnd100-cpuusage', 'Bitbrains-rnd100-memoryusage']
}

domain_seqlen = {
    'Energy': ['long', 'long', 'long'],
    'Healthcare': ['short', 'short'],
    'Environment': ['long', 'long', 'long'],
    'Nature': ['long', 'long'],
    'Traffic': ['long', 'long', 'short', 'short'],
    'Finance': ['long', 'short', 'short', 'short'],
    'Web': ['short', 'long', 'long', 'long'],
    'CloudOps': ['short', 'short']
}

domain_freq = {
    'Energy': ['h', 'h', '15min'],
    'Healthcare': ['d', '7d'],
    'Environment': ['10min', 'h', 'h'],
    'Nature': ['30min', '30min'],
    'Traffic': ['h', '5min', 'h', 'h'],
    'Finance': ['d', 'm', '7d', '7d'],
    'Web': ['d', '10s', '10s', '5min'],
    'CloudOps': ['15min', '15min']
}

domain_channel = {
    'Energy': ['34', '11', '7'],
    'Healthcare': ['753', '7'],
    'Environment': ['21', '11', '11'],
    'Nature': ['11', '11'],
    'Traffic': ['30', '207', '263', '263'],
    'Finance': ['8', '5', '5', '12'],
    'Web': ['2000', '4', '64', '7'],
    'CloudOps': ['100', '100']
}

for domain in domains:
    datasets = domain_dataset[domain]
    seqlens = domain_seqlen[domain]
    freqs = domain_freq[domain]
    channels = domain_channel[domain]

    for idx, dataset in enumerate(datasets):
        seq_type = seqlens[idx]
        freq = freqs[idx]
        channel = channels[idx]

        common_params = [
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--root_path", f"./dataset/{domain}/",
            "--data_path", f"{dataset}.csv",
            "--model", model_name,
            "--data", "custom",
            "--features", "M",
            "--freq", freq,
            "--d_model", "256",
            "--d_ff", "512",
            "--e_layers", "2",
            "--d_layers", "1",
            "--factor", "3",
            "--enc_in", channel,
            "--dec_in", channel,
            "--c_out", channel,
            "--batch_size", "64",
            "--des", "Exp",
            "--itr", "1",
        ]

        if seq_type == "long":
            print(f"处理长序列: {domain} - {dataset}")

            for seq_len, label_len, pred_len in [(512, 416, 96), (512, 320, 192), (512, 176, 336)]:
                model_id = f"{dataset}_{seq_len}_{pred_len}"
                cmd = ["python", "-u", "run.py"] + common_params + [
                    "--model_id", model_id,
                    "--seq_len", str(seq_len),
                    "--label_len", str(label_len),
                    "--pred_len", str(pred_len),
                ]
                subprocess.run(cmd)

        elif seq_type == "short":
            print(f"处理短序列: {domain} - {dataset}")

            for seq_len, label_len, pred_len in [(128, 104, 24), (128, 80, 48), (128, 68, 60)]:
                model_id = f"{dataset}_{seq_len}_{pred_len}"
                cmd = ["python", "-u", "run.py"] + common_params + [
                    "--model_id", model_id,
                    "--seq_len", str(seq_len),
                    "--label_len", str(label_len),
                    "--pred_len", str(pred_len),
                ]
                subprocess.run(cmd)

