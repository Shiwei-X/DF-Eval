import subprocess
import os
from datetime import datetime
from tabulate import tabulate  # 需要提前 pip install tabulate
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "TimeXer"

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


results = []  # 保存统计信息


def run_with_retries(cmd_template, model_name, domain, dataset, model_id, max_retries=5, init_batch_size=64):
    retries = 0
    batch_size = init_batch_size
    log_dir = f"./logs/{model_name}/{domain}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset}_{model_id}.log")

    with open(log_file, "w") as f:
        f.write(f"==== RUN LOG: {model_id} ====\nStart time: {datetime.now()}\n")

        while retries <= max_retries and batch_size >= 1:
            f.write(f"\n--- Try #{retries + 1} | Batch size: {batch_size} ---\n")
            print(f"尝试运行: {model_id}, batch_size={batch_size}（第 {retries + 1} 次）")

            cmd = cmd_template[:]
            for i in range(len(cmd)):
                if cmd[i] == "--batch_size":
                    cmd[i + 1] = str(batch_size)
                    break

            result = subprocess.run(cmd, capture_output=True, text=True)
            print("---------------------------------------------------------")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            f.write("Command:\n" + " ".join(cmd) + "\n")
            f.write("STDOUT:\n" + result.stdout + "\n")
            f.write("STDERR:\n" + result.stderr + "\n")

            if result.returncode == 0:
                f.write(f"✅ SUCCESS with batch_size={batch_size}\n")
                print(f"✅ 成功完成：{model_id}")
                results.append([domain, dataset, model_id, batch_size])
                return True
            elif "CUDA out of memory" in result.stderr:
                f.write("⚠️ CUDA Out of Memory detected\n")
                print("⚠️ 显存溢出，正在减小 batch_size 并重试")
                retries += 1
                batch_size = batch_size // 2
            else:
                f.write("❌ 非显存错误，停止运行\n")
                print(f"❌ 非显存错误，停止运行 {model_id}")
                break

        f.write(f"\n❌ 最终失败 | 最后使用的 batch_size: {batch_size}\nEnd time: {datetime.now()}\n")
        print(f"日志保存在: {log_file}")
        results.append([domain, dataset, model_id, "FAIL"])
    return False


# === 主执行 ===

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
            "--factor", "3",
            "--enc_in", channel,
            "--dec_in", channel,
            "--c_out", channel,
            "--batch_size", "64",
            "--des", "Exp",
            "--itr", "1",
        ]

        lengths = [(512, 416, 96), (512, 320, 192), (512, 176, 336)] if seq_type == "long" else [(128, 104, 24), (128, 80, 48), (128, 68, 60)]
        print(f"\n📌 处理{seq_type}序列: {domain} - {dataset}")

        for seq_len, label_len, pred_len in lengths:
            model_id = f"{dataset}_{seq_len}_{pred_len}"
            cmd = ["python", "-u", "run.py"] + common_params + [
                "--model_id", model_id,
                "--seq_len", str(seq_len),
                "--label_len", str(label_len),
                "--pred_len", str(pred_len),
            ]
            run_with_retries(cmd, model_name, domain, dataset, model_id)

# === 输出结果表格 ===
print("\n📊 最终 batch_size 成功统计：")
print(tabulate(results, headers=["Domain", "Dataset", "Model_ID", "Final_Batch_Size"], tablefmt="grid"))

# === 保存为 CSV ===
import csv
csv_path = f"./logs/{model_name}_batchsize_results.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Domain", "Dataset", "Model_ID", "Final_Batch_Size"])
    writer.writerows(results)

print(f"\n✅ 表格结果已保存为 CSV: {csv_path}")

