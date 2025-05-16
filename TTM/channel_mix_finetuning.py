import math
import os
import tempfile

import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions
from sklearn.metrics import mean_absolute_error, mean_squared_error


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
OUT_DIR = "/home/xiongshiwei/TTM/ttm_mix_finetuned_models"

def fewshot_finetune_eval(
    dataset_name,
    batch_size,
    fewshot_percent,
    context_length=512,
    forecast_length=96,
    num_epochs=10,
    save_dir=OUT_DIR,
    loss="mse",
    quantile=0.5,
):
    out_dir = os.path.join(save_dir, dataset_name)
    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    dset_train, dset_val, dset_test = get_datasets(
        tsp,
        data,
        fewshot_fraction=fewshot_percent / 100,
        fewshot_location="first",

    )

    finetune_forecast_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
        prediction_channel_indices=tsp.prediction_channel_indices,
        loss=loss,
        quantile=quantile,
    )

    print(
        "Number of params before freezing backbone",
        count_parameters(finetune_forecast_model),
    )

    # Freeze the backbone of the model
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False

    # Count params
    print(
        "Number of params after freezing the backbone",
        count_parameters(finetune_forecast_model),
    )

    learning_rate, finetune_forecast_model = optimal_lr_finder(
        finetune_forecast_model,
        dset_train,
        batch_size=batch_size,
        enable_prefix_tuning=False,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        seed=SEED,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    # Evaluation
    print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
    
    finetune_forecast_trainer.model.loss = "mse"
    fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
    print(f"MSE:{fewshot_output['eval_loss']}")

    print("Compute MSE and MAE on test data")
    predictions_dict = finetune_forecast_trainer.predict(dset_test)
    
    predictions_list = []
    num_target_columns = len(target_columns)
    
    if isinstance(predictions_dict.predictions, (list, tuple)):
        for pred in predictions_dict.predictions:
            pred_array = np.array(pred, dtype=np.float32)
            if pred_array.ndim == 3 and pred_array.shape[1] == forecast_length and pred_array.shape[2] == num_target_columns:
                predictions_list.append(pred_array)
    else:
        predictions_array = np.array(predictions_dict.predictions, dtype=np.float32)
        if predictions_array.ndim == 3 and predictions_array.shape[1] == forecast_length and predictions_array.shape[2] == num_target_columns:
            predictions_list.append(predictions_array)

    predictions_np = np.concatenate(predictions_list, axis=0)
    
    if predictions_np.shape[0] > len(dset_test):
        predictions_np = predictions_np[:len(dset_test)]

    true_values = [sample['future_values'] for sample in dset_test]
    true_values_np = np.array(true_values, dtype=np.float32)

    if predictions_np.ndim == 3:
        num_features = predictions_np.shape[-1]
        mse = np.mean([
            mean_squared_error(
                true_values_np[:, :, i].flatten(),
                predictions_np[:, :, i].flatten()
            ) for i in range(num_features)
        ])
        mae = np.mean([
            mean_absolute_error(
                true_values_np[:, :, i].flatten(),
                predictions_np[:, :, i].flatten()
            ) for i in range(num_features)
        ])
    else:
        mse = mean_squared_error(true_values_np.flatten(), predictions_np.flatten())
        mae = mean_absolute_error(true_values_np.flatten(), predictions_np.flatten())
    print(f"{dataset_name}_{context_length}_{forecast_length}")
    print(f"MSE:{mse}, MAE:{mae}")
    print("+" * 60)
    f = open("/home/xiongshiwei/TTM/Mix_result_5%_fewshot.txt", 'a')
    f.write(f"{dataset_name}_{context_length}_{forecast_length}")
    f.write('\n')
    f.write(f"MSE:{mse}, MAE:{mae}")
    f.write('\n')



if __name__ == "__main__":

    SEED = 42
    # SEED = np.random.randint(1, 100000)
    set_seed(SEED)

    TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
    CONTEXT_LENGTH = 512

    domains = ['Energy', 'Healthcare', 'Nature', 'Environment', 'Traffic', 'Finance', 'Web', 'CloudOps']

    domain_dataset = {
        'Energy': ['CAISO_AL', 'NYISO_AL', 'Wind'],
        'Healthcare': ['Covid-19', 'illness'],
        'Environment': ['weather', 'AQShunyi', 'AQWan'],
        'Nature': ['CzeLan', 'ZafNoo'],
        'Traffic': ['M_DENSE', 'NYCTAXI_Inflow', 'NYCTAXI_Outflow'],
        'Finance': ['Exchange_rate', 'IBM_m', 'MSFT_w', 'NASDAQ_w'],
        'Web': ['Wike2000', 'BizITObs-Application', 'BizITObs-Service', 'BizITObs-L2C'],
        'CloudOps': ['Bitbrains-rnd100-cpuusage', 'Bitbrains-rnd100-memoryusage']
    }

    domain_seqlen = {
        'Energy': ['long', 'long', 'long'],
        'Healthcare': ['short', 'short'],
        'Environment': ['long', 'long', 'long'],
         'Nature': ['long', 'long'],
        'Traffic': ['long', 'short', 'short'],
        'Finance': ['long', 'short', 'short', 'short'],
        'Web': ['short', 'long', 'long', 'long'],
        'CloudOps': ['short', 'short']
    }


    for domain in domains:
        datasets = domain_dataset[domain]
        seqlens = domain_seqlen[domain]

        for idx, dataset in enumerate(datasets):
            dataset_name=dataset
            dataset_path=f"/home/xiongshiwei/DF-Eval//TTM/Dataset/{domain}/{dataset}.csv"
            seq_type = seqlens[idx]
            if seq_type == 'long':
                for pre_len in [96,192,336]:

                    timestamp_column = "date"
                    id_columns = []

                    data = pd.read_csv(
                        dataset_path,
                        parse_dates=[timestamp_column],
                    )

                    target_columns = [col for col in data.columns if col != timestamp_column]
                    column_specifiers = {
                        "timestamp_column": timestamp_column,
                        "id_columns": id_columns,
                        "target_columns": target_columns,
                        "control_columns": [],
                    }

                    
                    fewshot_finetune_eval(
                        dataset_name=dataset_name,
                        context_length=CONTEXT_LENGTH,
                        forecast_length=pre_len,
                        batch_size=64,
                        fewshot_percent=100
                        # fewshot_percent=5
                    )
