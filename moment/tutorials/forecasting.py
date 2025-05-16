import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from momentfm import MOMENTPipeline
import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics


def run_forecasting_pipeline(pretraind_model, full_file_path_and_name, seq_len, forecast_horizon, seed, output_file):
    model = MOMENTPipeline.from_pretrained(
        # "AutonLab/MOMENT-1-large", 
        pretraind_model,
        model_kwargs={
            'task_name': 'forecasting',
            'seq_len' : seq_len,
            'forecast_horizon': forecast_horizon,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
    )
    model.init()

    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=seed) 
    
    # Load data
    train_dataset = InformerDataset(data_split="train", random_seed=seed, full_file_path_and_name=full_file_path_and_name, forecast_horizon=forecast_horizon, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = InformerDataset(data_split="test", random_seed=seed, full_file_path_and_name=full_file_path_and_name, forecast_horizon=forecast_horizon, seq_len=seq_len)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cur_epoch = 0
    max_epoch = 10

    if full_file_path_and_name == "/home/xiongshiwei/moment/Dataset/Traffic/METR_LA.csv":
        max_epoch = 5

    # Move the model to the GPU
    model = model.to(device)

    # Move the loss function to the GPU
    criterion = criterion.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create a OneCycleLR scheduler
    max_lr = 1e-4
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

    # Gradient clipping value
    max_norm = 5.0

    while cur_epoch < max_epoch:
        losses = []
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            with torch.amp.autocast(device_type='cuda'):
                # print(timeseries.shape, input_mask.shape)
                output = model(x_enc=timeseries, input_mask=input_mask)
            
            loss = criterion(output.forecast, forecast)

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step()
        cur_epoch += 1
        
        if (cur_epoch + 1) % max_epoch == 0:
            # Evaluate the model on the test split
            trues, preds, histories, losses = [], [], [], []
            model.eval()
            with torch.no_grad():
                for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(device)
                    input_mask = input_mask.to(device)
                    forecast = forecast.float().to(device)

                    with torch.amp.autocast(device_type='cuda'):
                        output = model(x_enc=timeseries, input_mask=input_mask)
                    
                    loss = criterion(output.forecast, forecast)                
                    losses.append(loss.item())

                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())
            
            losses = np.array(losses)
            average_loss = np.average(losses)
            model.train()

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)
            
            metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

            dataset_name = full_file_path_and_name.split('/')[-1]
            # print(f"Finetuned model: {pretraind_model} | Dataset: {dataset_name} | Forecast Horizon: {forecast_horizon} | Test MSE: {metrics.mse:.4f} | Test MAE: {metrics.mae:.4f}")
            print(f"Finetuned model: {pretraind_model} | Dataset: {dataset_name} | Forecast Horizon: {forecast_horizon} | Test MSE: {metrics.mse} | Test MAE: {metrics.mae}")
            
            
            # Write the results to a file
            with open(output_file, "a") as f:
                # f.write(f"Finetuned model: {pretraind_model} | Dataset: {dataset_name} | Forecast Horizon: {forecast_horizon} | Test MSE: {metrics.mse:.4f} | Test MAE: {metrics.mae:.4f}\n")
                f.write(f"Finetuned model: {pretraind_model} | Dataset: {dataset_name} | Forecast Horizon: {forecast_horizon} | Test MSE: {metrics.mse} | Test MAE: {metrics.mae}\n")
            print(f"Results written to {output_file}")
        