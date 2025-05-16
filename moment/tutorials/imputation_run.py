import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from momentfm.utils.utils import control_randomness
from torch.utils.data import DataLoader
from momentfm import MOMENTPipeline
from momentfm.utils.utils import EarlyStopping
import numpy as np
from momentfm.utils.masking import Masking
from tqdm import tqdm
from momentfm.utils.forecasting_metrics import mse, mae
from momentfm.data.informer_dataset import InformerDataset
import os





def run_imputation_pipeline(pretrain_model, full_file_path_and_name, seq_len, mask_ratio, seed, batchsize, output_file):

    control_randomness(seed=seed) # Set random seeds for PyTorch, Numpy etc.


    # Lodd model
    model = MOMENTPipeline.from_pretrained(
        pretrain_model,
        model_kwargs={'task_name': 'reconstruction',
                    'seq_len' : seq_len,
                    #   'freeze_encoder': True, # Freeze the patch embedding layer
                    #   'freeze_embedder': True, # Freeze the transformer encoder
                    #   'freeze_head': False, 
                    
                    } # For imputation, we will load MOMENT in `reconstruction` mode
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )
    model.init()
    num_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters: {num_params}")


    # Load data
    train_dataset = InformerDataset(data_split="train", task_name='imputation', full_file_path_and_name=full_file_path_and_name, seq_len=192, random_seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # val_dataset = InformerDataset(data_split="val", task_name='imputation', full_file_path_and_name=full_file_path_and_name, seq_len=192)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    test_dataset = InformerDataset(data_split="test", task_name='imputation', full_file_path_and_name=full_file_path_and_name, seq_len=192, random_seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    # Optimize Mean Squarred Error using your favourite optimizer
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=2, verbose=True)

    mask_generator = Masking(mask_ratio=mask_ratio) # Mask 30% of patches randomly 


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device).float()

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mask_generator = Masking(mask_ratio=mask_ratio) # Mask 30% of patches randomly 

    num_epochs = 10  # Number of epochs to train

    for epoch in range(num_epochs):
        losses = []
        model.train()  # Set model to training mode
        for batch_x, batch_masks in tqdm(train_loader, total=len(train_loader)):
        # for batch_x, batch_masks in train_loader:
            n_channels = batch_x.shape[1]
            
            # Reshape to [batch_size * n_channels, 1, window_size]
            batch_x = batch_x.reshape((-1, 1, seq_len)).float().to(device)  # Ensure it's on the correct device
            original = batch_x.clone().detach()

            batch_masks = batch_masks.to(device).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)
            
            # Randomly mask some patches of data
            mask = mask_generator.generate_mask(
                x=batch_x, input_mask=batch_masks).to(device).long()  # Ensure mask is also on the correct device

            # Forward pass
            output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask) 
            
            # Compute loss
            recon_loss = criterion(output.reconstruction, original)
            observed_mask = batch_masks * (1 - mask)
            masked_loss = observed_mask * recon_loss
            
            loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
            losses.append(loss.item())


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Epoch {epoch}: Train loss: {average_loss:.4f}")


        if (epoch + 1) % num_epochs == 0:
            model.eval()  # Set model to evaluation mode
            trues, preds, masks = [], [], []
            with torch.no_grad():
                for batch_x, batch_masks in tqdm(test_loader, total=len(test_loader)):
                    trues.append(batch_x.numpy())
                    
                    batch_x = batch_x.to(device).float()
                    n_channels = batch_x.shape[1]
                    
                    # Reshape to [batch_size * n_channels, 1, window_size]
                    batch_x = batch_x.reshape((-1, 1, seq_len)) 
                    
                    batch_masks = batch_masks.to(device).long()
                    batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)
                    
                    mask = mask_generator.generate_mask(
                        x=batch_x, input_mask=batch_masks).to(device).long()

                    output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)  # [batch_size, n_channels, window_size]
                    
                    reconstruction = output.reconstruction.detach().cpu().numpy()
                    mask = mask.detach().squeeze().cpu().numpy()
                    
                    # Reshape back to [batch_size, n_channels, window_size]
                    reconstruction = reconstruction.reshape((-1, n_channels, seq_len)) 
                    mask = mask.reshape((-1, n_channels, seq_len))
                            
                    preds.append(reconstruction)
                    masks.append(mask)

            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            masks = np.concatenate(masks)

            print(f"Shapes: preds={preds.shape} | trues={trues.shape} | masks={masks.shape}")
            print(f"MSE={mse(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')}, MAE={mae(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')}")

            mse_metric = mse(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')
            mae_metric = mae(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')

            dataset_name = full_file_path_and_name.split('/')[-1]
            
            with open(output_file, "a") as f:
                f.write(f"Finetuned model: {pretrain_model} | Dataset: {dataset_name} | Mask_ratio: {mask_ratio} | Test MSE: {mse_metric} | Test MAE: {mae_metric}\n")
            print(f"Results written to {output_file}")





if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    seq_len = 192
    seeds = [0, 100, 200]
    pretrained_model = ["AutonLab/MOMENT-1-small", "AutonLab/MOMENT-1-base", "AutonLab/MOMENT-1-large"]
    batchsize = 8
    
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



    for seed in seeds:
        output_file = f"imputation_results_seed{seed}.txt"
        for model in pretrained_model:
            print(f"Finetuning on Pretrained model: {model}")
            for domain in Domain:
                datasets = DomainDataset[domain]
                for dataset in datasets:
                    full_file_path_and_name = f'Dataset/{domain}/{dataset}.csv'
                    for mask_ratio in [0.125, 0.25, 0.375, 0.5]:
                        current_batchsize = batchsize
                        success = False
                        while not success and current_batchsize > 0:
                            try:
                                print(f"Running {model} on {dataset} | mask_ratio={mask_ratio} | batchsize={current_batchsize}")
                                run_imputation_pipeline(model, full_file_path_and_name, seq_len, mask_ratio, seed, current_batchsize, output_file)
                                success = True
                            except RuntimeError as e:
                                if 'CUDA out of memory' in str(e):
                                    print(f"⚠️ CUDA OOM: Reducing batch size from {current_batchsize} to {current_batchsize // 2}")
                                    current_batchsize = current_batchsize // 2
                                    torch.cuda.empty_cache()
                                else:
                                    raise e
                        if not success:
                            print(f"❌ Skipped {dataset} at mask_ratio={mask_ratio} due to repeated CUDA OOM")


