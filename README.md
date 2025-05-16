# DF-Eval: Are Time Series Foundation Model Really Effective Across Diverse Fields?
We introduce DF-Eval, a comprehensive benchmark evaluating forecasting and imputation performance of foundation models. It systematically
compares pre-trained models, LLM-based architectures, and specific approaches
under standardized protocols.

## Dataset Preparation
You can obtain the well pre-processed datasets from [Kaggle](https://www.kaggle.com/datasets/shiwei131/df-eval), including 24 downstream datasets, which span **8 critical application fields**: Energy, Healthcare, Environment, Nature, Transportation, Finance, Web and CloudOps. Then place the downloaded data in the folder ./Dataset.

## Environment Preparation
For TSLib and each foundation model, we provide the environment.yml to reproduce the project environment.

```bash
# Reproduce the project environment from environment.yml(conda)
conda env create -f environment.yml
```



<!-- ## Support Task
> **[Forecasting](./scripts/forecast/README.md)**: We provide scripts for full- or few-shot forecasting.

> **[Imputation](./scripts/forecast/README.md)**: We provide scripts for full- or few-shot forecasting. -->

## Foundation Models Evaluation
### Pre-trained Model
**（1）Timer:**  We extend the publicly available [Large-Time-Series-Model](https://github.com/thuml/Large-Time-Series-Model) repo to adapt it to our diverse-field datasets, performing fine-tuning and evaluation accordingly. 
Please put the checkpoint from [Google Drive](https://drive.google.com/drive/folders/15oaiAl4OO5gFqZMJD2lOtX2fxHbpgcU8) under the folder ./checkpoints/ and use the following script to start. 

```bash
# Forecasting task
bash ./scripts/forecast/general.sh

# Imputation task
bash ./scripts/imputation/general.sh
```
    
   
**（2）Moment:**  We extend the publicly available [Moment](https://github.com/moment-timeseries-foundation-model/moment) repo.
Please use the following script to start. 

```bash
# Forecasting task
python tutorials/forecast_run.py

# Imputation task
python tutorials/imputation_run.py
```

**（3）TTM:**  We extend the publicly available [TinyTimeMixer (TTM)](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer) repo. Since each pre-trained TTM is tailored for a specific forecasting setting (defined by the context and forecast lengths), we are only able to evaluate on datasets with an input length of 512.
Please use the following script to start.     

```bash
# Forecasting task
python channel_mix_finetuning.py
```

**（4）Moirai:**  We extend the publicly available [uni2ts](https://github.com/SalesforceAIResearch/uni2ts) repo to perform fine-tuning and evaluation accordingly. Given that Moirai is a probabilistic model, we adjusted both its evaluation approach and metrics to to align with other models.
Please use the following script to start.

```bash
# 0、Data processing
bash script/builddata.sh

# 1、Fine-tuning, the yaml file can be created with cli/conf/finetune/createdir.ipynb
bash script/finetune.sh

# 2、Evaluation, the yaml file can be created with cli/conf/eval/createdir.ipynb
bash script/eval.sh
```


**（5）Chronos:**  We extend the publicly available [Chronos](https://github.com/amazon-science/chronos-forecasting) repo to adapt it to perform fine-tuning and evaluation accordingly. Given that Chronos is a probabilistic model, we adjusted both its evaluation approach and metrics to to align with other models.
Please use the following script to start.     

```bash
# 0、Data processing
python dataconvert.py

# 1、Fine-tuning
bash scripts/training/fine-tuning.sh

# 2、Evaluation
bash scripts/evaluation/eval.sh
```



### LLM-based Model
**（1）Time-LLM:**  We extend the publicly available [Time-LLM](https://github.com/KimMeen/Time-LLM) repo.
Please use the following script to start. Due to the large number of parameters in LLaMA-7B, we use BERT as the backbone instead.

```bash
# Forecasting task
bash scripts/TimeLLM_General.sh
```

**（2）GPT4TS:**  We extend the publicly available [One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) repo. 
Please use the following script to start. 

```bash
# Forecasting task
bash Long-term_Forecasting/scripts/general.sh

# Imputation task
bash Imputation/scripts/general.sh
```



## Specific Models Evaluation

 According to the performance rankings of specific models across different tasks in TSlib, we select the top five forecasting models and top three imputation models as our baselines. We utilize the open-source
code available at [TSLib](https://github.com/thuml/Time-Series-Library).

 ```bash
# Forecasting task
python scripts/long_term_forecast/PatchTST.py
python scripts/long_term_forecast/TimeXer.py
python scripts/long_term_forecast/iTransformer.py
python scripts/long_term_forecast/TimeMixer.py
python scripts/long_term_forecast/DLinear.py

# Imputation task
bash scripts/imputation/TimesNet.sh
bash scripts/imputation/Ns_Transformer.sh
bash scripts/imputation/DLinear.sh
```

