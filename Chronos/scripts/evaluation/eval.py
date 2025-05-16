import logging
from pathlib import Path
from typing import Iterable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm
from gluonts.ev.metrics import MSE, MAE 



from chronos import (
    BaseChronosPipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
    ForecastType,
)

app = typer.Typer(pretty_exceptions_enable=False)

def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    print(hf_dataset.features)
    print("---------------------------------------")
    print(series_fields)
    series_fields.remove("timestamps")
    print(len(hf_dataset))
    print(len(series_fields))

    dataset_length = len(hf_dataset) * len(series_fields)
    print(f"dataset_length: {dataset_length}")

    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamps"]).to_period()[0].freqstr

    gts_dataset = []
    for hf_entry in hf_dataset:
        # print(hf_entry)
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamps"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    print(f"gts_dataset_len: {len(gts_dataset)}")
    assert len(gts_dataset) == dataset_length

    return gts_dataset

def load_and_split_dataset(backtest_config: dict):
    dataset_path = Path(backtest_config["dataset_path"])
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    max_history = backtest_config["max_history"]

    ds = datasets.Dataset.from_file(str(dataset_path))
    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(ds)
    
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=1, max_history=max_history) # 添加了distance

    return test_data

def generate_forecasts(
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length: int,
    batch_size: int,
    **predict_kwargs,
):
    forecast_outputs = []
    total_batches = len(test_data_input) // batch_size + int(len(test_data_input) % batch_size > 0)
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size),desc="Generating Forecasts",total=total_batches,leave=True,dynamic_ncols=True):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_outputs.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                **predict_kwargs,
            ).numpy()
        )
    forecast_outputs = np.concatenate(forecast_outputs)

    forecasts = []
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])

        if pipeline.forecast_type == ForecastType.SAMPLES:
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        elif pipeline.forecast_type == ForecastType.QUANTILES:
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, pipeline.quantiles)),
                    start_date=forecast_start_date,
                )
            )

    return forecasts

@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    pipeline = BaseChronosPipeline.from_pretrained(
        str(chronos_model_id),
        device_map=device,
        torch_dtype=torch_dtype,
    )

    if isinstance(pipeline, ChronosPipeline):
        predict_kwargs = dict(
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif isinstance(pipeline, ChronosBoltPipeline):
        predict_kwargs = {}

    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
        dataset_name = config["dataset_path"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        forecasts = generate_forecasts(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            **predict_kwargs,
        )
    
        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[
                    # MASE(),
                    # MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                    MSE(), #修改了metric中MSE[mean]为MSE[0.5]
                    MAE(),
                ],
                batch_size=5000,
                # batch_size=100,

            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append(
            {"dataset": dataset_name, "model": str(chronos_model_id), "prelen": prediction_length, **metrics[0]}
        )
        print(f"dataset: {dataset_name}, model: {str(chronos_model_id)}, prelen: {prediction_length}, {metrics[0]['MSE[0.5]']},{metrics[0]['MAE[0.5]']}")

    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            # {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL", "MSE[mean]": "MSE", "MAE[0.5]": "MAE"},
            # {"MSE[mean]": "MSE", "MAE[0.5]": "MAE"},
            {"MSE[0.5]": "MSE[0.5]", "MAE[0.5]": "MAE[0.5]"},
            axis="columns",
        )
        .sort_values(by="dataset")
        # .round(4)
    )
    results_df.to_csv(metrics_path, index=False)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()
