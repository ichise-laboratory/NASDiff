# NASDiff

NASDiff: a Noise-Aware Diffusion with Short-trajectory learning for consistent time series imputation.

## Environment

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

## Quick Start

Run a short smoke test first. This verifies the environment, data loading, model construction, and epoch-level console output without launching the full training schedule.

```bash
python framework.py --datasets Italy_Air --epochs 1 --device cpu
```

Run the complete default NASDiff configuration on ETT with point missing ratio 0.1:

```bash
python framework.py
```

The full NASDiff model is computationally heavy on CPU. When CUDA is available, prefer:

```bash
python framework.py --device cuda:0
```

Run the manuscript missing scenarios:

```bash
python framework.py --model_name NASDiff --datasets Italy_Air --pattern point --missing_ratio 0.1
python framework.py --model_name NASDiff --datasets ETT --pattern block --missing_ratio 0.5
python framework.py --model_name NASDiff --datasets ETT --pattern subseq --missing_ratio 0.5
```

Run the included baselines on the same setting:

```bash
python framework.py --model_name Transformer --datasets ETT --pattern point --missing_ratio 0.1
python framework.py --model_name CSDI --datasets ETT --pattern point --missing_ratio 0.1
```

Use a specific device:

```bash
python framework.py --device cpu
python framework.py --device cuda:0
```

Test a saved checkpoint:

```bash
python framework.py --mode test_with_trained --model_name NASDiff --checkpoint "ETT NASDiff 0.0 seed31 Test 0.1146 point 0.1.pt"
python framework.py --mode test_with_trained --model_name CSDI --checkpoint "ETT CSDI 0.0 seed31 Test 0.1542 point 0.1.pt"
python framework.py --mode test_with_trained --model_name Transformer --checkpoint "ETT Transformer 0.0 seed31 Test 0.1653 point 0.1.pt"
```

## Data and Logs

Supported datasets are `Air`, `Italy_Air`, `ETT`, `Pedestrian`, `Pems_Traffic`, `Electricity_Load`, `Physionet2012`, and `Physionet2019`. Dataset loaders follow the same split-first preprocessing path as the manuscript: split the raw data, fit normalization on the training split, transform validation/test splits with the training scaler, and then generate artificial missing masks.

Each run writes its configuration and final results to `logs/<model>.log`. During training, the console prints the selected configuration, data/model readiness, epoch-start messages, epoch-level train/validation metrics, and early-stopping patience status. In `test_with_trained` mode, the console prints the final test metrics.
