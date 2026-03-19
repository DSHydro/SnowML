# ConvLSTM (spatiotemporal model)

This folder contains the **ConvLSTM** workflow used in SnowML: data preparation (HUC10 gridded tensors), model training, inference, and evaluation/plots.

## What’s in here

- `dataprep/`
  - `get_huc_shapes.ipynb`: downloads **HUC10** shapes (and **HUC12 within each HUC10**) and uploads GeoJSON + metadata CSVs to S3.
  - `dataprep.ipynb`: converts each HUC10 into a **fixed-size** gridded dataset (default `24×24` at 4 km) and writes one **zarr per HUC10** to S3.
- `ConvLSTM_model.ipynb`: end-to-end notebook for **loading prepared zarrs**, defining the model, **training**, **testing**, saving predictions, and basin-level evaluation.
- `iter_artifacts/`: contains saved model artifacts from key training iterations (e.g., **best** and **last epoch** models after 20 epochs), which can be used to skip training and proceed directly to testing or inference.
- `test_metrics.ipynb`: plotting + analytics on saved evaluation tables (scatterplots, breakdowns by basin/elevation).
- `convlstm_results.csv`: metrics table (e.g. per-HUC12 `mse`, `kge`) used by `test_metrics.ipynb`.
- `huc12_snow_types.csv`: mapping of `huc12 → snow_type` used for grouped analysis.

## Typical run order

1. **(Once / when watershed set changes)** Create shapes + metadata  
   Run `dataprep/get_huc_shapes.ipynb`.
2. **Create model-ready gridded data**  
   Run `dataprep/dataprep.ipynb` to generate per-HUC10 zarrs.
3. **Train + evaluate the ConvLSTM**  
   Run `ConvLSTM_model.ipynb`.
4. **Make plots / deeper analysis**  
   Run `test_metrics.ipynb` (uses `convlstm_results.csv`).

