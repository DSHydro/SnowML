import pandas as pd

from snowML.datapipe.utils import set_data_constants as sdc
from snowML.datapipe.multi_huc_all import compile_geos, process_multi_huc


def get_extra_test_hucs(csv_path: str) -> list[int]:
    """Return list of HUC12s that are marked as not_in_training == 1."""
    df = pd.read_csv(csv_path)
    if "huc_id" not in df.columns or "not_in_training" not in df.columns:
        raise ValueError("CSV must contain 'huc_id' and 'not_in_training' columns.")
    mask = df["not_in_training"] == 1
    extra = df.loc[mask, "huc_id"].dropna().astype(str).tolist()
    return extra


def run_stgnn_dataprep(
    csv_path: str = "dhsvm_lidar_hucs.csv",
    overwrite_gold: bool = False,
    overwrite_mod: bool = False,
) -> None:
    """
    Create gold and ST-GNN-specific model-ready data for extra HUC12s.

    - Reads dhsvm_lidar_hucs.csv and selects rows with not_in_training == 1.
    - Builds geometries for those HUC12s.
    - Runs bronze->gold (as needed) and gold+silver->model-ready into the
      ST-GNN model-ready bucket configuration ('stgnn' env in set_data_constants).
    """
    extra_hucs = get_extra_test_hucs(csv_path)
    if not extra_hucs:
        print("No extra HUC12s found with not_in_training == 1.")
        return

    print(f"Found {len(extra_hucs)} extra HUC12s to process.")

    bucket_dict = sdc.create_bucket_dict("prod")

    # Build geometries for these HUC12s (HUC12-level polygons)
    geos = compile_geos(extra_hucs)

    # End-to-end pipeline: bronze -> gold (if missing) -> model-ready (ST-GNN bucket)
    process_multi_huc(
        geos,
        bucket_dict=bucket_dict,
        var_list=None,
        overwrite_gold=overwrite_gold,
        overwrite_mod=overwrite_mod,
    )


if __name__ == "__main__":
    run_stgnn_dataprep()

