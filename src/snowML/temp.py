

## SETUP STUFF 
# 1. Clone or update(pull) repo
# 2 pip install -e .
# 3 load requirement.txt if new instance
# 4 probably have to load earthengine-api and zarr anyway? 
# 5 run ee.Authenticate (auth_mode='notebook') if not ee.Authenticate = True
# 6 Adjust the number of workers for the instance type
# 7 install htop:  
#   sudo apt update
#   sudo apt install -y htop

def hist_from_run(run_id, f_out, tracking_id = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"):
    metrics = load_ml_metrics(tracking_id, run_id, save_local=True)
    metrics_last = summarize_bystep(metrics, 9, agg_lev = 12) # TO DO - Make final step dynamic
    plot_test_kge_histogram(metrics_last, output_file = f_out)