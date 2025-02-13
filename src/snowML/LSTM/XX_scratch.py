Share


You said:
I have a dictionary  called df_dict that contains huc_id as they key and pandas dataframes as the values.  The pandas dataframes will be used for model training 

I want to do k-fold cross validation.  Write a function called prep_input_folds that takes k as an input and returns k lists of huc_ids randomly selected from the available keys in df_dict
ChatGPT said:
Memory updated
Here’s a function to generate k random folds of huc_ids from df_dict:

python
Copy
Edit
import random

def prep_input_folds(df_dict, k):
    """Generate k random folds of huc_ids for cross-validation.
    
    Args:
        df_dict (dict): Dictionary with huc_id as keys and DataFrames as values.
        k (int): Number of folds.
    
    Returns:
        list of lists: k lists containing huc_ids for each fold.
    """
    huc_ids = list(df_dict.keys())
    random.shuffle(huc_ids)  # Shuffle the huc_ids to ensure randomness

    # Split huc_ids into k roughly equal-sized folds
    folds = [huc_ids[i::k] for i in range(k)]
    
    return folds