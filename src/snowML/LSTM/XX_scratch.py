

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