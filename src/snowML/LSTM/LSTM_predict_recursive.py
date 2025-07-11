# pylint: disable=C0103

import torch
from snowML.LSTM import LSTM_pre_process as pp


def recursive_forecast(model_dawgs, df_test, lagged_swe_idx, params):
    model_dawgs.eval()
    y_pred_recur = []
    lag_days = params["lag_days"]
    count = 0

    for idx in range (params["lookback"], df_test.shape[0]):
        df_small = df_test.iloc[count: idx+1, :].copy()
        X_test_small, _ = pp.create_tensor(df_small, params['lookback'], params['var_list'])
        with torch.no_grad():
            y_pred_next = model_dawgs(X_test_small).cpu().numpy()
            #print("Shape of y_pred_next:", y_pred_next.shape)

        # save prediction
        y_pred_recur.append(y_pred_next)

        # update df_test lagged swe with predicted
        # TO DO - Figure out if need a + 1 here
        replace_idx  = idx+1 + lag_days + 1
        if replace_idx < df_test.shape[0]:
            df_test.iloc[replace_idx, lagged_swe_idx] = y_pred_next

        # update count
        count += 1

    return y_pred_recur
    