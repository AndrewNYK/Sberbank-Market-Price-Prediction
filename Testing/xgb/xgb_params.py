xgb_params = {
    "n_estimators": 500,
    "device": "cuda",
    # 'random_state': 24,
    "objective": "reg:squarederror",
    "eval_metric": "rmsle",
    "enable_categorical": True,
    "early_stopping_rounds": 50,
    "colsample_bytree": 0.8,
    "eta": 0.05,
    "gamma": 8.084029345968737,
    "max_depth": 8,
    "min_child_weight": 2.0,
    "reg_alpha": 93.0,
    "reg_lambda": 0.8685796539747039,
}

# For split inv/own product type
xgb_params_inv = {
    "n_estimators": 250,  # Tested against 500 and returned same results
    "device": "cuda",
    # 'random_state': 24,
    "objective": "reg:squarederror",
    "eval_metric": "rmsle",
    "enable_categorical": True,
    "early_stopping_rounds": 10,
    "colsample_bytree": 0.8,
    "eta": 0.05,
    "gamma": 8.084029345968737,
    "max_depth": 6,
    "min_child_weight": 2.0,
    "reg_alpha": 93.0,
    "reg_lambda": 0.8685796539747039,
    "n_jobs": 4,
}

xgb_params_own = {
    "n_estimators": 250,
    "device": "cuda",
    # 'random_state': 24,
    "objective": "reg:squarederror",
    "eval_metric": "rmsle",
    "enable_categorical": True,
    "early_stopping_rounds": 10,
    "colsample_bytree": 0.8,
    "eta": 0.05,
    "gamma": 8.084029345968737,
    "max_depth": 6,
    "min_child_weight": 2.0,
    "reg_alpha": 93.0,
    "reg_lambda": 0.8685796539747039,
    "n_jobs": 4,
}

# Number of features
num_inv_top_features = 70
num_own_top_features = 45
