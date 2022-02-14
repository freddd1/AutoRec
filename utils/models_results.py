import json
import numpy as np
import pandas as pd
import glob


def models_results(model_name=None):
    l = []

    if model_name is None:
        models = ['MF', 'AutoRec', 'AutoRecV2']
    else:
        models = [model_name]

    for model_name in models:
        files = glob.glob(f'models_params/{model_name}*')
        for file_path in files:
            with open(file_path, 'r') as f:
                model_parmas = json.load(f)

            file_name = file_path.split("\\")[1]
            train_loss = np.nanmean(model_parmas['train_losses'][-1])
            val_loss = np.nanmean(model_parmas['val_losses'][-1])

            l.append([file_name, train_loss, val_loss, model_parmas])

    results = pd.DataFrame(l, columns=['model_name', 'train_loss', 'val_loss', 'params'])

    return results
