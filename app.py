import os, ast
import pandas as pd
from pycaret.datasets import get_data

dataset = os.environ["INPUT_DATASET"]
target = os.environ["INPUT_TARGET"]
usecase = os.environ["INPUT_USECASE"]

if usecase == 'regression':
    from pycaret.regression import *
elif usecase == 'classification':
    from pycaret.classification import *

exp1 = setup(data, target = target, session_id=123, silent=True, html=False, log_experiment=True, experiment_name='exp_github')

best = compare_models()

best_model = finalize_model(best)

save_model(best_model, 'model')

logs_exp_github = get_logs(save=True)


