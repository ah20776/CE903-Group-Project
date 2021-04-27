'''Utils for tuning model with optuna'''
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from optuna import create_study, visualization
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score

from preprocess import preprocessed_data

data = []

def create_objective(create_model):
    def objective(trial):
        model = create_model(trial)
        return cross_val_score(model, data[0], data[1], cv=10).mean()
    return objective

def tune_model(create_model, n_trials=50):
    global data

    data = preprocessed_data()
    objective = create_objective(create_model)
    study = create_study(sampler=TPESampler(), direction='maximize')
    max_workers = cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            executor.submit(study.optimize, objective, n_trials)

    print(study.best_value)
    print(study.best_params)

    visualization.plot_parallel_coordinate(study).show()
    visualization.plot_param_importances(study).show()
