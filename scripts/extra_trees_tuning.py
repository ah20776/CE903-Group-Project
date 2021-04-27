from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from optuna import Trial, create_study, visualization
from optuna.samplers import TPESampler

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

from preprocess import preprocessed_data

data = []

def objective(trial: Trial):
    '''Function that creates classifier and returns the score'''
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    if bootstrap:
        oob_score = trial.suggest_categorical('oob_score', [True, False])
    else:
        oob_score = False
    parameters = {
        'n_estimators': 108,
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 1e-10, 1e-9),
        'max_features': 2,
        'bootstrap': bootstrap,
        'oob_score': oob_score,
        'random_state': 42,
    }
    classifier = ExtraTreesClassifier(**parameters)
    return cross_val_score(classifier, data[0], data[1], cv=10).mean()

def main():
    global data

    data = preprocessed_data()
    study = create_study(
        sampler=TPESampler(),
        study_name='framingham_extra_trees',
        direction='maximize'
    )
    max_workers = cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            executor.submit(study.optimize, objective, 20)

    print(study.best_value)
    print(study.best_params)

    visualization.plot_parallel_coordinate(study).show()
    visualization.plot_param_importances(study).show()

if __name__ == "__main__":
    main()
