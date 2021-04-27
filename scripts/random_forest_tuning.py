from optuna import Trial
from sklearn.ensemble import RandomForestClassifier

from tune_model import tune_model

def objective(trial: Trial):
    '''Function that creates classifier and returns the score'''
    parameters = {
        'n_estimators': 115,
        'criterion': 'entropy',
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'max_features': 2,
        'random_state': 42,
    }
    return RandomForestClassifier(**parameters)

def main():
    tune_model(objective)

if __name__ == "__main__":
    main()
