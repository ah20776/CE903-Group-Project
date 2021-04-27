from optuna import Trial

from sklearn.tree import DecisionTreeClassifier

from tune_model import tune_model

def objective(trial: Trial):
    '''Function that creates classifier and returns the score'''
    return DecisionTreeClassifier(
        criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
        splitter='best',
        # max_depth=suggest_categorical(),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 5),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 2),
        max_features=10,
        random_state=42
    )

def main():
    tune_model(objective, 100)

if __name__ == "__main__":
    main()
