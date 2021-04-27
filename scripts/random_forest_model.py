from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from preprocess import preprocessed_data

def main():
    data = preprocessed_data()
    classifier = RandomForestClassifier(
        n_estimators=115,
        criterion='entropy',
        min_samples_split=3,
        max_features=2,
        random_state=42,
    )
    accuracy = cross_val_score(classifier, data[0], data[1], cv=10).mean()
    # 0.9324607329842932
    print('Cross validation mean accuracy:', accuracy)

if __name__ == "__main__":
    main()
