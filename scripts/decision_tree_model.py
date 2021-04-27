from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from preprocess import preprocessed_data

def main():
    data = preprocessed_data()
    classifier = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=10,
        random_state=42
    )
    accuracy = cross_val_score(classifier, data[0], data[1], cv=10).mean()
    # 0.9324607329842932
    print('Cross validation mean accuracy:', accuracy)

if __name__ == "__main__":
    main()
