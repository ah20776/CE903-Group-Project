from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

from preprocess import preprocessed_data

def main():
    data = preprocessed_data()
    classifier = ExtraTreesClassifier(
        n_estimators=108,
        min_weight_fraction_leaf=1.8170299645156592e-07,
        max_features=2,
        random_state=42,
    )
    accuracy = cross_val_score(classifier, data[0], data[1], cv=10).mean()
    # 0.9593368237347294
    print('Cross validation mean accuracy:', accuracy)

if __name__ == "__main__":
    main()
