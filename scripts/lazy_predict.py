from lazypredict.Supervised import LazyClassifier

from preprocess import split_preprocessed_data

if __name__ == "__main__":
    seed = 42
    train, _, test = split_preprocessed_data(seed)
    classifier = LazyClassifier(predictions=False, random_state=seed)
    scores = classifier.fit(train[0], test[0], train[1], test[1])
    print(scores)
