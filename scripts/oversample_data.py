from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import numpy as np

def oversample_data(x: np.ndarray, y: np.ndarray, seed=42):
    smote = SMOTE(random_state=seed)
    return smote.fit_resample(x, y)

def main():
    x, y = make_classification(weights=[0.3, 0.7])
    print((y == 0).sum())
    print((y == 1).sum())
    x, y  = oversample_data(x, y)
    print((y == 0).sum())
    print((y == 1).sum())

if __name__ == "__main__":
    main()
