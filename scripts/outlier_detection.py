from sklearn.datasets import make_moons
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

def plot_outliers(x: np.ndarray, pred: np.ndarray, clf: IsolationForest):
    """Plots the outliers that the IsolationForest has detected"""
    helper = lambda i: np.linspace(min((x[:, i])), max((x[:, i])), 500)
    xx, yy = np.meshgrid(helper(0), helper(1))
    pos = pred > 0
    neg = pred < 0

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title('Outliers detected using IsolationForest')
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('Blues_r'))

    plt.scatter(x[pos][:, 0], x[pos][:, 1], c='green', edgecolor='k')
    plt.scatter(x[neg][:, 0], x[neg][:, 1], c='red', edgecolor='k')

    plt.axis('tight')

    plt.xlim((xx.min(), xx.max()))
    plt.ylim((yy.min(), yy.max()))

    print(f'Total outliers detected: {neg.sum()}')

    plt.show()

def remove_outlier(x: np.ndarray, y: np.ndarray, plot=False, seed=42) -> np.ndarray:
    """Removes the outliers of the x vector. Optionally plots the vectors removed"""
    clf = IsolationForest(max_samples=100, contamination=0.1, random_state=seed)
    pred = clf.fit_predict(x)
    if plot:
        plot_outliers(x, pred, clf)
    return x[pred > 0], y[pred > 0]

def main():
    x, y = make_moons(1500, noise=0.2, random_state=42)
    remove_outlier(x, y, plot=True)

if __name__ == "__main__":
    main()
