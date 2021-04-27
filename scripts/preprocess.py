"""Script to preprocess the data"""
from pandas import read_csv, DataFrame
from numpy import append

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from outlier_detection import remove_outlier
from oversample_data import oversample_data


def get_data_from_dataframe(dataframe: DataFrame):
    '''Extract the X and y values from the dataframe'''
    target_column = 'TenYearCHD'
    X = dataframe.drop(columns=[target_column]).values
    y = dataframe[target_column].values
    return X, y

def get_data():
    '''
    Reads the csv file and gets the features and target values from the file
    *Warning* Currently removes the null values
    '''
    dataframe = read_csv('../data/framingham.csv').dropna()
    return get_data_from_dataframe(dataframe)

def preprocessed_data(seed=42):
    '''
    Gets the data after preprocessing it. The preprocess currently consists of:
    * Removing outliers
    * Oversampling the data for balancing target values

    '''
    X, y = get_data()
    X, y = remove_outlier(X, y, seed=seed)
    X, y = oversample_data(X, y, seed)
    return X, y

def get_saved_data(path):
    '''Get the saved data of the path'''
    return get_data_from_dataframe(read_csv(path))

def split_data(X, y, train_size=0.8, seed=42):
    '''
    Splits the dataset into train, test and validation. Test and validation
    sets have the same size
    '''
    test_size = 1 - train_size
    train_x, rest_x, train_y, rest_y = \
        train_test_split(X, y, test_size=test_size, random_state=seed)
    val_x, test_x, val_y, test_y = \
        train_test_split(rest_x, rest_y, test_size=0.5, random_state=seed)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def save_data(data: tuple, path: str):
    '''
    Saves the data stored as a tuple of two dataframes containing X and y
    respectively
    '''
    X, y = data
    complete_data = append(X, y.reshape(y.shape[0], 1), axis=1)
    original_dataframe = read_csv('../data/framingham.csv')
    dataframe = DataFrame(complete_data, columns=original_dataframe.columns)
    dataframe.to_csv(path)

def get_split_paths():
    '''Get the paths where each split is stored'''
    splits = ['train', 'validation', 'test']
    return [f'../data/{split}.csv' for split in splits]

def split_preprocessed_data(train_size=0.8, scale=True, seed=42, pre_saved=True):
    '''Splits the preprocessed data into train, test and validation'''
    split_paths = get_split_paths()
    if pre_saved:
        return tuple(map(get_saved_data, split_paths))
    X, y = preprocessed_data(seed)
    if scale:
        X = MinMaxScaler().fit_transform(X)
    datasets = split_data(X, y, train_size, seed)
    for data, split_path in zip(datasets, split_paths):
        save_data(data, split_path)
    return datasets
