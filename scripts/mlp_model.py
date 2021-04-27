from tensorflow import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from preprocess import split_preprocessed_data

train, val, test = [], [], []

def build_model():
    model = Sequential([
        InputLayer(input_shape=(train[0].shape[1],)),
        Dense(132, 'relu', True, 'lecun_uniform', 'variance_scaling'),
        Dense(244, 'relu', False, 'variance_scaling'),
        Dense(132, 'relu', False, 'lecun_normal', 'glorot_uniform'),
        Dense(79, 'tanh', True, 'lecun_normal', 'lecun_uniform'),
        Dense(21, 'relu', True, 'he_normal', 'truncated_normal'),
        Dense(1)
    ])
    lr = 1.3808590387180374e-3
    rho = 0.9456685761083077
    momentum = 0.360510467533835
    epsilon = 7.029756373620823e-8
    centered = True
    optimizer = RMSprop(lr, rho, momentum, epsilon, centered)
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def trained_model():
    model = build_model()
    epochs = 68
    patience = epochs // 4
    early_stopping = EarlyStopping(
        'val_accuracy',
        patience=patience,
        restore_best_weights=True
    )
    model.fit(
        train[0],
        train[1],
        validation_data=val,
        epochs=epochs,
        batch_size=57,
        callbacks=[early_stopping],
        verbose=0
    )
    return model

def get_accuracy(model: Sequential, dataset: tuple):
    return model.evaluate(dataset[0], dataset[1], verbose=0)[1]

def main():
    global train, val, test

    seed = 42
    random.set_seed(seed)
    train, val, test = split_preprocessed_data(seed=seed, pre_saved=False)
    model = trained_model()
    print('Training accuracy', get_accuracy(model, train))
    print('Val accuracy', get_accuracy(model, val))
    print('Test accuracy', get_accuracy(model, test))

if __name__ == "__main__":
    main()
