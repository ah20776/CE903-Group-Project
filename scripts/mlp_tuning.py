from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import cpu_count

from optuna import Trial, create_study, visualization
from optuna.integration import TFKerasPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, RandomSampler

from tensorflow import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from preprocess import split_preprocessed_data

possible_activation = ['softmax', 'relu', 'linear', 'sigmoid', 'swish', 'tanh']
initializers = [
    'zeros', 'variance_scaling', 'truncated_normal', 'random_uniform',
    'random_normal', 'lecun_uniform', 'lecun_normal', 'he_uniform',
    'glorot_uniform', 'glorot_normal', 'he_normal'
]
train, val, test = [], [], []

@dataclass
class Layer():
    number: int
    trial: Trial
    _layer_prefix = None

    @property
    def layer_prefix(self):
        if self._layer_prefix is None:
            self._layer_prefix = f'layer_{self.number}'
        return self._layer_prefix

    @property
    def units(self) -> int:
        number = self.number
        if number == 0:
            min_v, max_v = 129, 132
            # return 127
        elif number == 1:
            min_v, max_v = 240, 245
            # return 228
        elif number == 2:
            min_v, max_v = 130, 135
            # return 138
        elif number == 3:
            min_v, max_v = 78, 80
            # return 81
        elif number == 4:
            min_v, max_v = 20, 21
            # return 21
        else:
            min_v, max_v = 19, 251
        return self.trial.suggest_int(
            f'{self.layer_prefix}_units',
            min_v,
            max_v
        )

    @property
    def activation(self) -> str:
        number = self.number
        if number == 0:
            return 'relu'
        elif number == 1:
            return 'relu'
        elif number == 2:
            return 'relu'
        elif number == 3:
            return 'tanh'
        elif number == 4:
            return 'relu'
        return self.trial.suggest_categorical(
            f'{self.layer_prefix}_activation',
            possible_activation
        )

    @property
    def use_bias(self):
        # return True
        number = self.number
        if number == 3:
            return True
        if number == 1:
            return False
        return self.trial.suggest_categorical(
            f'{self.layer_prefix}_use_bias',
            [True, False],
        )

    @property
    def kernel_initializer(self):
        # return 'glorot_uniform'

        number = self.number
        if number == 0:
            return 'lecun_uniform'
        if number == 1:
            kernel_initializers = ['glorot_uniform', 'variance_scaling']
        elif number == 2:
            kernel_initializers = ['glorot_uniform', 'lecun_normal']
        elif number == 3:
            kernel_initializers = ['glorot_uniform', 'lecun_normal']
        elif number == 4:
            kernel_initializers = ['glorot_uniform', 'he_normal']
        else:
            kernel_initializers = initializers

        return self.trial.suggest_categorical(
            f'{self.layer_prefix}_kernel_initializer',
            kernel_initializers,
        )

    @property
    def bias_initializer(self):
        number = self.number
        if number == 1:
            return 'zeros'
        if number == 0:
            bias_initializers = ['lecun_uniform', 'variance_scaling', 'zeros']
        if number == 3:
            bias_initializers = ['he_normal', 'variance_scaling', 'glorot_uniform', 'lecun_uniform']
        if number == 4:
            bias_initializers = [init for init in initializers]
            bias_initializers.remove('random_uniform')
        else:
            bias_initializers = initializers
        # bias_initializers = initializers
        return self.trial.suggest_categorical(
            f'{self.layer_prefix}_bias_initializer',
            bias_initializers,
        )

    @property
    def dense_layer(self):
        return Dense(
            self.units,
            self.activation,
            self.use_bias,
            self.kernel_initializer,
            self.bias_initializer
        )

def create_model(trial: Trial):
    layers = [InputLayer(input_shape=(train[0].shape[1],))]
    num_layers = 5
    for number in range(num_layers):
        layers.append(Layer(number, trial).dense_layer)
    layers.append(Dense(1))
    model = Sequential(layers)
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

def objective(trial: Trial):
    model = create_model(trial)
    batch_size = 57
    epochs = 68
    patience = epochs // 4
    monitor = 'val_accuracy'
    callbacks = [
        TFKerasPruningCallback(trial, monitor),
        EarlyStopping(monitor, patience=patience, restore_best_weights=True)
    ]
    model.fit(
        train[0],
        train[1],
        validation_data=val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    _, accuracy = model.evaluate(test[0], test[1], verbose=0)
    return accuracy

def main():
    global train, val, test

    seed = 42
    random.set_seed(seed)
    train, val, test = split_preprocessed_data(seed=seed, pre_saved=False)
    n_trials = 50
    warmup = min(n_trials // 10, 10)
    interval_steps = max(warmup * 4 // 10, 1)
    n_startup_trials = n_trials // 2
    study = create_study(
        # sampler=RandomSampler(seed=seed),
        sampler=TPESampler(n_startup_trials=n_startup_trials, seed=seed),
        pruner=MedianPruner(n_startup_trials, warmup, interval_steps),
        study_name='framingham_mlp',
        direction='maximize',
        load_if_exists=True,
    )
    max_workers = cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            executor.submit(study.optimize, objective, n_trials)

    print(study.best_value)
    print(study.best_params)

    visualization.plot_parallel_coordinate(study).show()
    visualization.plot_param_importances(study).show()

if __name__ == "__main__":
    main()
